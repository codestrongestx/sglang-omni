# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Qwen3-ASR.

Unlike Whisper (encoder-decoder, features consumed inside the model forward),
Qwen3-ASR is a Qwen3 causal LM that ingests audio as multimodal embeddings:
the prompt contains an ``<|audio_pad|>`` placeholder repeated once per audio
token, and the model's ``general_mm_embed_routine`` scatters the encoder output
into those positions. So request_builder must:
  * extract mel features (WhisperFeatureExtractor) + attention mask,
  * compute how many audio tokens the encoder will emit,
  * build the chat prompt with that many ``<|audio_pad|>`` tokens,
  * hand the features over as a ``MultimodalDataItem``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

from .audio_lengths import qwen3_asr_num_audio_tokens

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000

_AUDIO_START = "<|audio_start|>"
_AUDIO_PAD = "<|audio_pad|>"
_AUDIO_END = "<|audio_end|>"
_ASR_TEXT = "<asr_text>"
_PREPROCESS_CACHE_MAX_ENTRIES = 128


@dataclass
class Qwen3ASRRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str = "en"
    engine_start_s: float = 0.0


@dataclass
class _PreprocessedAudio:
    audio_duration_s: float
    fingerprint: str
    features: torch.Tensor
    feature_attention_mask: torch.Tensor
    num_mel_frames: int
    num_audio_tokens: int


@dataclass
class _ProcessorContext:
    tokenizer: Any
    feature_extractor: Any
    audio_pad_token_id: int
    eos_token_id: int
    vocab_size: int


_PreprocessCache = OrderedDict[tuple[str, int, str], _PreprocessedAudio]


def _clone_preprocessed_audio(entry: _PreprocessedAudio) -> _PreprocessedAudio:
    return _PreprocessedAudio(
        audio_duration_s=entry.audio_duration_s,
        fingerprint=entry.fingerprint,
        features=entry.features.clone(),
        feature_attention_mask=entry.feature_attention_mask.clone(),
        num_mel_frames=entry.num_mel_frames,
        num_audio_tokens=entry.num_audio_tokens,
    )


def _audio_cache_key(source: Any) -> tuple[str, int, str] | None:
    if not isinstance(source, bytes):
        return None
    digest = hashlib.blake2b(source, digest_size=16).hexdigest()
    return ("bytes", len(source), digest)


def _normalize_cacheable_audio_source(
    source: Any,
) -> tuple[Any, tuple[str, int, str] | None]:
    if isinstance(source, memoryview):
        source = source.tobytes()
    elif isinstance(source, bytearray):
        source = bytes(source)
    return source, _audio_cache_key(source)


def _get_cached_preprocessed_audio(
    cache: _PreprocessCache,
    lock: threading.Lock,
    key: tuple[str, int, str] | None,
) -> _PreprocessedAudio | None:
    if key is None:
        return None
    with lock:
        entry = cache.get(key)
        if entry is None:
            return None
        cache.move_to_end(key)
        return _clone_preprocessed_audio(entry)


def _put_cached_preprocessed_audio(
    cache: _PreprocessCache,
    lock: threading.Lock,
    key: tuple[str, int, str] | None,
    entry: _PreprocessedAudio,
) -> None:
    if key is None:
        return
    cached = _PreprocessedAudio(
        audio_duration_s=entry.audio_duration_s,
        fingerprint=entry.fingerprint,
        features=entry.features.detach().cpu().clone(),
        feature_attention_mask=entry.feature_attention_mask.detach().cpu().clone(),
        num_mel_frames=entry.num_mel_frames,
        num_audio_tokens=entry.num_audio_tokens,
    )
    with lock:
        cache[key] = cached
        cache.move_to_end(key)
        while len(cache) > _PREPROCESS_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)


def _audio_source_from_payload(payload: StagePayload) -> Any:
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("audio_bytes", "bytes", "file"):
            value = inputs.get(key)
            if value is not None:
                return value
        for key in ("audio_path", "path", "url"):
            value = inputs.get(key)
            if value is not None:
                return value
    return inputs


def load_audio(source: Any) -> np.ndarray:
    import torchaudio

    if isinstance(source, memoryview):
        source = source.tobytes()
    if isinstance(source, bytearray):
        source = bytes(source)

    if isinstance(source, bytes):
        audio, sample_rate = torchaudio.load(io.BytesIO(source))
    elif isinstance(source, str):
        audio, sample_rate = torchaudio.load(source)
    else:
        raise ValueError(f"Unsupported Qwen3-ASR audio input: {type(source).__name__}")

    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0).to(torch.float32)
    if sample_rate != _SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, _SAMPLE_RATE)
    return audio.cpu().numpy()


def _audio_fingerprint(audio: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(audio, dtype=np.float32)
    return hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()


def _audio_fingerprint_int(fingerprint: str) -> int:
    return int(fingerprint[:16], 16)


def _decode_token_ids(
    tokenizer: Any, token_ids: list[int], *, skip_special_tokens: bool
) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern:
        return None
    limit = len(values) - len(pattern) + 1
    for start in range(max(limit, 0)):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def _encode_literal(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    else:
        input_ids = encoded["input_ids"]
    return list(input_ids)


def make_qwen3_asr_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens: int,
    feature_extractor: Any = None,
    tokenizer_factory: Callable[[], Any] | None = None,
    feature_extractor_factory: Callable[[], Any] | None = None,
) -> tuple[
    Callable[[StagePayload], Qwen3ASRRequestData], Callable[[Any], StagePayload]
]:
    if feature_extractor is None and feature_extractor_factory is None:
        raise ValueError("Qwen3-ASR processor is missing a feature_extractor")

    asr_text_token_ids = _encode_literal(tokenizer, _ASR_TEXT)
    request_build_processor_local = threading.local()
    preprocess_cache: _PreprocessCache = OrderedDict()
    preprocess_cache_lock = threading.Lock()

    def _make_processor_context(
        local_tokenizer: Any,
        local_feature_extractor: Any,
    ) -> _ProcessorContext:
        return _ProcessorContext(
            tokenizer=local_tokenizer,
            feature_extractor=local_feature_extractor,
            audio_pad_token_id=int(local_tokenizer.convert_tokens_to_ids(_AUDIO_PAD)),
            eos_token_id=int(local_tokenizer.eos_token_id),
            vocab_size=int(local_tokenizer.vocab_size),
        )

    shared_processor_context = _make_processor_context(tokenizer, feature_extractor)

    def _request_build_processor_context() -> _ProcessorContext:
        if tokenizer_factory is None and feature_extractor_factory is None:
            return shared_processor_context
        context = getattr(request_build_processor_local, "context", None)
        if context is None:
            local_tokenizer = tokenizer_factory() if tokenizer_factory else tokenizer
            local_feature_extractor = (
                feature_extractor_factory()
                if feature_extractor_factory
                else feature_extractor
            )
            context = _make_processor_context(local_tokenizer, local_feature_extractor)
            request_build_processor_local.context = context
        return context

    def _build_prompt_ids(
        prompt_tokenizer: Any, num_audio_tokens: int, language: str
    ) -> list[int]:
        prompt = (
            f"<|im_start|>user\n"
            f"{_AUDIO_START}{_AUDIO_PAD * num_audio_tokens}{_AUDIO_END}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        # Qwen3-ASR needs a forced prefix "language <Lang><asr_text>" on the
        # assistant turn; the model then generates only the transcription after
        # <asr_text>. Without it the (small) model emits the language tag then
        # stops. Upstream qwen_asr does the same (_build_text_prompt).
        prompt = prompt + f"language {language}<asr_text>"
        return prompt_tokenizer(prompt, add_special_tokens=False).input_ids

    def request_builder(payload: StagePayload) -> Qwen3ASRRequestData:
        processor_context = _request_build_processor_context()
        local_tokenizer = processor_context.tokenizer
        local_feature_extractor = processor_context.feature_extractor
        params = payload.request.params or {}
        audio_source = _audio_source_from_payload(payload)
        audio_source, cache_key = _normalize_cacheable_audio_source(audio_source)
        preprocessed = _get_cached_preprocessed_audio(
            preprocess_cache, preprocess_cache_lock, cache_key
        )
        if preprocessed is None:
            audio = load_audio(audio_source)
            audio_duration_s = float(len(audio) / _SAMPLE_RATE)
            fingerprint = _audio_fingerprint(audio)

            extracted = local_feature_extractor(
                audio,
                sampling_rate=_SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            features = extracted.input_features  # 128, 3000
            feature_attention_mask = getattr(extracted, "attention_mask", None)
            if feature_attention_mask is None:
                # WhisperFeatureExtractor normally returns one; fall back to all-valid.
                feature_attention_mask = torch.ones(
                    (features.shape[0], features.shape[-1]), dtype=torch.long
                )
            # Keep the full padded mel; the model's get_audio_feature uses the mask
            # to select valid frames. Its no-mask branch transposes wrong, so the
            # mask path must be taken.
            num_mel_frames = int(feature_attention_mask.sum().item())
            num_audio_tokens = int(qwen3_asr_num_audio_tokens(num_mel_frames))
            preprocessed = _PreprocessedAudio(
                audio_duration_s=audio_duration_s,
                fingerprint=fingerprint,
                features=features,
                feature_attention_mask=feature_attention_mask,
                num_mel_frames=num_mel_frames,
                num_audio_tokens=num_audio_tokens,
            )
            _put_cached_preprocessed_audio(
                preprocess_cache, preprocess_cache_lock, cache_key, preprocessed
            )

        audio_duration_s = preprocessed.audio_duration_s
        fingerprint = preprocessed.fingerprint
        features = preprocessed.features
        feature_attention_mask = preprocessed.feature_attention_mask
        num_mel_frames = preprocessed.num_mel_frames
        num_audio_tokens = preprocessed.num_audio_tokens
        logger.debug(
            f"[qwen3-asr] mel_frames={num_mel_frames} "
            f"num_audio_tokens={num_audio_tokens} feat_shape={tuple(features.shape)}"
        )

        lang_raw = str(params.get("language") or "en").strip().lower()
        forced_language = {"zh": "Chinese", "cn": "Chinese"}.get(
            lang_raw, "Chinese" if lang_raw.startswith("zh") else "English"
        )
        input_ids = _build_prompt_ids(
            local_tokenizer, num_audio_tokens, forced_language
        )

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=_audio_fingerprint_int(fingerprint),
            feature=features,
            model_specific_data={
                "feature_attention_mask": feature_attention_mask,
            },
        )
        # general_mm_embed_routine locates audio positions by matching each
        # item's pad_value against input_ids. The omni scheduler does not run
        # pad_input_ids for us, so compute the pad_value, replace the
        # <|audio_pad|> placeholders with it, and record the placeholder span as
        # item.offsets. SGLang treats offsets as inclusive.
        audio_item.set_pad_value()
        audio_start = input_ids.index(processor_context.audio_pad_token_id)
        input_ids = [
            audio_item.pad_value if tok == processor_context.audio_pad_token_id else tok
            for tok in input_ids
        ]
        audio_item.offsets = [(audio_start, audio_start + num_audio_tokens - 1)]

        mm_inputs = MultimodalInputs(
            mm_items=[audio_item],
            num_image_tokens=num_audio_tokens,
        )
        mm_inputs.audio_token_id = processor_context.audio_pad_token_id
        # sglang indexes mm_input.mrope_positions[:, start:end] during prefill and
        # does not compute a default, so we must supply it. Qwen3-ASR's MRoPE is
        # degenerate for ASR (all 3 sections share the text position), i.e. plain
        # 1-D positions broadcast to shape [3, seq_len].
        seq_len = len(input_ids)
        positions = torch.arange(seq_len, dtype=torch.long)
        mm_inputs.mrope_positions = positions.unsqueeze(0).expand(3, -1).clone()
        mm_inputs.mrope_position_delta = torch.tensor([0], dtype=torch.long)

        temperature = float(params.get("temperature") or 0.0)
        if temperature == 0.0:
            # Qwen3-ASR degenerates under pure-greedy (emits only the language
            # tag then EOS); upstream uses 0.01 near-greedy.
            temperature = 0.01
        request_max_new_tokens = int(params.get("max_new_tokens") or max_new_tokens)
        logger.debug(
            f"[qwen3-asr] sampling temp={temperature} "
            f"max_new_tokens={request_max_new_tokens} params={dict(params)}"
        )
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[processor_context.eos_token_id],
        )
        sampling_params.normalize(tokenizer=None)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=processor_context.vocab_size,
            extra_key=fingerprint,
        )
        req.multimodal_inputs = mm_inputs
        req._codec_suppress_tokens = None

        return Qwen3ASRRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            audio_duration_s=audio_duration_s,
            language=str(params.get("language") or "en"),
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: Qwen3ASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        # Keep the marker handling at token level. Byte-level BPE decode->encode
        # is not an identity transform for all whitespace/Unicode transcripts.
        raw = _decode_token_ids(tokenizer, output_ids, skip_special_tokens=False)
        logger.debug(
            f"[qwen3-asr] n_out={len(output_ids)} ids={output_ids[:40]} " f"raw={raw!r}"
        )
        asr_text_idx = _find_subsequence(output_ids, asr_text_token_ids)
        transcript_ids = (
            output_ids[asr_text_idx + len(asr_text_token_ids) :]
            if asr_text_idx is not None
            else output_ids
        )
        text = _decode_token_ids(tokenizer, transcript_ids, skip_special_tokens=True)
        engine_time_s = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "language": data.language,
                "duration_s": data.audio_duration_s,
                "asr_latency_s": engine_time_s,
                "usage": {"engine_time_s": engine_time_s},
                "modality": "text",
            },
        )

    return request_builder, result_adapter


__all__ = [
    "Qwen3ASRRequestData",
    "load_audio",
    "make_qwen3_asr_scheduler_adapters",
]
