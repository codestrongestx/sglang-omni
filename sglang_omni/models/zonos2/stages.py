# SPDX-License-Identifier: Apache-2.0
"""Passthrough executors for the ZONOS2 plan-step-2 topology skeleton."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.zonos2.payload_types import (
    DEFAULT_CODEBOOK_SIZE,
    DEFAULT_NUM_CODEBOOKS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEAKER_EMBEDDING_DIM,
    Zonos2TTSState,
    infer_shape,
    prompt_frame_width,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

DEFAULT_MAX_CONCURRENCY = 16


def load_state(payload: StagePayload) -> Zonos2TTSState:
    return Zonos2TTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: Zonos2TTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def create_text_frontend_executor(
    model_path: str,
    *,
    num_codebooks: int = DEFAULT_NUM_CODEBOOKS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> SimpleScheduler:
    del model_path

    def _text_frontend(payload: StagePayload) -> StagePayload:
        state = _state_from_request(payload, num_codebooks=num_codebooks)
        state.mark("text_frontend")
        if state.prompt_shape is None:
            state.prompt_shape = (0, prompt_frame_width(state.num_codebooks))
        return store_state(payload, state)

    return SimpleScheduler(_text_frontend, max_concurrency=max_concurrency)


def create_speaker_embedding_executor(
    model_path: str,
    *,
    speaker_embedding_dim: int = DEFAULT_SPEAKER_EMBEDDING_DIM,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> SimpleScheduler:
    del model_path

    def _speaker_embedding(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        state.speaker_embedding_dim = int(speaker_embedding_dim)
        state.mark("speaker_embedding")
        if state.speaker_embedding_shape is None:
            state.speaker_embedding_shape = (state.speaker_embedding_dim,)
        if state.speaker_embedding is not None:
            state.speaker_embedding_shape = infer_shape(state.speaker_embedding)
        return store_state(payload, state)

    return SimpleScheduler(_speaker_embedding, max_concurrency=max_concurrency)


def create_lm_decode_executor(
    model_path: str,
    *,
    num_codebooks: int = DEFAULT_NUM_CODEBOOKS,
    codebook_size: int = DEFAULT_CODEBOOK_SIZE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> SimpleScheduler:
    del model_path

    def _lm_decode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        state.num_codebooks = int(num_codebooks)
        state.codebook_size = int(codebook_size)
        state.mark("lm_decode")
        if state.codebook_shape is None:
            state.codebook_shape = (state.num_codebooks, 0)
        if state.codebook_tokens is not None:
            state.codebook_shape = infer_shape(state.codebook_tokens)
        return store_state(payload, state)

    return SimpleScheduler(_lm_decode, max_concurrency=max_concurrency)


def create_dac_vocoder_executor(
    model_path: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    emit_dummy_audio: bool = True,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> SimpleScheduler:
    del model_path

    def _dac_vocoder(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        state.sample_rate = int(sample_rate)
        state.mark("dac_vocoder")
        data = state.to_dict()
        data.setdefault("finish_reason", "stop")
        data.setdefault("modality", "audio")
        if emit_dummy_audio and not _has_audio_payload(data):
            audio = state.audio_samples if state.audio_samples is not None else [0.0]
            data.update(
                audio_waveform_payload(
                    audio,
                    sample_rate=state.sample_rate,
                    modality="audio",
                    source_hint="ZONOS2 topology stub",
                )
            )
        payload.data = data
        return payload

    return SimpleScheduler(_dac_vocoder, max_concurrency=max_concurrency)


def _state_from_request(
    payload: StagePayload,
    *,
    num_codebooks: int,
) -> Zonos2TTSState:
    state = load_state(payload)
    state.num_codebooks = int(num_codebooks)

    inputs = payload.request.inputs
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params") or {}

    if isinstance(inputs, str):
        state.text = state.text or inputs
    elif isinstance(inputs, dict):
        state.text = state.text or str(
            inputs.get("text") or inputs.get("input") or inputs.get("prompt") or ""
        )
        state.reference_audio = (
            state.reference_audio
            or inputs.get("reference_audio")
            or inputs.get("ref_audio")
        )
        state.reference_text = state.reference_text or inputs.get("reference_text")
        references = inputs.get("references")
        if (
            state.reference_audio is None
            and isinstance(references, list)
            and references
        ):
            first_ref = references[0]
            if isinstance(first_ref, dict):
                state.reference_audio = first_ref
                state.reference_text = state.reference_text or first_ref.get("text")
    elif isinstance(inputs, list):
        state.text = state.text or _text_from_messages(inputs)

    if not state.text and tts_params.get("text"):
        state.text = str(tts_params["text"])
    if tts_params.get("ref_audio") is not None and state.reference_audio is None:
        state.reference_audio = tts_params.get("ref_audio")
    if tts_params.get("ref_text") is not None and state.reference_text is None:
        state.reference_text = tts_params.get("ref_text")

    language = params.get("language") or tts_params.get("language")
    if language:
        state.language = str(language)
    if "text_normalization" in params:
        state.text_normalization = bool(params["text_normalization"])
    return state


def _text_from_messages(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
    return "\n".join(parts)


def _has_audio_payload(data: dict[str, Any]) -> bool:
    return any(key in data for key in ("audio_waveform", "audio_data", "audio"))
