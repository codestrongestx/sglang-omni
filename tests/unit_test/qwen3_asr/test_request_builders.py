# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import sglang_omni.models.qwen3_asr.request_builders as request_builders
from sglang_omni.models.qwen3_asr.audio_lengths import (
    qwen3_asr_audio_token_lengths,
    qwen3_asr_num_audio_tokens,
)
from sglang_omni.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASRProcessor
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    make_qwen3_asr_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTokenizer:
    eos_token_id = 2
    vocab_size = 1000

    def __init__(self) -> None:
        self.encode_calls: list[str] = []
        self.decode_calls: list[dict] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|audio_pad|>"
        return 42

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        self.encode_calls.append(text)
        assert text == "<asr_text>"
        return [100, 101]

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        audio_pad_count = text.count("<|audio_pad|>")
        return SimpleNamespace(input_ids=[11] + [42] * audio_pad_count + [12, 13, 14])

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        self.decode_calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        pieces = {
            10: "language English",
            100: "<asr_text>",
            101: "",
            20: " leading",
            21: "\u00a0middle",
            22: "  ",
            99: "<|endoftext|>",
        }
        text = "".join(pieces[token_id] for token_id in token_ids)
        if skip_special_tokens:
            text = text.replace("<|endoftext|>", "")
        return text


def test_qwen3_asr_audio_token_length_formula_is_shared() -> None:
    lengths = torch.tensor([0, 1, 99, 100, 101, 3000], dtype=torch.long)
    expected = torch.tensor([0, 1, 13, 13, 14, 390], dtype=torch.long)

    processor = object.__new__(Qwen3ASRProcessor)

    assert torch.equal(qwen3_asr_audio_token_lengths(lengths), expected)
    assert torch.equal(processor._get_feat_extract_output_lengths(lengths), expected)
    assert qwen3_asr_num_audio_tokens(3000) == 390


def test_qwen3_asr_request_builder_records_inclusive_audio_offsets(monkeypatch) -> None:
    num_mel_frames = 101
    num_audio_tokens = qwen3_asr_num_audio_tokens(num_mel_frames)
    feature_extractor = lambda *args, **kwargs: SimpleNamespace(
        input_features=torch.zeros((1, 128, 3000)),
        attention_mask=torch.ones((1, num_mel_frames), dtype=torch.long),
    )
    monkeypatch.setattr(
        request_builders,
        "load_audio",
        lambda source: np.zeros(1600, dtype=np.float32),
    )
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor,
    )
    payload = StagePayload(
        request_id="req-asr",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    data = request_builder(payload)

    audio_item = data.req.multimodal_inputs.mm_items[0]
    start, end = audio_item.offsets[0]
    assert audio_item.feature_attention_mask.shape == (1, num_mel_frames)
    assert end - start + 1 == num_audio_tokens
    assert data.prompt_token_ids[start : end + 1] == (
        [audio_item.pad_value] * num_audio_tokens
    )


def test_qwen3_asr_request_builder_caches_uploaded_audio_preprocessing(
    monkeypatch,
) -> None:
    load_calls = 0
    extract_calls = 0
    num_mel_frames = 101

    def fake_load_audio(source):
        nonlocal load_calls
        load_calls += 1
        assert source == b"same-wav"
        return np.zeros(1600, dtype=np.float32)

    def fake_feature_extractor(*args, **kwargs):
        nonlocal extract_calls
        extract_calls += 1
        return SimpleNamespace(
            input_features=torch.full((1, 128, 3000), float(extract_calls)),
            attention_mask=torch.ones((1, num_mel_frames), dtype=torch.long),
        )

    monkeypatch.setattr(request_builders, "load_audio", fake_load_audio)
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=fake_feature_extractor,
    )
    payload_1 = StagePayload(
        request_id="req-asr-1",
        request=OmniRequest(inputs={"audio_bytes": b"same-wav"}),
        data={},
    )
    payload_2 = StagePayload(
        request_id="req-asr-2",
        request=OmniRequest(inputs={"audio_bytes": b"same-wav"}),
        data={},
    )

    data_1 = request_builder(payload_1)
    audio_item_1 = data_1.req.multimodal_inputs.mm_items[0]
    audio_item_1.feature.fill_(99.0)
    audio_item_1.feature_attention_mask.zero_()
    data_2 = request_builder(payload_2)

    assert load_calls == 1
    assert extract_calls == 1
    feature_1 = audio_item_1.feature
    feature_2 = data_2.req.multimodal_inputs.mm_items[0].feature
    mask_2 = data_2.req.multimodal_inputs.mm_items[0].feature_attention_mask
    assert torch.equal(feature_1, torch.full((1, 128, 3000), 99.0))
    assert torch.equal(feature_2, torch.ones((1, 128, 3000)))
    assert torch.equal(mask_2, torch.ones((1, num_mel_frames), dtype=torch.long))
    assert feature_1 is not feature_2


def test_qwen3_asr_preprocess_cache_normalizes_mutable_audio_source(
    monkeypatch,
) -> None:
    load_sources = []
    extract_calls = 0

    def fake_load_audio(source):
        load_sources.append(source)
        assert isinstance(source, bytes)
        return np.zeros(1600, dtype=np.float32)

    def fake_feature_extractor(*args, **kwargs):
        nonlocal extract_calls
        extract_calls += 1
        return SimpleNamespace(
            input_features=torch.full((1, 128, 3000), float(extract_calls)),
            attention_mask=torch.ones((1, 101), dtype=torch.long),
        )

    monkeypatch.setattr(request_builders, "load_audio", fake_load_audio)
    request_builder, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=fake_feature_extractor,
    )
    audio = bytearray(b"same-wav")
    payload = StagePayload(
        request_id="req-asr",
        request=OmniRequest(inputs={"audio_bytes": audio}),
        data={},
    )

    data_1 = request_builder(payload)
    audio[:] = b"otherwav"
    data_2 = request_builder(payload)

    assert load_sources == [b"same-wav", b"otherwav"]
    assert extract_calls == 2
    feature_1 = data_1.req.multimodal_inputs.mm_items[0].feature
    feature_2 = data_2.req.multimodal_inputs.mm_items[0].feature
    assert torch.equal(feature_1, torch.full((1, 128, 3000), 1.0))
    assert torch.equal(feature_2, torch.full((1, 128, 3000), 2.0))


def test_qwen3_asr_preprocess_cache_is_scoped_to_adapter(monkeypatch) -> None:
    load_calls = 0

    def fake_load_audio(source):
        nonlocal load_calls
        load_calls += 1
        assert source == b"same-wav"
        return np.zeros(1600, dtype=np.float32)

    def feature_extractor_with_value(value: float):
        def fake_feature_extractor(*args, **kwargs):
            return SimpleNamespace(
                input_features=torch.full((1, 128, 3000), value),
                attention_mask=torch.ones((1, 101), dtype=torch.long),
            )

        return fake_feature_extractor

    monkeypatch.setattr(request_builders, "load_audio", fake_load_audio)
    request_builder_1, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor_with_value(1.0),
    )
    request_builder_2, _ = make_qwen3_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        max_new_tokens=32,
        feature_extractor=feature_extractor_with_value(2.0),
    )
    payload = StagePayload(
        request_id="req-asr",
        request=OmniRequest(inputs={"audio_bytes": b"same-wav"}),
        data={},
    )

    data_1 = request_builder_1(payload)
    data_2 = request_builder_2(payload)

    assert load_calls == 2
    feature_1 = data_1.req.multimodal_inputs.mm_items[0].feature
    feature_2 = data_2.req.multimodal_inputs.mm_items[0].feature
    assert torch.equal(feature_1, torch.full((1, 128, 3000), 1.0))
    assert torch.equal(feature_2, torch.full((1, 128, 3000), 2.0))


def test_qwen3_asr_result_adapter_decodes_without_text_round_trip() -> None:
    tokenizer = _FakeTokenizer()
    _, result_adapter = make_qwen3_asr_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=32,
        feature_extractor=object(),
    )
    payload = StagePayload(
        request_id="req-asr",
        request=OmniRequest(inputs={}),
        data={},
    )
    data = Qwen3ASRRequestData(
        output_ids=[10, 100, 101, 20, 21, 22, 99],
        stage_payload=payload,
        language="en",
        audio_duration_s=1.25,
    )

    result = result_adapter(data)

    assert result.data["text"] == " leading\u00a0middle  "
    assert tokenizer.encode_calls == ["<asr_text>"]
    assert tokenizer.decode_calls[-1] == {
        "token_ids": [20, 21, 22, 99],
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
