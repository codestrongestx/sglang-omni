# SPDX-License-Identifier: Apache-2.0
"""Payload contracts for the ZONOS2 topology skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_NUM_CODEBOOKS = 9
DEFAULT_CODEBOOK_SIZE = 1024
DEFAULT_SPEAKER_EMBEDDING_DIM = 2048
DEFAULT_SPEAKER_LDA_DIM = 1024
DEFAULT_SAMPLE_RATE = 44100
TEXT_LANE_WIDTH = 1
ZONOS2_TOPOLOGY_STAGES = (
    "text_frontend",
    "speaker_embedding",
    "lm_decode",
    "dac_vocoder",
)
ZONOS2_STAGE_TYPES = {
    "text_frontend": "cpu_text_frontend",
    "speaker_embedding": "speaker_embedding_encoder",
    "lm_decode": "autoregressive_multi_codebook_lm",
    "dac_vocoder": "dac_vocoder",
}


def prompt_frame_width(num_codebooks: int = DEFAULT_NUM_CODEBOOKS) -> int:
    return int(num_codebooks) + TEXT_LANE_WIDTH


ZONOS2_SHAPE_CONTRACT: dict[str, dict[str, str]] = {
    "text_frontend": {
        "input": "text/ref-audio request payload",
        "output": "conditioning prompt frames int64 [seq, 10]",
    },
    "speaker_embedding": {
        "input": "reference audio or cached speaker artifact",
        "output": "raw speaker embedding float32 [2048]; LDA projects to [1024]",
    },
    "lm_decode": {
        "input": "prompt frames [seq, 10] plus raw speaker embedding [2048]",
        "output": "DAC codebook token grid int64 [9, frames] plus eos_frame",
    },
    "dac_vocoder": {
        "input": "DAC codebook token grid int64 [9, frames]",
        "output": "PCM waveform float32 [samples] at 44100 Hz",
    },
}

_KNOWN_KEYS = {
    "text",
    "language",
    "text_normalization",
    "reference_audio",
    "reference_text",
    "prompt_tokens",
    "prompt_shape",
    "speaker_embedding",
    "speaker_embedding_shape",
    "codebook_tokens",
    "codebook_shape",
    "eos_frame",
    "audio_samples",
    "sample_rate",
    "num_codebooks",
    "codebook_size",
    "speaker_embedding_dim",
    "speaker_lda_dim",
    "visited_stages",
    "stage_types",
    "shape_contract",
}


@dataclass
class Zonos2TTSState:
    """Per-request data passed through the four ZONOS2 skeleton components.

    The current implementation is intentionally a topology probe. It preserves
    handoff names for the real tokenizer, speaker, LM, sampler, and DAC outputs.
    """

    text: str = ""
    language: str = "en_us"
    text_normalization: bool = True
    reference_audio: Any | None = None
    reference_text: str | None = None
    prompt_tokens: Any | None = None
    prompt_shape: tuple[int, ...] | None = None
    speaker_embedding: Any | None = None
    speaker_embedding_shape: tuple[int, ...] | None = None
    codebook_tokens: Any | None = None
    codebook_shape: tuple[int, ...] | None = None
    eos_frame: int | None = None
    audio_samples: Any | None = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    num_codebooks: int = DEFAULT_NUM_CODEBOOKS
    codebook_size: int = DEFAULT_CODEBOOK_SIZE
    speaker_embedding_dim: int = DEFAULT_SPEAKER_EMBEDDING_DIM
    speaker_lda_dim: int = DEFAULT_SPEAKER_LDA_DIM
    visited_stages: list[str] = field(default_factory=list)
    stage_types: dict[str, str] = field(
        default_factory=lambda: dict(ZONOS2_STAGE_TYPES)
    )
    shape_contract: dict[str, dict[str, str]] = field(
        default_factory=lambda: dict(ZONOS2_SHAPE_CONTRACT)
    )
    extra: dict[str, Any] = field(default_factory=dict)

    def mark(self, stage_name: str) -> None:
        if stage_name not in self.visited_stages:
            self.visited_stages.append(stage_name)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra)
        data.update(
            {
                "text": self.text,
                "language": self.language,
                "text_normalization": self.text_normalization,
                "sample_rate": int(self.sample_rate),
                "num_codebooks": int(self.num_codebooks),
                "codebook_size": int(self.codebook_size),
                "speaker_embedding_dim": int(self.speaker_embedding_dim),
                "speaker_lda_dim": int(self.speaker_lda_dim),
                "visited_stages": list(self.visited_stages),
                "stage_types": dict(self.stage_types),
                "shape_contract": dict(self.shape_contract),
            }
        )
        if self.reference_audio is not None:
            data["reference_audio"] = self.reference_audio
        if self.reference_text is not None:
            data["reference_text"] = self.reference_text
        if self.prompt_tokens is not None:
            data["prompt_tokens"] = _to_payload_value(self.prompt_tokens)
        if self.prompt_shape is not None:
            data["prompt_shape"] = list(self.prompt_shape)
        if self.speaker_embedding is not None:
            data["speaker_embedding"] = _to_payload_value(self.speaker_embedding)
        if self.speaker_embedding_shape is not None:
            data["speaker_embedding_shape"] = list(self.speaker_embedding_shape)
        if self.codebook_tokens is not None:
            data["codebook_tokens"] = _to_payload_value(self.codebook_tokens)
        if self.codebook_shape is not None:
            data["codebook_shape"] = list(self.codebook_shape)
        if self.eos_frame is not None:
            data["eos_frame"] = int(self.eos_frame)
        if self.audio_samples is not None:
            data["audio_samples"] = _to_payload_value(self.audio_samples)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "Zonos2TTSState":
        if not isinstance(data, dict):
            data = {}
        extra = {key: value for key, value in data.items() if key not in _KNOWN_KEYS}
        return cls(
            text=str(data.get("text") or ""),
            language=str(data.get("language") or "en_us"),
            text_normalization=bool(data.get("text_normalization", True)),
            reference_audio=data.get("reference_audio"),
            reference_text=data.get("reference_text"),
            prompt_tokens=data.get("prompt_tokens"),
            prompt_shape=_tuple_or_none(data.get("prompt_shape")),
            speaker_embedding=data.get("speaker_embedding"),
            speaker_embedding_shape=_tuple_or_none(
                data.get("speaker_embedding_shape")
            ),
            codebook_tokens=data.get("codebook_tokens"),
            codebook_shape=_tuple_or_none(data.get("codebook_shape")),
            eos_frame=(
                int(data["eos_frame"]) if data.get("eos_frame") is not None else None
            ),
            audio_samples=data.get("audio_samples"),
            sample_rate=int(
                data.get("sample_rate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE
            ),
            num_codebooks=int(
                data.get("num_codebooks", DEFAULT_NUM_CODEBOOKS)
                or DEFAULT_NUM_CODEBOOKS
            ),
            codebook_size=int(
                data.get("codebook_size", DEFAULT_CODEBOOK_SIZE)
                or DEFAULT_CODEBOOK_SIZE
            ),
            speaker_embedding_dim=int(
                data.get("speaker_embedding_dim", DEFAULT_SPEAKER_EMBEDDING_DIM)
                or DEFAULT_SPEAKER_EMBEDDING_DIM
            ),
            speaker_lda_dim=int(
                data.get("speaker_lda_dim", DEFAULT_SPEAKER_LDA_DIM)
                or DEFAULT_SPEAKER_LDA_DIM
            ),
            visited_stages=list(data.get("visited_stages") or []),
            stage_types=dict(data.get("stage_types") or ZONOS2_STAGE_TYPES),
            shape_contract=dict(data.get("shape_contract") or ZONOS2_SHAPE_CONTRACT),
            extra=extra,
        )


def infer_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if not isinstance(value, (list, tuple)):
        return None
    dims: list[int] = []
    cursor = value
    while isinstance(cursor, (list, tuple)):
        dims.append(len(cursor))
        if not cursor:
            break
        cursor = cursor[0]
    return tuple(dims)


def _tuple_or_none(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(dim) for dim in value)


def _to_payload_value(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        return value.tolist()
    return value
