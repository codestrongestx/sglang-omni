# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Zyphra ZONOS2 TTS.

This declares the ZONOS2 handoffs and uses passthrough executors until the
real text frontend, speaker encoder, LM sampler, and DAC vocoder are wired in.
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.zonos2.payload_types import (
    DEFAULT_CODEBOOK_SIZE,
    DEFAULT_NUM_CODEBOOKS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEAKER_EMBEDDING_DIM,
    DEFAULT_SPEAKER_LDA_DIM,
)
from sglang_omni.models.zonos2.stages import DEFAULT_MAX_CONCURRENCY

_PKG = "sglang_omni.models.zonos2"


class Zonos2PipelineConfig(PipelineConfig):
    """4-component ZONOS2 DAG: text frontend -> speaker -> LM -> DAC."""

    architecture: ClassVar[str] = "Zonos2ForCausalLM"
    architecture_aliases: ClassVar[tuple[str, ...]] = ("ZONOS2", "Zyphra/ZONOS2")

    @classmethod
    def talker_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"talker": "lm_decode"}

    @classmethod
    def code2wav_stage(cls) -> str | None:
        return "dac_vocoder"

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="text_frontend",
            process="pipeline",
            factory=f"{_PKG}.stages.create_text_frontend_executor",
            factory_args={
                "num_codebooks": DEFAULT_NUM_CODEBOOKS,
                "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            },
            next="speaker_embedding",
        ),
        StageConfig(
            name="speaker_embedding",
            process="pipeline",
            factory=f"{_PKG}.stages.create_speaker_embedding_executor",
            factory_args={
                "speaker_embedding_dim": DEFAULT_SPEAKER_EMBEDDING_DIM,
                "speaker_lda_dim": DEFAULT_SPEAKER_LDA_DIM,
                "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            },
            next="lm_decode",
        ),
        StageConfig(
            name="lm_decode",
            process="pipeline",
            factory=f"{_PKG}.stages.create_lm_decode_executor",
            factory_args={
                "num_codebooks": DEFAULT_NUM_CODEBOOKS,
                "codebook_size": DEFAULT_CODEBOOK_SIZE,
                "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            },
            gpu=0,
            next="dac_vocoder",
        ),
        StageConfig(
            name="dac_vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_dac_vocoder_executor",
            factory_args={
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "emit_dummy_audio": True,
                "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            },
            gpu=0,
            terminal=True,
        ),
    ]

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = Zonos2PipelineConfig
