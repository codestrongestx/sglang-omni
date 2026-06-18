# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.config.schema import EndpointsConfig
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.zonos2.config import Zonos2PipelineConfig
from sglang_omni.models.zonos2.payload_types import (
    DEFAULT_NUM_CODEBOOKS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEAKER_EMBEDDING_DIM,
    ZONOS2_SHAPE_CONTRACT,
    ZONOS2_STAGE_TYPES,
    ZONOS2_TOPOLOGY_STAGES,
)
from sglang_omni.models.zonos2.stages import (
    create_dac_vocoder_executor,
    create_lm_decode_executor,
    create_speaker_embedding_executor,
    create_text_frontend_executor,
)
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.fixtures.pipeline_fakes import FakeMpContext


def test_zonos2_config_declares_plan_step_2_topology() -> None:
    config = Zonos2PipelineConfig(model_path="Zyphra/ZONOS2")
    stages = {stage.name: stage for stage in config.stages}

    assert tuple(stages) == ZONOS2_TOPOLOGY_STAGES
    assert config.resolved_entry_stage == "text_frontend"
    assert config.terminal_stages == ["dac_vocoder"]
    assert config.gpu_placement == {"lm_decode": 0, "dac_vocoder": 0}
    assert config.code2wav_stage() == "dac_vocoder"
    assert config.talker_sglang_role_to_stage() == {"talker": "lm_decode"}
    assert config.supports_uploaded_voice_references() is True

    assert stages["text_frontend"].next == "speaker_embedding"
    assert stages["speaker_embedding"].next == "lm_decode"
    assert stages["lm_decode"].next == "dac_vocoder"
    assert stages["dac_vocoder"].terminal is True
    assert stages["text_frontend"].factory.endswith(
        ".stages.create_text_frontend_executor"
    )
    assert stages["speaker_embedding"].factory.endswith(
        ".stages.create_speaker_embedding_executor"
    )
    assert stages["lm_decode"].factory.endswith(".stages.create_lm_decode_executor")
    assert stages["dac_vocoder"].factory.endswith(
        ".stages.create_dac_vocoder_executor"
    )

    registry_cls = PIPELINE_CONFIG_REGISTRY.get_config("Zonos2ForCausalLM")
    assert registry_cls is Zonos2PipelineConfig


def test_zonos2_runtime_specs_wire_all_inbox_outbox_links(tmp_path) -> None:
    config = Zonos2PipelineConfig(
        model_path="Zyphra/ZONOS2",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
    )
    prep = prepare_pipeline_runtime(config)
    try:
        groups = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prep.stages_cfg,
            name_map=prep.name_map,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )
    finally:
        assert prep.runtime_dir is not None
        prep.runtime_dir.close()

    assert len(groups) == 1
    specs = {spec.stage_name: spec for spec in groups[0].specs}
    assert tuple(specs) == ZONOS2_TOPOLOGY_STAGES
    assert specs["text_frontend"].next_stages == "speaker_embedding"
    assert specs["speaker_embedding"].next_stages == "lm_decode"
    assert specs["lm_decode"].next_stages == "dac_vocoder"
    assert specs["dac_vocoder"].is_terminal is True
    assert specs["text_frontend"].same_process_targets == {"speaker_embedding"}
    assert specs["speaker_embedding"].same_process_targets == {"lm_decode"}
    assert specs["lm_decode"].same_process_targets == {"dac_vocoder"}


def test_zonos2_passthrough_stubs_move_dummy_payload_end_to_end() -> None:
    payload = StagePayload(
        request_id="req-zonos2",
        request=OmniRequest(
            inputs={
                "text": "hello from topology",
                "references": [{"audio_path": "/tmp/ref.wav", "text": "guide"}],
            },
            params={"language": "en_us"},
            metadata={"task": "tts"},
        ),
        data={"probe": "keep-me"},
    )
    schedulers = [
        create_text_frontend_executor("Zyphra/ZONOS2"),
        create_speaker_embedding_executor("Zyphra/ZONOS2"),
        create_lm_decode_executor("Zyphra/ZONOS2"),
        create_dac_vocoder_executor("Zyphra/ZONOS2"),
    ]

    for scheduler in schedulers:
        payload = scheduler._fn(payload)

    data = payload.data
    assert data["probe"] == "keep-me"
    assert data["text"] == "hello from topology"
    assert data["reference_text"] == "guide"
    assert data["visited_stages"] == list(ZONOS2_TOPOLOGY_STAGES)
    assert data["stage_types"] == ZONOS2_STAGE_TYPES
    assert data["shape_contract"] == ZONOS2_SHAPE_CONTRACT
    assert data["prompt_shape"] == [0, DEFAULT_NUM_CODEBOOKS + 1]
    assert data["speaker_embedding_shape"] == [DEFAULT_SPEAKER_EMBEDDING_DIM]
    assert data["codebook_shape"] == [DEFAULT_NUM_CODEBOOKS, 0]
    assert data["sample_rate"] == DEFAULT_SAMPLE_RATE
    assert data["modality"] == "audio"
    assert data["finish_reason"] == "stop"
    assert data["audio_waveform_shape"] == [1]
    assert data["audio_waveform_dtype"] == "float32"
