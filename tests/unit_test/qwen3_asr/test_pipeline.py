# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import sglang_omni.models.qwen3_asr.stages as qwen3_asr_stages
from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_asr.stages import create_sglang_qwen3_asr_executor
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_qwen3_asr_config_uses_batched_stage_with_32_running_requests() -> None:
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_qwen3_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert config.stages[0].factory_args["max_running_requests"] == 32
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3ASRForConditionalGeneration")
        is Qwen3ASRPipelineConfig
    )


def test_qwen3_asr_stage_default_allows_32_running_requests() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["max_running_requests"].default == 32


def test_qwen3_asr_stage_default_uses_auto_static_kv_budget() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["mem_fraction_static"].default is None


def test_qwen3_asr_stage_default_disables_multimodal_embedding_cache() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0


def test_qwen3_asr_stage_default_disables_torch_compile() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["enable_torch_compile"].default is False


def test_qwen3_asr_stage_request_build_workers_default_to_sync() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["request_build_max_workers"].default == 1
    assert signature.parameters["request_build_max_pending"].default is None
    assert signature.parameters["request_build_max_backlog"].default is None


def test_qwen3_asr_stage_forwards_request_build_config(monkeypatch) -> None:
    scheduler_kwargs: dict = {}
    adapter_kwargs: dict = {}

    monkeypatch.setattr(qwen3_asr_stages, "get_visible_gpu_sm_version", lambda _gpu: 90)
    monkeypatch.setattr(
        qwen3_asr_stages.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        qwen3_asr_stages.AutoFeatureExtractor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(nb_max_frames=3000)),
    )

    def fake_build_sglang_server_args(model_path, context_length, **kwargs):
        del model_path, context_length
        return SimpleNamespace(
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            disable_overlap_schedule=kwargs["disable_overlap_schedule"],
        )

    class FakeWorker:
        def __init__(self):
            self.model_runner = SimpleNamespace(
                model=object(),
                init_device_graphs=lambda: None,
            )

    monkeypatch.setattr(
        qwen3_asr_stages,
        "build_sglang_server_args",
        fake_build_sglang_server_args,
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "create_sglang_infrastructure",
        lambda *args, **kwargs: (
            FakeWorker(),
            object(),
            object(),
            object(),
            object(),
            object(),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "init_mm_embedding_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def fake_make_qwen3_asr_scheduler_adapters(**kwargs):
        adapter_kwargs.update(kwargs)
        return lambda payload: payload, lambda data: data

    monkeypatch.setattr(
        qwen3_asr_stages,
        "make_qwen3_asr_scheduler_adapters",
        fake_make_qwen3_asr_scheduler_adapters,
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "ModelRunner",
        lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )
    monkeypatch.setattr(
        qwen3_asr_stages,
        "OmniScheduler",
        lambda **kwargs: scheduler_kwargs.update(kwargs) or SimpleNamespace(**kwargs),
    )

    create_sglang_qwen3_asr_executor(
        "Qwen/Qwen3-ASR-1.7B",
        request_build_max_workers=8,
        request_build_max_pending=6,
        request_build_max_backlog=24,
    )

    assert scheduler_kwargs["request_build_max_workers"] == 8
    assert scheduler_kwargs["request_build_max_pending"] == 6
    assert scheduler_kwargs["request_build_max_backlog"] == 24
    assert set(adapter_kwargs) == {"tokenizer", "feature_extractor", "max_new_tokens"}
