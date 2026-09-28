# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o decode-stage result contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.minicpm_o.merge import build_decode_result
from sglang_omni.proto import OmniRequest, StagePayload

STAGES_PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni"
    / "models"
    / "minicpm_o"
    / "stages.py"
)


class FakeTokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return "hello"


def test_build_decode_result_returns_text_and_usage() -> None:
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": False}),
        data={"engine_outputs": {"thinker": {"output_ids": [1, 2], "is_final": True}}},
    )
    result = build_decode_result(
        payload,
        tokenizer=FakeTokenizer(),
        eos_token_id=None,
        is_streaming=False,
    )
    assert result["text"] == "hello"
    assert result["events"][0]["type"] == "text_final"
    assert result["usage"]["completion_tokens"] == 2


def test_minicpm_stages_do_not_import_qwen3_omni() -> None:
    assert "qwen3_omni" not in STAGES_PATH.read_text()


@pytest.mark.parametrize(
    "sampling_options,logprobs,hidden",
    [
        ({"top_k": 20}, False, False),
        ({"top_k": 1, "frequency_penalty": 0.2}, False, False),
        ({"top_k": 1, "min_new_tokens": 2}, False, False),
        ({"top_k": 1}, True, False),
        ({"top_k": 1}, False, True),
    ],
)
def test_speculative_unsupported_requests_use_ordinary_decode(
    sampling_options: dict[str, int | float], logprobs: bool, hidden: bool
) -> None:
    backend = pytest.importorskip("sglang_omni.models.minicpm_o.speculative")
    # note (Codex): The serving backend is optional for these CPU-only contracts.
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams

    decoder = backend.MiniCPMOSpeculativeDecoder.__new__(
        backend.MiniCPMOSpeculativeDecoder
    )
    decoder.pending = {}
    decoder.statistics = {}
    request = Req.__new__(Req)
    request.rid = "unsupported"
    request.sampling_params = SamplingParams(**sampling_options)
    request.return_logprob = logprobs
    request.grammar = None
    request.custom_logit_processor = None
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        input_ids=torch.tensor([1]),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
        out_cache_loc=torch.tensor([1]),
        seq_lens_sum=1,
        capture_hidden_mode=(
            CaptureHiddenMode.FULL if hidden else CaptureHiddenMode.NULL
        ),
    )
    assert decoder.forward(batch, request) is None
    assert decoder.statistics[request.rid].ordinary_decodes == 1
    assert decoder.statistics[request.rid].rounds == 0


@dataclass(kw_only=True)
class RetainedSlotAllocator:
    allocated_slots: set[int]
    freed_slots: list[int] = field(default_factory=list)

    def free(self, slots: torch.Tensor) -> None:
        slot_ids = slots.tolist()
        assert set(slot_ids).issubset(self.allocated_slots)
        self.allocated_slots.difference_update(slot_ids)
        self.freed_slots.extend(slot_ids)


@pytest.mark.parametrize("consume_count", [0, 1, 2])
def test_speculative_finish_or_abort_frees_retained_slots_once(
    consume_count: int,
) -> None:
    backend = pytest.importorskip("sglang_omni.models.minicpm_o.speculative")
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams

    allocator = RetainedSlotAllocator(allocated_slots={10, 11})
    worker = Mock()
    worker.model_runner.token_to_kv_pool_allocator = allocator
    decoder = backend.MiniCPMOSpeculativeDecoder.__new__(
        backend.MiniCPMOSpeculativeDecoder
    )
    decoder.worker = worker
    decoder.pending = {
        "retained": backend.VerifiedTokens(
            token_ids=torch.tensor([8, 9]),
            cpu_token_ids=[8, 9],
            cache_slots=torch.tensor([10, 11]),
            expected_output_length=1,
            expected_last_token=5,
        )
    }
    decoder.statistics = {"retained": backend.SpeculationStatistics(accepted=2)}
    request = Req.__new__(Req)
    request.rid = "retained"
    request.output_ids = [5]
    request.sampling_params = SamplingParams(top_k=1)
    request.return_logprob = False
    request.grammar = None
    request.custom_logit_processor = None
    emitted: list[int] = []
    for index in range(consume_count):
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([request.output_ids[-1]]),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([index + 1]),
            out_cache_loc=torch.tensor([30 + index]),
            seq_lens_sum=index + 1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        result = decoder.forward(batch, request)
        assert result is not None
        token_id = int(result.next_token_ids.item())
        emitted.append(token_id)
        request.output_ids.append(token_id)
    decoder.reset_request(request.rid)
    decoder.reset_request(request.rid)
    assert emitted == [8, 9][:consume_count]
    assert allocator.allocated_slots == set()
    assert sorted(allocator.freed_slots) == [10, 11]
    assert (
        worker.model_runner.token_to_kv_pool.move_kv_cache.call_count == consume_count
    )


@pytest.mark.parametrize("failure", ["eof", "count", "capacity", "model-path"])
def test_speculative_handshake_failure_closes_connection(
    failure: Literal["eof", "count", "capacity", "model-path"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = pytest.importorskip("sglang_omni.models.minicpm_o.speculative")
    connection = Mock()
    worker = Mock()
    worker.server_args.context_length = 8192
    worker.model_config.model_path = str(tmp_path / "target")
    if failure == "eof":
        connection.recv.side_effect = EOFError
    else:
        connection.recv.return_value = backend.DraftReady(
            draft_tokens=3 if failure == "count" else 4,
            max_sequence_length=4096 if failure == "capacity" else 8192,
            target_model_path=str(
                tmp_path / ("other" if failure == "model-path" else "target")
            ),
        )
    monkeypatch.setattr(backend, "Client", Mock(return_value=connection))
    with pytest.raises(EOFError if failure == "eof" else ValueError):
        backend.MiniCPMOSpeculativeDecoder(
            worker, Mock(), str(tmp_path / "draft.sock"), 4
        )
    connection.close.assert_called_once()
