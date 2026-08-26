# SPDX-License-Identifier: Apache-2.0
"""Speech hidden capture across asynchronous thinker launches."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner._hidden_capture import StaticAuxHiddenCapture
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest


def _scheduler_output() -> SchedulerOutput:
    return SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="text"),
            SchedulerRequest(request_id="audio"),
        ],
        batch_data=SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            reqs=[SimpleNamespace(), SimpleNamespace()],
        ),
    )


def _runner(
    capture: StaticAuxHiddenCapture | None,
) -> tuple[ThinkerModelRunner, SGLangOutputProcessor]:
    processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        model=SimpleNamespace(_omni_aux_hidden_capture=capture),
        should_emit_hidden=lambda request: request.request_id == "audio",
    )
    runner = object.__new__(ThinkerModelRunner)
    runner.output_processor = processor
    runner.model = processor._model
    runner._th_hidden_bufs = None
    runner._th_hidden_slot = 0
    return runner, processor


def _result(stream_hidden: torch.Tensor | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        next_token_ids=torch.tensor([11, 22]),
        logits_output=SimpleNamespace(hidden_states=stream_hidden),
    )


def test_launch_snapshot_survives_later_capture_refresh_and_slot_reuse() -> None:
    live_embed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    live_layer = live_embed + 10
    capture = StaticAuxHiddenCapture(
        buffers=[live_embed, live_layer],
        hook_handles=[],
        max_tokens=2,
    )
    runner, processor = _runner(capture)
    requests = _scheduler_output().requests

    first_result = _result(torch.tensor([[21.0, 22.0], [23.0, 24.0]]))
    first_aux, first_stream = runner._stage_async_hidden_capture(first_result, requests)
    first_result._omni_aux_hidden_states = first_aux
    first_result._omni_stream_hidden_states = first_stream

    live_embed.add_(100)
    live_layer.add_(100)
    second_result = _result()
    runner._stage_async_hidden_capture(second_result, requests)

    outputs = processor.process(first_result, _scheduler_output())
    hidden = outputs["audio"].extra["hidden_states"]
    torch.testing.assert_close(hidden["embed"], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(hidden[24], torch.tensor([13.0, 14.0]))
    torch.testing.assert_close(
        outputs["audio"].extra["stream_hidden_states"],
        torch.tensor([23.0, 24.0]),
    )

    live_embed.add_(100)
    live_layer.add_(100)
    runner._stage_async_hidden_capture(_result(), requests)
    torch.testing.assert_close(hidden["embed"], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(hidden[24], torch.tensor([13.0, 14.0]))


def test_text_only_launch_does_not_read_static_capture() -> None:
    class _UnexpectedCaptureRead:
        def views(self, _num_rows: int) -> None:
            raise AssertionError("text-only launch must not read hidden capture")

    runner, _ = _runner(_UnexpectedCaptureRead())
    requests = [SchedulerRequest(request_id="text")]

    assert runner._stage_async_hidden_capture(_result(), requests) == (None, None)


def test_speech_launch_requires_static_capture() -> None:
    runner, _ = _runner(None)
    requests = [SchedulerRequest(request_id="audio")]

    with pytest.raises(RuntimeError, match="no static auxiliary hidden capture"):
        runner._stage_async_hidden_capture(_result(), requests)


def test_resolve_publishes_launch_owned_capture() -> None:
    runner, _ = _runner(None)
    result = _result()
    launch = SimpleNamespace(
        token_ids=torch.tensor([31, 32]),
        aux_hidden_states=(torch.ones(2, 2), torch.full((2, 2), 2.0)),
        stream_hidden_states=torch.full((2, 2), 3.0),
    )

    runner.post_decode_resolve(
        launch,
        result,
        forward_batch=None,
        schedule_batch=None,
        requests=[object(), object()],
    )

    torch.testing.assert_close(result.next_token_ids, torch.tensor([31, 32]))
    assert result._omni_aux_hidden_states is launch.aux_hidden_states
    assert result._omni_stream_hidden_states is launch.stream_hidden_states
