# SPDX-License-Identifier: Apache-2.0
"""Speech hidden capture across asynchronous thinker launches."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

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
    runner._th_host_bufs = None
    runner._th_slot = 0
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
    first_pointers = tuple(tensor.data_ptr() for tensor in (*first_aux, first_stream))
    first_result._omni_aux_hidden_states = first_aux
    first_result._omni_stream_hidden_states = first_stream

    live_embed.add_(100)
    live_layer.add_(100)
    second_result = _result(torch.tensor([[121.0, 122.0], [123.0, 124.0]]))
    second_aux, second_stream = runner._stage_async_hidden_capture(
        second_result, requests
    )
    second_pointers = tuple(
        tensor.data_ptr() for tensor in (*second_aux, second_stream)
    )
    assert all(
        first != second for first, second in zip(first_pointers, second_pointers)
    )

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
    third_aux, third_stream = runner._stage_async_hidden_capture(
        _result(torch.tensor([[221.0, 222.0], [223.0, 224.0]])), requests
    )
    third_pointers = tuple(tensor.data_ptr() for tensor in (*third_aux, third_stream))
    assert third_pointers == first_pointers
    torch.testing.assert_close(hidden["embed"], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(hidden[24], torch.tensor([13.0, 14.0]))


def test_capture_growth_keeps_pending_snapshot_alive_and_shrink_reuses_capacity() -> (
    None
):
    live_embed = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    live_layer = live_embed + 10
    capture = StaticAuxHiddenCapture(
        buffers=[live_embed, live_layer],
        hook_handles=[],
        max_tokens=4,
    )
    runner, _ = _runner(capture)

    small_requests = [
        SchedulerRequest(request_id="text"),
        SchedulerRequest(request_id="audio"),
    ]
    small_aux, _ = runner._stage_async_hidden_capture(_result(), small_requests)
    small_values = tuple(tensor.clone() for tensor in small_aux)

    live_embed.add_(100)
    live_layer.add_(100)
    grown_requests = [*small_requests, SchedulerRequest(request_id="text-2")]
    grown_aux, _ = runner._stage_async_hidden_capture(_result(), grown_requests)
    grown_pointers = tuple(tensor.data_ptr() for tensor in grown_aux)
    grown_slots = runner._th_hidden_bufs

    for snapshot, expected in zip(small_aux, small_values):
        torch.testing.assert_close(snapshot, expected)
    assert all(tensor.shape[0] == 3 for tensor in grown_aux)
    assert grown_slots is not None
    assert all(buffer.shape[0] == 3 for slot in grown_slots for buffer in slot)

    live_embed.add_(100)
    live_layer.add_(100)
    shrunk_aux, _ = runner._stage_async_hidden_capture(
        _result(), [SchedulerRequest(request_id="audio")]
    )
    assert runner._th_hidden_bufs is grown_slots
    assert all(tensor.shape[0] == 1 for tensor in shrunk_aux)
    assert all(buffer.shape[0] == 3 for slot in grown_slots for buffer in slot)

    reused_grown_aux, _ = runner._stage_async_hidden_capture(_result(), small_requests)
    assert tuple(tensor.data_ptr() for tensor in reused_grown_aux) == grown_pointers


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


def test_post_decode_launch_and_resolve_publish_launch_owned_capture() -> None:
    live_embed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    live_layer = live_embed + 10
    capture = StaticAuxHiddenCapture(
        buffers=[live_embed, live_layer],
        hook_handles=[],
        max_tokens=2,
    )
    runner, processor = _runner(capture)
    runner._async_host_buf = lambda like, n: torch.empty(n, dtype=like.dtype)
    requests = _scheduler_output().requests
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([31, 32]),
        logits_output=SimpleNamespace(
            hidden_states=torch.tensor([[21.0, 22.0], [23.0, 24.0]])
        ),
    )

    launch = runner.post_decode_launch(result, forward_batch=None, requests=requests)
    live_embed.add_(100)
    live_layer.add_(100)

    runner.post_decode_resolve(
        launch,
        result,
        forward_batch=None,
        schedule_batch=None,
        requests=requests,
    )

    torch.testing.assert_close(result.next_token_ids, torch.tensor([31, 32]))
    assert result._omni_aux_hidden_states is launch.aux_hidden_states
    assert result._omni_stream_hidden_states is launch.stream_hidden_states
    outputs = processor.process(result, _scheduler_output())
    hidden = outputs["audio"].extra["hidden_states"]
    torch.testing.assert_close(hidden["embed"], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(hidden[24], torch.tensor([13.0, 14.0]))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_snapshot_event_resolve_clone_and_slot_reuse_are_ordered() -> None:
    device = torch.device("cuda")
    live_embed = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    live_layer = live_embed + 10
    capture = StaticAuxHiddenCapture(
        buffers=[live_embed, live_layer],
        hook_handles=[],
        max_tokens=2,
    )
    runner, processor = _runner(capture)
    requests = _scheduler_output().requests

    first_result = _result(torch.tensor([[21.0, 22.0], [23.0, 24.0]], device=device))
    first_aux, first_stream = runner._stage_async_hidden_capture(first_result, requests)
    first_pointers = tuple(tensor.data_ptr() for tensor in (*first_aux, first_stream))
    completion = torch.cuda.Event()
    completion.record()

    live_embed.add_(100)
    live_layer.add_(100)
    second_aux, second_stream = runner._stage_async_hidden_capture(
        _result(torch.tensor([[121.0, 122.0], [123.0, 124.0]], device=device)),
        requests,
    )
    completion.synchronize()

    first_result._omni_aux_hidden_states = first_aux
    first_result._omni_stream_hidden_states = first_stream
    outputs = processor.process(first_result, _scheduler_output())
    resolved_embed = outputs["audio"].extra["hidden_states"]["embed"]
    resolved_layer = outputs["audio"].extra["hidden_states"][24]

    live_embed.add_(100)
    live_layer.add_(100)
    third_aux, third_stream = runner._stage_async_hidden_capture(
        _result(torch.tensor([[221.0, 222.0], [223.0, 224.0]], device=device)),
        requests,
    )
    torch.cuda.synchronize()

    second_pointers = tuple(
        tensor.data_ptr() for tensor in (*second_aux, second_stream)
    )
    third_pointers = tuple(tensor.data_ptr() for tensor in (*third_aux, third_stream))
    assert all(
        first != second for first, second in zip(first_pointers, second_pointers)
    )
    assert third_pointers == first_pointers
    torch.testing.assert_close(resolved_embed, torch.tensor([3.0, 4.0], device=device))
    torch.testing.assert_close(
        resolved_layer, torch.tensor([13.0, 14.0], device=device)
    )
