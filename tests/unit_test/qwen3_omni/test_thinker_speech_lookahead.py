# SPDX-License-Identifier: Apache-2.0
"""Speech hidden-capture ownership across thinker lookahead launches."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    Qwen3OmniThinkerForCausalLM,
)
from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest


def _runner() -> ThinkerModelRunner:
    runner = object.__new__(ThinkerModelRunner)
    runner._capture_hidden_layers = [0, 24]
    runner._capture_hidden_width = 2
    runner._th_hidden_bufs = None
    runner._th_hidden_slot = 0
    return runner


def _result(stream_hidden: torch.Tensor) -> SimpleNamespace:
    # Mirrors the base-runner mailbox contract: every batch result carries the
    # capture slots stamped to None before any post-decode hook runs.
    return SimpleNamespace(
        next_token_ids=torch.tensor([11, 22]),
        logits_output=SimpleNamespace(hidden_states=stream_hidden),
        _captured_aux_hidden_states=None,
        _captured_stream_hidden_states=None,
    )


def _packed_result(*hidden_parts: torch.Tensor) -> SimpleNamespace:
    return _result(torch.cat(hidden_parts, dim=-1))


def test_thinker_model_publishes_aux_and_stream_on_logits_output() -> None:
    calls: list[dict] = []
    logits_result = object()

    def logits_processor(
        input_ids,
        hidden_states,
        lm_head,
        forward_batch,
        *,
        aux_hidden_states,
    ):
        calls.append(
            {
                "input_ids": input_ids,
                "hidden_states": hidden_states,
                "lm_head": lm_head,
                "forward_batch": forward_batch,
                "aux_hidden_states": aux_hidden_states,
            }
        )
        return logits_result

    lm_head = object()
    model = SimpleNamespace(logits_processor=logits_processor, lm_head=lm_head)
    input_ids = torch.tensor([1, 2])
    stream_hidden = torch.tensor([[21.0, 22.0], [23.0, 24.0]])
    aux_hidden = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[11.0, 12.0], [13.0, 14.0]]),
    ]
    forward_batch = object()

    result = Qwen3OmniThinkerForCausalLM.process_hidden_states(
        model,
        input_ids=input_ids,
        hidden_states=(stream_hidden, aux_hidden),
        forward_batch=forward_batch,
    )

    assert result is logits_result
    assert len(calls) == 1
    call = calls[0]
    assert call["input_ids"] is input_ids
    assert call["hidden_states"] is stream_hidden
    assert call["lm_head"] is lm_head
    assert call["forward_batch"] is forward_batch
    assert all(
        actual is expected
        for actual, expected in zip(
            call["aux_hidden_states"],
            [*aux_hidden, stream_hidden],
            strict=True,
        )
    )


def test_speech_batches_request_last_prefill_and_full_decode_hidden_output() -> None:
    runner = _runner()
    runner._should_capture_hidden = lambda request: request.request_id == "audio"
    text_request = SimpleNamespace(request_id="text")
    audio_request = SimpleNamespace(request_id="audio")

    assert (
        runner.requested_capture_hidden_mode_decode(None, [text_request]).name == "NULL"
    )
    assert (
        runner.requested_capture_hidden_mode_decode(None, [audio_request]).name
        == "FULL"
    )
    assert (
        runner.requested_capture_hidden_mode_prefill(None, [audio_request]).name
        == "LAST"
    )


def test_text_only_lookahead_skips_hidden_snapshot() -> None:
    runner = _runner()
    runner._should_capture_hidden = lambda request: request.request_id == "audio"
    runner._async_host_buf = lambda like, n: torch.empty(n, dtype=like.dtype)
    result = _packed_result(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[11.0, 12.0], [13.0, 14.0]]),
        torch.tensor([[21.0, 22.0], [23.0, 24.0]]),
    )

    runner.post_decode_launch(
        result,
        forward_batch=None,
        requests=[
            SimpleNamespace(request_id="text-1"),
            SimpleNamespace(request_id="text-2"),
        ],
    )

    assert result._captured_aux_hidden_states is None
    assert result._captured_stream_hidden_states is None
    assert runner._th_hidden_bufs is None


def test_speech_hidden_capture_pingpongs_across_lookahead_launches() -> None:
    runner = _runner()
    first_aux = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[11.0, 12.0], [13.0, 14.0]]),
    ]
    first_stream = torch.tensor([[21.0, 22.0], [23.0, 24.0]])
    first_result = _packed_result(*first_aux, first_stream)

    runner._stage_async_hidden_capture(first_result)

    second_aux = [
        torch.tensor([[101.0, 102.0], [103.0, 104.0]]),
        torch.tensor([[111.0, 112.0], [113.0, 114.0]]),
    ]
    second_stream = torch.tensor([[121.0, 122.0], [123.0, 124.0]])
    second_result = _packed_result(*second_aux, second_stream)

    runner._stage_async_hidden_capture(second_result)

    assert torch.equal(first_result._captured_aux_hidden_states[0], first_aux[0])
    assert torch.equal(first_result._captured_aux_hidden_states[1], first_aux[1])
    assert torch.equal(first_result._captured_stream_hidden_states, first_stream)
    assert torch.equal(second_result._captured_aux_hidden_states[0], second_aux[0])
    assert (
        first_result._captured_aux_hidden_states[0].data_ptr()
        != second_result._captured_aux_hidden_states[0].data_ptr()
    )


def test_speech_hidden_capture_uses_the_replayed_graph_output() -> None:
    runner = _runner()
    graph_a_aux = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[11.0, 12.0], [13.0, 14.0]]),
    ]
    graph_a_stream = torch.tensor([[21.0, 22.0], [23.0, 24.0]])
    graph_b_aux = [
        torch.tensor([[101.0, 102.0], [103.0, 104.0]]),
        torch.tensor([[111.0, 112.0], [113.0, 114.0]]),
    ]
    graph_b_stream = torch.tensor([[121.0, 122.0], [123.0, 124.0]])
    graph_a_output = torch.cat([*graph_a_aux, graph_a_stream], dim=-1)
    graph_b_output = torch.cat([*graph_b_aux, graph_b_stream], dim=-1)

    graph_a_result = _result(graph_a_output)
    runner._stage_async_hidden_capture(graph_a_result)

    assert torch.equal(graph_a_result._captured_aux_hidden_states[0], graph_a_aux[0])
    assert torch.equal(graph_a_result._captured_stream_hidden_states, graph_a_stream)

    graph_b_result = _result(graph_b_output)
    runner._stage_async_hidden_capture(graph_b_result)
    assert torch.equal(graph_b_result._captured_aux_hidden_states[0], graph_b_aux[0])
    assert torch.equal(graph_b_result._captured_stream_hidden_states, graph_b_stream)

    # Replaying graph A mutates and returns graph A's own output allocation.
    graph_a_output.add_(1000)
    graph_a_replay = _result(graph_a_output)
    runner._stage_async_hidden_capture(graph_a_replay)

    assert torch.equal(
        graph_a_replay._captured_aux_hidden_states[0],
        torch.tensor([[1001.0, 1002.0], [1003.0, 1004.0]]),
    )
    assert torch.equal(
        graph_a_replay._captured_stream_hidden_states,
        torch.tensor([[1021.0, 1022.0], [1023.0, 1024.0]]),
    )


def test_speech_raw_hidden_capture_pingpongs_when_aux_capture_is_missing() -> None:
    runner = _runner()
    graph_owned_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    first_result = _result(graph_owned_hidden)

    runner._stage_async_hidden_capture(first_result)
    graph_owned_hidden.add_(100)

    second_result = _result(torch.tensor([[11.0, 12.0], [13.0, 14.0]]))
    runner._stage_async_hidden_capture(second_result)

    assert first_result._captured_aux_hidden_states is None
    assert torch.equal(
        first_result._captured_stream_hidden_states,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert (
        first_result._captured_stream_hidden_states.data_ptr()
        != second_result._captured_stream_hidden_states.data_ptr()
    )


def test_output_processor_consumes_result_owned_capture_not_later_launch() -> None:
    runner = _runner()
    aux = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[11.0, 12.0], [13.0, 14.0]]),
    ]
    stream = torch.tensor([[21.0, 22.0], [23.0, 24.0]])
    result = _packed_result(*aux, stream)
    runner._stage_async_hidden_capture(result)

    # Launch(N+1) replays the same graph before this step resolves: the packed
    # logits-output buffer now holds the NEXT step's values. Emitted extras
    # must come from the launch-owned snapshot, not the live buffer.
    result.logits_output.hidden_states = torch.cat(
        [part + 100.0 for part in [*aux, stream]], dim=-1
    )
    output_processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        capture_hidden_width=2,
        should_emit_hidden=lambda request: request.request_id == "audio",
    )
    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="text"),
            SchedulerRequest(request_id="audio"),
        ],
        batch_data=SimpleNamespace(
            reqs=[
                SimpleNamespace(extend_input_len=1),
                SimpleNamespace(extend_input_len=1),
            ]
        ),
    )

    outputs = output_processor.process(result, scheduler_output)

    assert outputs["text"].extra is None
    audio_extra = outputs["audio"].extra
    assert torch.equal(audio_extra["hidden_states"]["embed"], torch.tensor([3.0, 4.0]))
    assert torch.equal(audio_extra["hidden_states"][24], torch.tensor([13.0, 14.0]))
    assert torch.equal(audio_extra["stream_hidden_states"], torch.tensor([23.0, 24.0]))


def test_output_processor_reads_graph_owned_packed_capture() -> None:
    output_processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        capture_hidden_width=2,
        should_emit_hidden=lambda request: request.request_id == "audio",
    )
    result = _packed_result(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[11.0, 12.0], [13.0, 14.0]]),
        torch.tensor([[21.0, 22.0], [23.0, 24.0]]),
    )
    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="text"),
            SchedulerRequest(request_id="audio"),
        ],
        batch_data=SimpleNamespace(
            reqs=[
                SimpleNamespace(extend_input_len=1),
                SimpleNamespace(extend_input_len=1),
            ]
        ),
    )

    outputs = output_processor.process(result, scheduler_output)

    assert outputs["text"].extra is None
    audio_extra = outputs["audio"].extra
    assert torch.equal(audio_extra["hidden_states"]["embed"], torch.tensor([3.0, 4.0]))
    assert torch.equal(audio_extra["hidden_states"][24], torch.tensor([13.0, 14.0]))
    assert torch.equal(audio_extra["stream_hidden_states"], torch.tensor([23.0, 24.0]))


def test_output_processor_reads_last_only_capture_after_multi_token_prefill() -> None:
    output_processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        capture_hidden_width=2,
        should_emit_hidden=lambda request: True,
    )
    result = _packed_result(
        torch.tensor([[7.0, 8.0]]),
        torch.tensor([[17.0, 18.0]]),
        torch.tensor([[27.0, 28.0]]),
    )
    scheduler_output = SchedulerOutput(
        requests=[SchedulerRequest(request_id="audio")],
        batch_data=SimpleNamespace(reqs=[SimpleNamespace(extend_input_len=5)]),
    )

    output = output_processor.process(result, scheduler_output)["audio"]

    assert torch.equal(output.extra["hidden_states"]["embed"], torch.tensor([7.0, 8.0]))
    assert torch.equal(output.extra["hidden_states"][24], torch.tensor([17.0, 18.0]))
    assert torch.equal(output.extra["stream_hidden_states"], torch.tensor([27.0, 28.0]))


def test_output_processor_maps_last_only_capture_for_mixed_prefill() -> None:
    output_processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        capture_hidden_width=2,
        should_emit_hidden=lambda request: request.request_id == "audio",
    )
    result = _packed_result(
        torch.tensor([[1.0, 2.0], [7.0, 8.0]]),
        torch.tensor([[11.0, 12.0], [17.0, 18.0]]),
        torch.tensor([[21.0, 22.0], [27.0, 28.0]]),
    )
    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="text"),
            SchedulerRequest(request_id="audio"),
        ],
        batch_data=SimpleNamespace(
            reqs=[
                SimpleNamespace(extend_input_len=3),
                SimpleNamespace(extend_input_len=5),
            ]
        ),
    )

    outputs = output_processor.process(result, scheduler_output)

    assert outputs["text"].extra is None
    audio_extra = outputs["audio"].extra
    assert torch.equal(audio_extra["hidden_states"]["embed"], torch.tensor([7.0, 8.0]))
    assert torch.equal(audio_extra["hidden_states"][24], torch.tensor([17.0, 18.0]))
    assert torch.equal(audio_extra["stream_hidden_states"], torch.tensor([27.0, 28.0]))


def test_output_processor_uses_owned_raw_snapshot_when_aux_capture_is_missing() -> None:
    runner = _runner()
    graph_owned_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = _result(graph_owned_hidden)
    runner._stage_async_hidden_capture(result)
    graph_owned_hidden.add_(100)

    output_processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        should_emit_hidden=lambda request: request.request_id == "audio",
    )
    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="text"),
            SchedulerRequest(request_id="audio"),
        ],
        batch_data=SimpleNamespace(
            reqs=[
                SimpleNamespace(extend_input_len=1),
                SimpleNamespace(extend_input_len=1),
            ]
        ),
    )

    outputs = output_processor.process(result, scheduler_output)

    assert outputs["text"].extra is None
    emitted_hidden = outputs["audio"].extra["hidden_states"]
    assert torch.equal(
        emitted_hidden,
        torch.tensor([3.0, 4.0]),
    )

    # The third launch reuses the first ping-pong slot after this payload has
    # entered the async stream queue. The emitted request must own its slice.
    runner._stage_async_hidden_capture(
        _result(torch.tensor([[11.0, 12.0], [13.0, 14.0]]))
    )
    runner._stage_async_hidden_capture(
        _result(torch.tensor([[101.0, 102.0], [103.0, 104.0]]))
    )

    assert torch.equal(emitted_hidden, torch.tensor([3.0, 4.0]))


def test_output_processor_uses_launch_rows_after_live_batch_shrinks() -> None:
    output_processor = SGLangOutputProcessor(
        capture_hidden=True,
        capture_hidden_layers=[0, 24],
        should_emit_hidden=lambda request: request.request_id == "audio",
    )
    result = _result(torch.tensor([[21.0, 22.0], [23.0, 24.0], [25.0, 26.0]]))
    result._captured_aux_hidden_states = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]),
    )
    result._captured_stream_hidden_states = torch.tensor(
        [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0]]
    )
    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="finished"),
            SchedulerRequest(request_id="retracted"),
            SchedulerRequest(request_id="audio"),
        ],
        # The live ScheduleBatch has already dropped two launch-time rows.
        batch_data=SimpleNamespace(reqs=[SimpleNamespace(extend_input_len=1)]),
    )

    outputs = output_processor.process(result, scheduler_output)

    audio_extra = outputs["audio"].extra
    assert torch.equal(audio_extra["hidden_states"]["embed"], torch.tensor([5.0, 6.0]))
    assert torch.equal(audio_extra["hidden_states"][24], torch.tensor([15.0, 16.0]))
    assert torch.equal(audio_extra["stream_hidden_states"], torch.tensor([25.0, 26.0]))


def test_lookahead_profile_event_records_capture_and_step(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        omni_scheduler_module,
        "_get_event_recorder",
        lambda: SimpleNamespace(is_active=lambda: True),
    )
    monkeypatch.setattr(
        omni_scheduler_module,
        "_emit_event",
        lambda **event: events.append(event),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.is_entry_rank = True
    scheduler_output = SchedulerOutput(
        requests=[
            SchedulerRequest(request_id="speech-1"),
            SchedulerRequest(request_id="speech-2"),
        ],
        batch_data=None,
        step_id=17,
    )

    scheduler._emit_lookahead_events(
        scheduler_output,
        "scheduler_lookahead_resolve",
        hidden_capture=True,
        event_ready=False,
    )

    assert [event["request_id"] for event in events] == ["speech-1", "speech-2"]
    assert all(event["event_name"] == "scheduler_lookahead_resolve" for event in events)


def test_unconfigured_capture_ignores_audio_default_and_requests_null_mode() -> None:
    """A text-only deployment installs no capture layers. Requests that default
    to audio output (missing output_modalities) must still keep NULL capture
    and never reach the hidden-snapshot path. Regression: capture gating used
    to read only per-request metadata, so such a batch requested FULL decode
    capture (mismatching the NULL-captured CUDA graphs and disabling replay)
    and the launch snapshot dereferenced capture state that text deployments
    never create (AttributeError, failing every lookahead batch)."""
    runner = object.__new__(ThinkerModelRunner)
    runner._capture_hidden_layers = None
    runner._capture_hidden_width = None
    runner._should_capture_hidden = lambda request: True  # modalities default
    runner._th_hidden_bufs = None
    runner._th_hidden_slot = 0
    runner._async_host_buf = lambda like, n: torch.empty(n, dtype=like.dtype)
    requests = [
        SimpleNamespace(request_id="text-1"),
        SimpleNamespace(request_id="text-2"),
    ]

    assert runner.requested_capture_hidden_mode_decode(None, requests).name == "NULL"
    assert runner.requested_capture_hidden_mode_prefill(None, requests).name == "NULL"

    result = _result(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    runner.post_decode_launch(result, forward_batch=None, requests=requests)

    assert result._captured_aux_hidden_states is None
    assert result._captured_stream_hidden_states is None
    assert runner._th_hidden_bufs is None
    assert all(
        event["metadata"]
        == {
            "step_id": 17,
            "batch_size": 2,
            "hidden_capture": True,
            "event_ready": False,
        }
        for event in events
    )
