# SPDX-License-Identifier: Apache-2.0
"""Repetition penalties preserve per-codebook sampling scores."""

import pytest
import torch

from sglang_omni.models.zonos2.sampler import apply_repetition_penalty


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("penalty_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_repetition_penalty_matches_distinct_history_tokens(
    dtype: torch.dtype, penalty_dtype: torch.dtype, strided: bool, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    logits = torch.linspace(-4, 4, 48, dtype=dtype, device=device).reshape(3, 2, 8)
    if strided:
        logits = logits.transpose(1, 2).contiguous().transpose(1, 2)
    logits[0, 0, 0] = -float("inf")
    logits[0, 1, 7] = float("inf")
    logits[1, 0, 2] = 0.0
    history = torch.tensor(
        [
            [[-1, 0, 0, 7, 8, 99], [7, 7, -2, 0, 8, 99]],
            [[2, 2, 2, 3, -1, 8], [-1, -2, 8, 9, 99, -1]],
            [[0, 1, 2, 3, 4, 5], [7, 6, 5, 4, 3, 2]],
        ],
        device=device,
    )
    penalties = torch.tensor([1.2, 2.0, 0.5], dtype=penalty_dtype, device=device)
    original = logits.clone()
    strengths = penalties[:, None, None].clamp(min=1.0)
    adjusted = torch.where(logits > 0, logits / strengths, logits * strengths)
    expected = logits.to(adjusted.dtype).clone()
    for row in range(3):
        for channel in range(2):
            for token in set(history[row, channel].tolist()):
                if 0 <= token < logits.shape[-1]:
                    expected[row, channel, token] = adjusted[row, channel, token]
    actual = apply_repetition_penalty(logits, history, penalties)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits, original, rtol=0, atol=0)


@pytest.mark.parametrize("history", [None, torch.empty(2, 3, 0, dtype=torch.long)])
def test_repetition_penalty_without_history(history: torch.Tensor | None) -> None:
    logits = torch.randn(2, 3, 8)
    actual = apply_repetition_penalty(logits, history, torch.tensor([1.2, 2.0]))
    torch.testing.assert_close(actual, logits, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_repetition_penalty_cuda_graph_replays_changed_history() -> None:
    logits = torch.randn(2, 3, 16, device="cuda")
    history = torch.full((2, 3, 5), -1, dtype=torch.long, device="cuda")
    penalties = torch.tensor([1.2, 2.0], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            apply_repetition_penalty(logits, history, penalties)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = apply_repetition_penalty(logits, history, penalties)
    for token in (0, 15, -1):
        history.fill_(token)
        logits.add_(0.1)
        expected = logits.clone()
        if token >= 0:
            scores = logits[..., token]
            strengths = penalties[:, None]
            expected[..., token] = torch.where(
                scores > 0, scores / strengths, scores * strengths
            )
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
