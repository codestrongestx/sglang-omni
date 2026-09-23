# SPDX-License-Identifier: Apache-2.0
"""Sparse penalties preserve Torch arithmetic across devices and layouts."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from sglang_omni.models.minicpm_o.sampling_kernels import (
    apply_window_penalty,
    apply_window_penalty_torch,
)

CUDA_AVAILABLE = torch.cuda.is_available() and torch.version.hip is None


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
@pytest.mark.parametrize("strided", [False, True])
def test_cpu_window_penalty_updates_only_requested_tokens(
    dtype: torch.dtype, strided: bool
) -> None:
    values = torch.tensor(
        [[2.0, -3.0, 4.0, -5.0], [6.0, -7.0, 8.0, -9.0], [10.0, -11.0, 12.0, -13.0]],
        dtype=dtype,
    )
    logits = values.repeat_interleave(2, dim=1)[:, ::2] if strided else values.clone()
    expected = values.clone()
    expected[2, 0] /= 4
    expected[2, 1] *= 2
    expected[0, 2] /= 0.5
    expected[0, 3] *= 0.25

    apply_window_penalty(logits, [2, 0], [2.0, 0.5], [[0, 0, 1], [2, 3, 3]])

    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_cpu_window_penalty_preserves_nonfinite_and_signed_zero() -> None:
    logits = torch.tensor(
        [[0.0, -0.0, float("inf"), -float("inf"), float("nan"), -2.0]]
    )
    expected = logits.clone()
    expected[0, 5] *= 4

    apply_window_penalty(logits, [0], [2.0], [[0, 1, 2, 3, 4, 5, 5]])

    torch.testing.assert_close(logits, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(torch.signbit(logits), torch.signbit(expected))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="NVIDIA CUDA and Triton are required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("strided", [False, True])
def test_cuda_window_penalty_matches_torch(
    dtype: torch.dtype, batch: int, strided: bool
) -> None:
    histories = [[0, 1, 2, 3, 4, 5], [3], list(range(16))]
    histories.extend([[9] * count + [10] * (16 - count) for count in range(1, 17)])
    for history in histories:
        windows = [history if i % 2 == 0 else history[-(i + 1) :] for i in range(batch)]
        penalties = [[1.05, 1.0, 0.9, 2.0][i % 4] for i in range(batch)]
        rows = list(reversed(range(batch)))
        logits = torch.randn(batch, 1024, dtype=dtype, device="cuda")
        logits[:, :6] = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan"), -0.25],
            dtype=dtype,
            device="cuda",
        )
        logits[:, 9:11] = torch.tensor([-0.75, 0.75], dtype=dtype, device="cuda")
        logits = logits.repeat_interleave(2, dim=1)[:, ::2] if strided else logits
        expected = logits.clone()
        apply_window_penalty_torch(expected, rows, penalties, windows)

        apply_window_penalty(logits, rows, penalties, windows)

        torch.testing.assert_close(logits, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(torch.signbit(logits), torch.signbit(expected))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="NVIDIA CUDA and Triton are required")
def test_cuda_window_penalty_on_independent_streams() -> None:
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    for stream in streams:
        with torch.cuda.stream(stream):
            logits = torch.tensor([[2.0, -3.0]], device="cuda")
            apply_window_penalty(logits, [0], [2.0], [[0, 1, 1]])
            results.append(logits)
    for stream in streams:
        stream.synchronize()
    for logits in results:
        torch.testing.assert_close(
            logits.cpu(), torch.tensor([[1.0, -12.0]]), rtol=0, atol=0
        )


@pytest.mark.skipif(
    not CUDA_AVAILABLE or torch.cuda.device_count() < 2,
    reason="Two NVIDIA CUDA devices and Triton are required",
)
def test_cuda_window_penalty_uses_logits_device() -> None:
    with torch.cuda.device(0):
        logits = torch.tensor([[2.0, -3.0]], device="cuda:1")
        apply_window_penalty(logits, [0], [2.0], [[0, 1, 1]])
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(
            logits.cpu(), torch.tensor([[1.0, -12.0]]), rtol=0, atol=0
        )
