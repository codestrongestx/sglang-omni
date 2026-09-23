# SPDX-License-Identifier: Apache-2.0
"""Windowed codec penalties preserve Torch arithmetic and fallback behavior."""

from __future__ import annotations

import builtins
import runpy
from typing import Literal

import pytest
import torch

from sglang_omni.models.minicpm_o import sampling_kernels
from sglang_omni.models.minicpm_o.sampling_kernels import apply_window_penalty

# note (Codex): delay queued copies so pinned allocations churn before GPU consumption.
GPU_QUEUE_DELAY_CYCLES = 20_000_000


CUDA_AVAILABLE = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and sampling_kernels.window_penalty_kernel is not None
)


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
def test_cuda_window_penalty_matches_torch(
    dtype: torch.dtype, batch: int, monkeypatch: pytest.MonkeyPatch
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
        expected = logits.clone()
        with monkeypatch.context() as fallback:
            fallback.setattr(sampling_kernels, "window_penalty_kernel", None)
            apply_window_penalty(expected, rows, penalties, windows)

        apply_window_penalty(logits, rows, penalties, windows)

        torch.testing.assert_close(logits, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(torch.signbit(logits), torch.signbit(expected))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="NVIDIA CUDA and Triton are required")
@pytest.mark.parametrize("fallback", ["missing_triton", "float64", "strided"])
def test_cuda_window_penalty_fallback(
    fallback: Literal["missing_triton", "float64", "strided"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dtype = torch.float64 if fallback == "float64" else torch.float32
    logits = torch.tensor([[2.0, -3.0, 4.0, -5.0]], device="cuda", dtype=dtype)
    if fallback == "missing_triton":
        monkeypatch.setattr(sampling_kernels, "window_penalty_kernel", None)
    elif fallback == "strided":
        logits = logits.repeat_interleave(2, dim=1)[:, ::2]
    else:
        assert dtype == torch.float64

    apply_window_penalty(logits, [0], [2.0], [[0, 1, 1]])

    expected = torch.tensor([[1.0, -12.0, 4.0, -5.0]], device="cuda", dtype=dtype)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="NVIDIA CUDA and Triton are required")
def test_cuda_window_penalty_preserves_queued_metadata() -> None:
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results: list[tuple[int, torch.Tensor]] = []
    for stream in streams:
        with torch.cuda.stream(stream):
            warmup = torch.ones(1, 32, device="cuda")
            apply_window_penalty(warmup, [0], [2.0], [[0, 1, 1]])
        stream.synchronize()

    for stream in streams:
        with torch.cuda.stream(stream):
            logits = [torch.ones(1, 32, device="cuda") for _ in range(16)]
            torch.cuda._sleep(GPU_QUEUE_DELAY_CYCLES)
            for token, scores in enumerate(logits):
                apply_window_penalty(
                    scores, [0], [2.0], [[token, token + 1, token + 1]]
                )
                torch.empty((2, 4), dtype=torch.int32, pin_memory=True).fill_(-1)
                results.append((token, scores))
    for stream in streams:
        stream.synchronize()
    for token, scores in results:
        expected = torch.ones(1, 32)
        expected[0, token] = 0.5
        expected[0, token + 1] = 0.25
        torch.testing.assert_close(scores.cpu(), expected, rtol=0, atol=0)


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


@pytest.mark.parametrize(
    "failure",
    [
        ModuleNotFoundError("Triton is not installed", name="triton"),
        ModuleNotFoundError("Broken Triton dependency", name="triton_dependency"),
        ImportError("Broken Triton installation"),
    ],
)
def test_optional_triton_import_only_allows_missing_package(
    failure: ImportError, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def import_with_failure(name: str, *args: object, **kwargs: object) -> object:
        if name == "triton":
            raise failure
        else:
            return original_import(name, *args, **kwargs)

    with monkeypatch.context() as imports:
        imports.setattr(builtins, "__import__", import_with_failure)
        if failure.name == "triton":
            module = runpy.run_path(sampling_kernels.__file__)
            logits = torch.tensor([[2.0, -3.0]])
            module["apply_window_penalty"](logits, [0], [2.0], [[0, 1, 1]])
            torch.testing.assert_close(logits, torch.tensor([[1.0, -12.0]]))
        else:
            with pytest.raises(type(failure)) as raised:
                runpy.run_path(sampling_kernels.__file__)
            assert raised.value is failure
