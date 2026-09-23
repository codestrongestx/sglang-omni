# SPDX-License-Identifier: Apache-2.0
"""Sparse Torch and fused NVIDIA penalties for recent MiniCPM-o codec tokens."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import torch
import triton
import triton.language as tl

# note (MayDomine): the checkpoint penalizes only the most recent 16 codec tokens.
REP_PENALTY_WINDOW = 16
PENALTY_BLOCK_SIZE = 128


@lru_cache(maxsize=32)
def penalty_powers(
    penalties: tuple[float, ...], stream: torch.cuda.Stream
) -> torch.Tensor:
    """Cache powers per stream to keep asynchronous initialization ordered."""
    with torch.cuda.stream(stream):
        bases = torch.tensor(penalties, dtype=torch.float32, device=stream.device)[
            :, None
        ]
        counts = torch.arange(
            REP_PENALTY_WINDOW + 1, dtype=torch.float32, device=stream.device
        )[None, :]
        return bases**counts


@triton.jit
def window_penalty_kernel(
    logits: tl.tensor,
    entries: tl.tensor,
    alphas: tl.tensor,
    count: int,
    row_stride: tl.constexpr,
    token_stride: tl.constexpr,
    block: tl.constexpr,
    power_stride: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < count
    row = tl.load(entries + offsets * 4, mask, other=0)
    token = tl.load(entries + offsets * 4 + 1, mask, other=0)
    frequency = tl.load(entries + offsets * 4 + 2, mask, other=0)
    penalty_row = tl.load(entries + offsets * 4 + 3, mask, other=0)
    alpha = tl.load(alphas + penalty_row * power_stride + frequency, mask, other=1)
    pointer = logits + row * row_stride + token * token_stride
    score = tl.load(pointer, mask, other=0).to(tl.float32)
    updated = tl.where(score < 0, score * alpha, tl.div_rn(score, alpha))
    tl.store(pointer, updated, mask)


def apply_window_penalty_torch(
    logits: torch.Tensor,
    rows: list[int],
    penalties: list[float],
    windows: list[list[int]],
) -> None:
    """Update only distinct tokens using FP32 Torch penalty arithmetic."""
    device = logits.device
    entries = [
        (row, token, count, penalty)
        for row, penalty, window in zip(rows, penalties, windows)
        for token, count in Counter(window).items()
    ]
    row_ids, token_ids, counts, token_penalties = zip(*entries)
    row_indices = torch.tensor(row_ids, dtype=torch.long, device=device)
    token_indices = torch.tensor(token_ids, dtype=torch.long, device=device)
    alphas = torch.tensor(
        token_penalties, dtype=torch.float32, device=device
    ) ** torch.tensor(counts, dtype=torch.float32, device=device)
    scores = logits[row_indices, token_indices].to(torch.float32)
    penalized = torch.where(scores < 0, scores * alphas, scores / alphas)
    logits[row_indices, token_indices] = penalized.to(logits.dtype)


def apply_window_penalty(
    logits: torch.Tensor,
    rows: list[int],
    penalties: list[float],
    windows: list[list[int]],
) -> None:
    """Apply penalties to filtered, nonempty windows; fuse on NVIDIA GPUs."""
    if not logits.is_cuda or torch.version.hip is not None:
        apply_window_penalty_torch(logits, rows, penalties, windows)
    else:
        assert logits.dtype in (torch.float32, torch.float16, torch.bfloat16)
        with torch.cuda.device(logits.device):
            entries = [
                (row, token, frequency, penalty_row)
                for penalty_row, (row, window) in enumerate(
                    zip(rows, windows, strict=True)
                )
                for token, frequency in Counter(window).items()
            ]
            metadata = torch.tensor(entries, dtype=torch.int32, device=logits.device)
            alphas = penalty_powers(
                tuple(penalties), torch.cuda.current_stream(logits.device)
            )
            window_penalty_kernel[(triton.cdiv(len(entries), PENALTY_BLOCK_SIZE),)](
                logits,
                metadata,
                alphas,
                len(entries),
                logits.stride(0),
                logits.stride(1),
                PENALTY_BLOCK_SIZE,
                REP_PENALTY_WINDOW + 1,
                enable_fp_fusion=False,
            )
