# SPDX-License-Identifier: Apache-2.0
"""Sparse windowed repetition penalties for MiniCPM-o codec sampling."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# note (MayDomine): the checkpoint penalizes only the most recent 16 codec tokens.
REP_PENALTY_WINDOW = 16


@lru_cache(maxsize=32)
def penalty_powers(
    penalties: tuple[float, ...], stream: torch.cuda.Stream
) -> torch.Tensor:
    # note (Codex): stream-local tables avoid reading powers queued on another stream.
    with torch.cuda.stream(stream):
        bases = torch.tensor(penalties, dtype=torch.float32, device=stream.device)[
            :, None
        ]
        counts = torch.arange(
            REP_PENALTY_WINDOW + 1, dtype=torch.float32, device=stream.device
        )[None, :]
        return bases**counts


if triton is not None:

    @triton.jit
    def window_penalty_kernel(
        logits,
        entries,
        alphas,
        count,
        row_stride: tl.constexpr,
        token_stride: tl.constexpr,
        block: tl.constexpr,
        power_stride: tl.constexpr,
    ):
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

else:
    window_penalty_kernel = None


def apply_window_penalty(
    logits: torch.Tensor,
    rows: list[int],
    penalties: list[float],
    windows: list[list[int]],
) -> None:
    """Apply frequency penalties in place to filtered, nonempty token windows."""
    if (
        window_penalty_kernel is None
        or not logits.is_cuda
        or torch.version.hip is not None
        or logits.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or not logits.is_contiguous()
    ):
        vocab = logits.shape[1]
        device = logits.device
        # note (MayDomine): a dummy vocabulary bin excludes ragged-window padding.
        num = len(windows)
        window_ids = torch.full((num, REP_PENALTY_WINDOW), vocab, dtype=torch.long)
        for i, window in enumerate(windows):
            window_ids[i, : len(window)] = torch.tensor(window, dtype=torch.long)
        window_ids = window_ids.to(device)
        counts = torch.zeros(num, vocab + 1, dtype=torch.float32, device=device)
        counts.scatter_add_(
            1, window_ids, torch.ones_like(window_ids, dtype=torch.float32)
        )
        counts = counts[:, :vocab]
        alphas = (
            torch.tensor(penalties, dtype=torch.float32, device=device).unsqueeze(1)
            ** counts
        )
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        orig_dtype = logits.dtype
        scores = logits[rows_t].to(torch.float32)
        penalized = torch.where(scores < 0, scores * alphas, scores / alphas)
        scores = torch.where(counts > 0, penalized, scores)
        logits[rows_t] = scores.to(orig_dtype)
    else:
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
            window_penalty_kernel[(triton.cdiv(len(entries), 128),)](
                logits,
                metadata,
                alphas,
                len(entries),
                logits.stride(0),
                logits.stride(1),
                128,
                REP_PENALTY_WINDOW + 1,
                enable_fp_fusion=False,
            )
