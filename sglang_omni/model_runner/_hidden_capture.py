# SPDX-License-Identifier: Apache-2.0
"""Packed hidden-state capture helpers.

Speech capture publishes the requested layers through
``LogitsProcessorOutput.hidden_states``: the model packs
``[captured layers..., final stream state]`` side by side along the last axis
(see ``Qwen3OmniThinkerForCausalLM.process_hidden_states``) so the tensor rides
CUDA-graph replay as a single ordinary output. This module holds the matching
unpack helper shared by the model runner and the output processor.
"""

from __future__ import annotations

from typing import Any

import torch


def unpack_packed_hidden_capture(
    packed: Any,
    *,
    capture_layer_count: int,
    hidden_size: int | None,
) -> tuple[tuple[torch.Tensor, ...] | None, torch.Tensor | None]:
    """Split ``[captured layers..., final stream state]`` along the last axis."""
    if not isinstance(packed, torch.Tensor):
        return None, None
    if (
        capture_layer_count <= 0
        or hidden_size is None
        or hidden_size <= 0
        or packed.ndim == 0
    ):
        return None, packed

    part_count = capture_layer_count + 1
    if packed.shape[-1] != hidden_size * part_count:
        return None, packed

    parts = packed.split(hidden_size, dim=-1)
    return tuple(parts[:-1]), parts[-1]
