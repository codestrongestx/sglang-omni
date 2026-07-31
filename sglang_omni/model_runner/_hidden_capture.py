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
    packed: torch.Tensor | None,
    *,
    capture_layer_count: int,
    hidden_size: int,
) -> tuple[tuple[torch.Tensor, ...] | None, torch.Tensor | None]:
    """Split ``[captured layers..., final stream state]`` along the last axis.

    ``packed`` is None on steps that captured nothing (NULL capture mode);
    any other width than ``hidden_size * (capture_layer_count + 1)`` means the
    capture configuration and the model disagree — fail loud instead of
    silently dropping speech hidden states.
    """
    if packed is None:
        return None, None
    part_count = capture_layer_count + 1
    assert packed.shape[-1] == hidden_size * part_count, (
        f"packed hidden capture width {packed.shape[-1]} != "
        f"{hidden_size} * {part_count} (hidden_size * (capture layers + 1))"
    )
    parts = packed.split(hidden_size, dim=-1)
    return tuple(parts[:-1]), parts[-1]
