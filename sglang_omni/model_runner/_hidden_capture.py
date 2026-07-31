# SPDX-License-Identifier: Apache-2.0
"""Hidden state capture helpers for multi-layer extraction.

SGLang's VL wrapper (Qwen3VLForConditionalGeneration) doesn't support
the aux_hidden_states tuple returned by the text model when layers_to_capture
is set. Omni-owned model wrappers can publish those layers through
``LogitsProcessorOutput.hidden_states``; the generic fallback below retains the
older hook/side-channel behavior for wrappers that do not support that path.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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


def install_hidden_capture_hooks(
    model: nn.Module,
    capture_layers: list[int],
) -> None:
    """Install forward wrapper on the text model to capture aux hidden states.

    Args:
        model: Top-level SGLang model (e.g. Qwen3OmniMoeForConditionalGeneration)
        capture_layers: Layer indices to capture (e.g. [0, 24]).
            Layer 0 captures embed output (input to first transformer layer).
            Layer N captures input to layer N (= output of layer N-1).
    """
    from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
        Qwen3OmniThinkerForCausalLM,
    )

    if isinstance(model, Qwen3OmniThinkerForCausalLM):
        model.configure_hidden_capture_layers(capture_layers)
        # Keep the legacy attribute available to generic output processing, but
        # the model-specific forward publishes capture data on its result.
        model._captured_aux_hidden_states = None
        logger.info(
            "Configured first-class hidden capture on %s for layers %s",
            type(model).__name__,
            capture_layers,
        )
        return

    # Navigate to the text model that has layers_to_capture.
    # Qwen3OmniMoeForConditionalGeneration -> .thinker -> .model (Qwen3MoeLLMModel)
    # Qwen3OmniTalker -> .model (text model)
    if hasattr(model, "thinker"):
        text_model = model.thinker.model
    elif hasattr(model, "model"):
        text_model = model.model
    else:
        raise AttributeError(
            f"Cannot find text model on {type(model).__name__}. "
            "Expected .thinker.model or .model attribute."
        )

    # Set layers_to_capture on the text model
    text_model.layers_to_capture = list(capture_layers)

    # Storage for captured aux hidden states (overwritten each forward pass)
    model._captured_aux_hidden_states = None

    # Wrap the text model's forward to intercept tuple returns
    original_forward = text_model.forward

    @functools.wraps(original_forward)
    def _capturing_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = original_forward(*args, **kwargs)
        if isinstance(result, tuple):
            hidden_states, aux_hidden_states = result
            model._captured_aux_hidden_states = aux_hidden_states
            return hidden_states
        else:
            model._captured_aux_hidden_states = None
            return result

    text_model.forward = _capturing_forward
    logger.info(
        "Installed hidden capture hooks on %s for layers %s",
        type(text_model).__name__,
        capture_layers,
    )
