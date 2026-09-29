# SPDX-License-Identifier: Apache-2.0
"""Run the MiniCPM-o text backbone on SGLang's Qwen3 model."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from transformers import PretrainedConfig

from sglang_omni.models.minicpm_o.hf_config import derive_text_config

NON_TEXT_PREFIXES = (
    "vpm.",
    "resampler.",
    "apm.",
    "audio_projection_layer.",
    "tts.",
)


class MiniCPMOThinkerForCausalLM(nn.Module):
    """MiniCPM-o text backbone without the multimodal towers."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.root_config = config
        online_int8 = config.to_dict().get("online_int8", False)
        if not isinstance(online_int8, bool):
            raise ValueError("online_int8 must be a boolean")
        else:
            self.online_int8: bool = online_int8
        if self.online_int8 and quant_config is not None:
            raise ValueError("Online INT8 requires a full-precision checkpoint")
        else:
            pass
        self.config = derive_text_config(config)
        self.language_model = Qwen3ForCausalLM(
            self.config,
            quant_config,
            prefix=prefix,
        )

    @property
    def thinker(self) -> "MiniCPMOThinkerForCausalLM":
        # note (MayDomine): the shared thinker runner expects this backbone view.
        return self

    @property
    def model(self) -> nn.Module:
        return self.language_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> LogitsProcessorOutput:
        return self.language_model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        def _text_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                if name.startswith(NON_TEXT_PREFIXES):
                    continue
                else:
                    pass
                if name.startswith("llm."):
                    yield name[len("llm.") :], loaded_weight
                else:
                    pass

        self.language_model.load_weights(_text_weights())
        if self.online_int8:
            from sglang_omni.models.minicpm_o.components.online_int8 import (
                OnlineInt8LinearMethod,
            )

            for layer in self.language_model.model.modules():
                if isinstance(layer, LinearBase):
                    layer.quant_method = OnlineInt8LinearMethod()
                else:
                    pass
        else:
            pass


EntryClass = MiniCPMOThinkerForCausalLM
