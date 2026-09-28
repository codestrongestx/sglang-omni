# SPDX-License-Identifier: Apache-2.0
"""Load full-precision linear weights into grouped INT8 Marlin storage."""

from __future__ import annotations

import torch
from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import get_scalar_types

GROUP_SIZE = 32
INT8_MAX = 127
INT8_OFFSET = 128
PACK_FACTOR = 4
BITS_PER_WEIGHT = 8
MARLIN_OUTPUT_ALIGNMENT = 64
MARLIN_INPUT_ALIGNMENT = 128


class OnlineInt8LinearMethod(UnquantizedLinearMethod):
    """Quantize ordinary checkpoint weights after their tensor-parallel loading."""

    input_features: int
    output_features: int
    empty_indices: torch.Tensor
    workspace: torch.Tensor

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.detach()
        if not weight.is_cuda or weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Online INT8 requires FP16 or BF16 CUDA weights")
        else:
            pass
        self.output_features, self.input_features = weight.shape
        if (
            self.input_features % MARLIN_INPUT_ALIGNMENT
            or self.output_features % MARLIN_OUTPUT_ALIGNMENT
        ):
            raise ValueError(
                "Online INT8 requires input multiples of 128 and output multiples of 64"
            )
        else:
            pass
        grouped = weight.float().reshape(self.output_features, -1, GROUP_SIZE)
        scales = (
            (grouped.abs().amax(dim=-1) / INT8_MAX)
            .clamp_min(torch.finfo(weight.dtype).tiny)
            .to(weight.dtype)
        )
        quantized = (
            (grouped / scales.float().unsqueeze(-1))
            .round()
            .clamp(-INT8_MAX, INT8_MAX)
            .to(torch.int32)
        )
        quantized = (
            quantized.reshape(weight.shape).T.contiguous() + INT8_OFFSET
        ).reshape(-1, PACK_FACTOR, self.output_features)
        shifts = (
            torch.arange(PACK_FACTOR, device=weight.device, dtype=torch.int32)
            * BITS_PER_WEIGHT
        )
        packed = torch.sum(quantized << shifts[None, :, None], dim=1).to(torch.int32)
        self.empty_indices = torch.empty(0, device=weight.device, dtype=torch.int32)
        packed = gptq_marlin_repack(
            packed,
            self.empty_indices,
            self.input_features,
            self.output_features,
            BITS_PER_WEIGHT,
        )
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)
        layer.register_buffer(
            "weight_scale",
            marlin_permute_scales(
                scales.T.contiguous(),
                self.input_features,
                self.output_features,
                GROUP_SIZE,
            ),
        )
        self.workspace = marlin_make_workspace(weight.device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, scalar_types = get_scalar_types()
        return apply_gptq_marlin_linear(
            x,
            layer.weight,
            layer.weight_scale,
            self.empty_indices,
            self.empty_indices,
            self.empty_indices,
            self.workspace,
            scalar_types.uint8b128,
            self.output_features,
            self.input_features,
            True,
            bias,
        )

    def apply_into(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        output: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output.copy_(self.apply(layer, x, bias))
        return output
