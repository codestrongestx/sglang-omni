# SPDX-License-Identifier: Apache-2.0
"""Check online INT8 against independent dequantized linear references."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

pytest.importorskip("sglang")

from sglang_omni.models.minicpm_o.components.online_int8 import OnlineInt8LinearMethod


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Marlin requires CUDA")
@pytest.mark.parametrize(
    "input_features,output_features", [(4096, 6144), (4096, 24576), (12288, 4096)]
)
@pytest.mark.parametrize("token_count", [1, 33])
def test_online_int8_linear(
    input_features: int, output_features: int, token_count: int
) -> None:
    torch.manual_seed(2026)
    layer = torch.nn.Linear(
        input_features, output_features, bias=False, device="cuda", dtype=torch.bfloat16
    )
    with torch.no_grad():
        layer.weight[:4, :128] = 0
    weight = layer.weight.detach().float()
    grouped = weight.reshape(output_features, -1, 32)
    scales = (
        (grouped.abs().amax(-1) / 127)
        .clamp_min(torch.finfo(torch.bfloat16).tiny)
        .to(torch.bfloat16)
        .float()
    )
    reference_weight = (
        (grouped / scales.unsqueeze(-1)).round().clamp(-127, 127) * scales.unsqueeze(-1)
    ).reshape_as(weight)
    hidden = torch.randn(
        token_count, input_features, device="cuda", dtype=torch.bfloat16
    )
    bias = torch.randn(output_features, device="cuda", dtype=torch.bfloat16)
    expected = (
        functional.linear(hidden.float(), reference_weight)
        .to(torch.bfloat16)
        .add_(bias)
    )
    method = OnlineInt8LinearMethod()
    method.process_weights_after_loading(layer)
    actual = method.apply(layer, hidden, bias)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    relative_error = torch.linalg.vector_norm(
        actual.float() - expected.float()
    ) / torch.linalg.vector_norm(expected.float())
    assert relative_error < 0.004
    output = torch.empty_like(expected)
    returned = method.apply_into(layer, hidden, output, bias)
    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
    original = functional.linear(hidden.float(), weight).to(torch.bfloat16).add_(bias)
    quantization_error = torch.linalg.vector_norm(
        actual.float() - original.float()
    ) / torch.linalg.vector_norm(original.float())
    assert quantization_error < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Marlin requires CUDA")
def test_online_int8_zero_weights() -> None:
    layer = torch.nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.zero_()
    method = OnlineInt8LinearMethod()
    method.process_weights_after_loading(layer)
    hidden = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    result = method.apply(layer, hidden)
    torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
