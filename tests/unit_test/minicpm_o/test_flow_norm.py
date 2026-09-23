# SPDX-License-Identifier: Apache-2.0
"""Numerical and fallback contracts for Flow normalization."""

import pytest
import torch

from sglang_omni.models.minicpm_o.components.token2wav.dit import DiTBlock


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [64, 512])
def test_flow_norm_cpu_fallback(dtype: torch.dtype, width: int) -> None:
    block = DiTBlock(width, 1, width, enable_flow_norm_fusion=True).to(dtype).eval()
    x = torch.randn(2, 7, width, dtype=dtype, requires_grad=True)
    shift = torch.randn(2, 1, width, dtype=dtype, requires_grad=True)
    scale = torch.randn(2, 1, width, dtype=dtype, requires_grad=True)
    actual = block.norm_modulate(x, shift, scale, block.norm1)
    expected = block.norm1(x) * (1 + scale) + shift
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert all(tensor.grad is not None for tensor in (x, shift, scale))


@pytest.mark.accelerator
@pytest.mark.parametrize("offset", [0.0, 1000.0])
@pytest.mark.parametrize("frames", [7, 436, 522])
@torch.inference_mode()
def test_flow_norm_modulation_matches_native(offset: float, frames: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused Flow normalization")
    pytest.importorskip("triton")
    block = DiTBlock(512, 8, 64, enable_flow_norm_fusion=True).cuda().eval()
    torch.manual_seed(42)
    x = torch.randn(2, frames, 512, device="cuda") * 0.1 + offset
    shift, scale = torch.randn(2, 1, 4608, device="cuda").chunk(9, dim=-1)[:2]
    expected = block.norm1(x) * (1 + scale) + shift
    actual = block.norm_modulate(x, shift, scale, block.norm1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.accelerator
@pytest.mark.parametrize("constant", [False, True])
@torch.inference_mode()
def test_flow_norm_strided_inputs(constant: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused Flow normalization")
    pytest.importorskip("triton")
    block = DiTBlock(512, 8, 64, enable_flow_norm_fusion=True).cuda().eval()
    x = torch.randn(3, 512, 14, device="cuda").transpose(1, 2)[:, ::2, :]
    if constant:
        x.fill_(1000.0)
    shift, scale = torch.randn(3, 1, 1536, device="cuda").chunk(3, dim=-1)[:2]
    before = x.clone()
    expected = block.norm1(x) * (1 + scale) + shift
    actual = block.norm_modulate(x, shift, scale, block.norm1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert actual.data_ptr() != x.data_ptr()


@pytest.mark.accelerator
@pytest.mark.parametrize("fallback", ["training", "grad", "width", "dtype", "disabled"])
def test_flow_norm_cuda_fallback(fallback: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fallback validation")
    width = 64 if fallback == "width" else 512
    dtype = torch.float16 if fallback == "dtype" else torch.float32
    block = (
        DiTBlock(width, 1, width, enable_flow_norm_fusion=fallback != "disabled")
        .cuda()
        .to(dtype)
    )
    block.train(fallback == "training")
    with torch.set_grad_enabled(fallback == "grad"):
        x = torch.randn(2, 7, width, device="cuda", dtype=dtype, requires_grad=True)
        shift = torch.randn(2, 1, width, device="cuda", dtype=dtype)
        scale = torch.randn_like(shift)
        actual = block.norm_modulate(x, shift, scale, block.norm1)
        expected = block.norm1(x) * (1 + scale) + shift
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if fallback == "grad":
            actual.sum().backward()
            assert x.grad is not None


@pytest.mark.parametrize("autocast", [False, True])
def test_flow_norm_block_cpu_parity(autocast: bool) -> None:
    torch.manual_seed(42)
    native = DiTBlock(64, 1, 64, enable_flow_norm_fusion=False).eval()
    fused = DiTBlock(64, 1, 64, enable_flow_norm_fusion=True).eval()
    fused.load_state_dict(native.state_dict(), strict=True)
    x = torch.randn(2, 7, 64)
    conditioning = torch.randn(2, 1, 64)
    mask = torch.ones(2, 1, 7, dtype=torch.bool)
    mask[1, :, -2:] = False
    with (
        torch.inference_mode(),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast),
    ):
        torch.testing.assert_close(
            fused(x, conditioning, mask), native(x, conditioning, mask), rtol=0, atol=0
        )


@pytest.mark.accelerator
@torch.inference_mode()
def test_flow_norm_uses_input_device() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices are required")
    pytest.importorskip("triton")
    with torch.cuda.device(0):
        block = DiTBlock(512, 8, 64, enable_flow_norm_fusion=True).to("cuda:1").eval()
        x = torch.randn(2, 7, 512, device="cuda:1")
        shift, scale = torch.randn(2, 1, 4608, device="cuda:1").chunk(9, dim=-1)[:2]
        expected = block.norm1(x) * (1 + scale) + shift
        actual = block.norm_modulate(x, shift, scale, block.norm1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert torch.cuda.current_device() == 0
