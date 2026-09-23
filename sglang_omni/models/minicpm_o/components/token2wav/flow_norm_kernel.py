# SPDX-License-Identifier: Apache-2.0
"""FP32 width-512 LayerNorm and adaptive modulation kernel."""

import triton
import triton.language as tl


@triton.jit
def norm_modulate_kernel(
    x,
    shift,
    scale,
    output,
    frames: tl.constexpr,
    width: tl.constexpr,
    x_batch_stride: tl.constexpr,
    x_frame_stride: tl.constexpr,
    x_channel_stride: tl.constexpr,
    shift_batch_stride: tl.constexpr,
    shift_channel_stride: tl.constexpr,
    scale_batch_stride: tl.constexpr,
    scale_channel_stride: tl.constexpr,
    epsilon: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // frames
    frame = row % frames
    channels = tl.arange(0, block)
    mask = channels < width
    values = tl.load(
        x
        + batch * x_batch_stride
        + frame * x_frame_stride
        + channels * x_channel_stride,
        mask,
        other=0,
    )
    # note (Codex): Match native four-channel lane order for cancellation-sensitive rows.
    lanes = tl.arange(0, 128)
    lane_mean = tl.full((128,), 0, tl.float32)
    lane_m2 = tl.full((128,), 0, tl.float32)
    for item in tl.static_range(4):
        sample = tl.load(
            x
            + batch * x_batch_stride
            + frame * x_frame_stride
            + (lanes * 4 + item) * x_channel_stride
        )
        delta = sample - lane_mean
        lane_mean = tl.fma(delta, 1.0 / (item + 1), lane_mean)
        lane_m2 = tl.fma(delta, sample - lane_mean, lane_m2)
    for stage in tl.static_range(5):
        distance = 16 >> stage
        peers = tl.where(lanes % 32 + distance < 32, lanes + distance, lanes)
        peer_mean = tl.gather(lane_mean, peers, 0)
        peer_m2 = tl.gather(lane_m2, peers, 0)
        delta = lane_mean - peer_mean
        lane_mean = tl.fma(0.5, lane_mean, 0.5 * peer_mean)
        lane_m2 = tl.fma((delta * delta) * (4 << stage), 0.5, peer_m2 + lane_m2)
    for stage in tl.static_range(2):
        distance = 64 >> stage
        peers = tl.where(lanes + distance < 128, lanes + distance, lanes)
        peer_mean = tl.gather(lane_mean, peers, 0)
        peer_m2 = tl.gather(lane_m2, peers, 0)
        delta = lane_mean - peer_mean
        lane_mean = tl.fma(0.5, lane_mean, 0.5 * peer_mean)
        lane_m2 = tl.fma((delta * delta) * (128 << stage), 0.5, peer_m2 + lane_m2)
    mean = tl.sum(tl.where(lanes == 0, lane_mean, 0), 0)
    variance = tl.sum(tl.where(lanes == 0, lane_m2, 0), 0) / width
    normalized = (values - mean) * tl.rsqrt(variance + epsilon)
    scales = tl.load(
        scale + batch * scale_batch_stride + channels * scale_channel_stride,
        mask,
        other=0,
    )
    shifts = tl.load(
        shift + batch * shift_batch_stride + channels * shift_channel_stride,
        mask,
        other=0,
    )
    result = normalized * (1.0 + scales) + shifts
    tl.store(output + row * width + channels, result, mask)
