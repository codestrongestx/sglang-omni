# MiniCPM-o sparse repetition penalty

The talker applies its repetition penalty using the frequency of each token in
its last 16 output positions. Invalid token IDs are excluded after taking that
window. The CUDA implementation updates each distinct recent token once instead
of constructing vocabulary-sized counts and intermediate tensors on every decode
step. Duplicate frequencies, sampling parameters, model precision, and output
limits are unchanged.

Penalty powers are computed with Torch FP32 tensor exponentiation and cached in a
bounded, stream-local table. The kernel converts logits to FP32, multiplies
negative values or applies correctly rounded division to other values, then
casts back once. Each `(row, token)` has one writer. CPU, non-NVIDIA backends,
missing Triton, unsupported dtypes, and noncontiguous logits use the original
Torch implementation. Compilation/execution errors on the supported path remain
visible rather than being silently converted to fallback execution.

## Recorded prototype results

These measurements used an earlier experimental integration of the same kernel
arithmetic, based on commit `e57fd94d6f1324ddcfe05ea358ced7096b416696`.
They are not measurements of this cleaned production integration.

- One H100 80GB HBM3; Torch 2.13.0+cu130, SGLang 0.5.20, Transformers 5.12.1,
  Triton 3.7.1.
- Model `openbmb/MiniCPM-o-4_5`, revision
  `503e754207c94da6bb26850b4469f367c9ea3582`.
- Thinker/talker memory fractions 0.55/0.15; maximum running requests and decoder
  graph batch caps both 8 for both stages. Code2Wav remained FP32. No Flow graphs,
  normalization fusion, HiFT fusion, or quantization changes.
- Loopback, non-streaming HTTP text-plus-WAV completion. Four short English
  read-aloud prompts, each repeated 20 times per concurrency. Greedy settings,
  seeds, and output limits matched; warmup excluded.
- A/B/B/A order: native, sparse, sparse, native. Each cohort contained 80 requests
  at concurrency 1 and 80 at concurrency 4; all 640 requests succeeded.

| Cohort | C1 mean latency | C4 throughput |
|---|---:|---:|
| Native A1 | 569 ms | 3.895 requests/s |
| Sparse B1 | 548 ms | 4.267 requests/s |
| Sparse B2 | 533 ms | 4.255 requests/s |
| Native A2 | 562 ms | 3.931 requests/s |
| Pooled native | 565.464 ms | 3.912895 requests/s |
| Pooled sparse | 540.656 ms | 4.261185 requests/s |

The pooled result was **4.39% lower C1 latency** and **8.90% higher C4 throughput**.
The two C1 reductions were 3.7% and 5.1%; every prompt improved at C1. Throughput
is total requests divided by summed cohort wall time. Two serial cohort pairs
and four prompts do not establish general-workload or independent-server
confidence, streaming first-audio performance, or portability to other GPUs.

With batch 1, BF16 logits and vocabulary 6,562, prepared GPU work measured
24.491 microseconds for the dense Torch operations, 13.695 microseconds for
sparse Torch operations with cached powers, and 1.515 microseconds for the fused
kernel. Each CUDA graph replay contained 32 operations; transfers were excluded
from all arms. A separate synchronized runner measurement including CPU metadata
preparation and transfer decreased from 142.6 to 40.0 microseconds. These isolated
results do not replace the HTTP measurements.

## Correctness and limits

The prototype passed exact output comparisons for FP32/FP16/BF16 logits at batches
1/4/8, including duplicate counts 1–16, ragged and empty histories, invalid IDs,
window boundaries, mixed penalties, signed zero, infinities, and NaNs. Serving
calibration compared native and candidate live logits with `atol=rtol=0` on every
penalized decode step of a real request before timing.

All 320 candidate HTTP requests matched their baselines in settings, returned
text, audio frame/channel counts, sample rate, normal termination, and reported
usage/token counts. Those checks did not compare every codec token ID or prove
waveform/perceptual equivalence. Four saved WAV pairs had different PCM despite
matching lengths (maximum absolute difference 0.543–0.736). Decoder randomness is
a plausible contributor, but its cause was not isolated. Exact same-input live
logits are the stronger penalty correctness check.

The production extraction preserves the measured arithmetic and metadata
packing. Changes are optional imports, automatic dispatch with the existing
fallback, a named power-table width, cache ownership by CUDA stream instead
of device alone, and an explicit device context for the kernel launch. The production wrapper, stream-local cache, and added GPU tests
have **not been rerun on CUDA**; no new paid GPU resources were used for this PR.
The recorded results therefore support the prototype, with production CUDA
validation still required before accepting identical performance/parity claims.

Local validation on macOS with Torch 2.14.0: the standalone kernel suite passed
9 CPU cases and skipped 14 CUDA cases. Triton was absent, exercising the genuine
optional-import/CPU fallback path. The SGLang runner integration test was added
but not run in that minimal environment. Changed-file pre-commit checks passed;
repository-wide checks passed except Rust formatting because cargo was absent.

## Reproduce correctness checks

From the repository root with Torch and pytest installed:

```bash
python -m pytest -q tests/unit_test/minicpm_o/test_sampling_kernels.py
```

CPU cases run without SGLang or Triton; GPU cases skip without NVIDIA CUDA and
Triton. On a CUDA machine with the project's serving dependencies, also run:

```bash
python -m pytest -q tests/unit_test/minicpm_o/test_sampling_kernels.py \
  tests/unit_test/minicpm_o/test_sglang_talker.py
```

The GPU suite compares the automatic fused path against the original Torch
fallback at zero tolerance, including independently executing CUDA streams.
The runner test checks window slicing before ID filtering and no-op requests.
A production performance rerun should use matched requests and output checks,
warmed A/B/B/A cohorts at concurrency 1 and 4, unchanged precision and output
limits, and report full HTTP completion time separately from kernel timings.
