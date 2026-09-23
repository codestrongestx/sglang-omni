# MiniCPM-o Flow normalization fusion

The optional `code2wav` factory setting `enable_flow_norm_fusion` fuses LayerNorm
and adaptive modulation at the three DiTBlock sites. It defaults to `false`.
Enable it in a pipeline YAML override:

```yaml
stages:
  code2wav:
    factory:
      enable_flow_norm_fusion: true
```

The kernel supports NVIDIA CUDA FP32 tensors shaped `(batch, frames, 512)` and
per-batch `(batch, 1, 512)` modulation, including strided tensors. Other shapes,
dtypes, devices, training, autograd, and missing Triton use eager PyTorch. Attention,
convolution, MLPs, residuals, final normalization, and the Flow schedule are unchanged.

## Recorded measurements

Measurements on 2026-09-23 used one H100, MiniCPM-o-4_5 revision
`503e754207c94da6bb26850b4469f367c9ea3582`, Torch 2.13.0+cu130, Triton 3.7.1,
SGLang 0.5.20, and Transformers 5.12.1. Flow was FP32 with ten steps and CUDA graphs
disabled. The source baseline was `e57fd94d6f1324ddcfe05ea358ced7096b416696`.

| Measurement | Native | Fused |
| --- | ---: | ---: |
| Primitive, 436 frames | 10.578 µs | 3.243 µs |
| Primitive, 522 frames | 11.196 µs | 3.363 µs |
| Full HTTP mean latency, concurrency 1 | 562 ms | 571 ms |
| Full HTTP throughput, concurrency 4 | 3.931 requests/s | 4.122 requests/s |

The primitive comparisons observed zero maximum absolute output error. Six
primitive cases (7/436/522 frames, offsets 0/1000, standard deviation 0.1) passed
`rtol=atol=1e-5`. Eleven trained ten-step Flow comparisons with identical conditioning
and fixed noise covered batch sizes 1–3 and lengths 422–522, with zero observed
maximum absolute mel error. The Welford statistics operation order follows the
pinned Torch implementation; this is not a general numerical equivalence guarantee
across Torch versions or GPU architectures.

All 160 norm-only HTTP requests passed checks of settings, text, frame counts,
channels, sample rate, termination, and token counts. These checks do not establish
waveform or perceptual equivalence. Concurrency-4 throughput improved about 4.9%,
but concurrency-1 latency regressed about 1.6%. There was only one cohort per
concurrency; the result has not been independently repeated and is not a general
serving speedup claim. This tradeoff is why the setting defaults off.

The measured kernel's function AST and launch arithmetic are preserved in this
implementation. The final configuration and DiT integration have not been rerun
on CUDA. Run the primitive, adversarial, and fallback suite on the target CUDA stack:

```bash
python -m pytest tests/unit_test/minicpm_o/test_flow_norm.py -q
```

Before enabling in production, repeat trained Flow comparisons with fixed noise
and paired HTTP measurements on the intended workload and software versions.
