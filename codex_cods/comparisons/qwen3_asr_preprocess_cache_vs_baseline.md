# Qwen3-ASR Preprocess Cache vs Baseline

Same H100 shape for both runs:

- Model: `Qwen/Qwen3-ASR-1.7B`
- Dataset: SeedTTS EN reference clips
- Samples: 20
- Concurrency: 32
- Repeats: 3
- Warmup: enabled
- GPU: 1x NVIDIA H100 80GB HBM3

Runs:

- Baseline: `codex_cods/runs/20260620_qwen3_asr_h100_baseline_c32_warmup/`
- Candidate: `codex_cods/runs/20260620_qwen3_asr_h100_cache_c32_warmup/`

## Results

| metric | baseline | candidate | change |
|---|---:|---:|---:|
| Evaluated | 20/20 | 20/20 | same |
| Skipped/failures | 0 | 0 | same |
| Corpus WER mean | 0.012687 | 0.011534 | stable |
| Corpus WER max | 0.017301 | 0.013841 | stable |
| Throughput mean | 38.054/s | 90.854/s | +138.8% |
| Throughput best | 38.884/s | 93.939/s | +141.6% |
| Wall clock mean | 0.526 s | 0.220 s | -58.1% |
| Latency mean | 0.465 s | 0.156 s | -66.5% |
| Latency p95 mean | 0.518 s | 0.210 s | -59.5% |
| RTF mean | 0.0939 | 0.0311 | -66.9% |
| RTF p95 mean | 0.1208 | 0.0419 | -65.3% |

## Interpretation

The candidate materially improves the official dev benchmark while preserving WER and failure count.

The effect is expected because the benchmark sends the same 20 uploaded audio files in the warmup pass and in each timed repeat. The cache removes repeated decode and feature-extraction work for duplicate uploaded bytes and also reduces serial request-build pressure before scheduler admission.

This is not evidence of the same gain for entirely unique ASR traffic. It is evidence that duplicate-input preprocessing caching improves the current checker-sized H100 dev workload.

Decision: keep.
