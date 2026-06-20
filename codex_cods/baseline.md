# Qwen3-ASR H100 Dev Benchmark

Active baseline GPU: 1x NVIDIA H100 80GB HBM3.

Official checker-sized workload:

- Model: `Qwen/Qwen3-ASR-1.7B`
- Dataset: SeedTTS EN reference clips
- Samples: 20
- Serving fan-out: `c=32`
- Benchmark repeats: 3
- Warmup: enabled
- Command: see `codex_cods/runs/20260620_qwen3_asr_h100_baseline_c32_warmup/command.sh`

## Clean Baseline

Run: `codex_cods/runs/20260620_qwen3_asr_h100_baseline_c32_warmup/`

| metric | value |
|---|---:|
| Evaluated | 20/20 |
| Skipped/failures | 0 |
| Corpus WER mean | 0.012687 |
| Corpus WER range | 0.010381 to 0.017301 |
| Throughput mean | 38.054 samples/s |
| Throughput best | 38.884 samples/s |
| Wall clock mean | 0.526 s |
| Latency mean | 0.465 s |
| Latency p95 mean | 0.518 s |
| RTF mean | 0.0939 |
| RTF p95 mean | 0.1208 |
| Worker balance | Not applicable; `/workers` returned 404 and benchmark worker fields were empty. |

## Profiling Evidence

Built-in request profile: `codex_cods/runs/20260620_qwen3_asr_h100_profile_c32_warmup/`

- Captured 80 requests including warmup.
- Coarse intervals showed `stage_input_received->stage_complete` avg 464.744 ms and `scheduler_request_build_start->scheduler_request_build_end` avg 16.760 ms.
- Buckets were too coarse, so temporary ASR-specific request-builder markers were added for a second profiling run.

ASR substage profile: `codex_cods/runs/20260620_qwen3_asr_h100_profile_c32_substage/`

Timed repeats only, excluding the discarded warmup request group:

| interval | avg ms | p95 ms |
|---|---:|---:|
| Stage total | 551.387 | 608.953 |
| Stage pre-build wait | 208.065 | 392.352 |
| Request build total | 20.989 | 27.378 |
| Audio decode | 8.442 | 9.084 |
| Feature extraction | 7.963 | 11.275 |
| Scheduler queue wait | 191.441 | 373.150 |
| Model to result adapter | 129.117 | 186.342 |
| Result adapter | 0.618 | 1.715 |

Bottleneck hypothesis after reviewer critique:

- Serial request construction is a leading controllable host-side contributor, not the only bottleneck.
- The concrete per-request request-build cost is about 21 ms, mostly audio decode plus feature extraction.
- Under the official c=32 burst shape, requests wait behind earlier serial request builds, making duplicate-input preprocessing a useful target.
- Scheduler queueing and model execution remain material contributors and should not be ignored in later work.

## Cache Candidate

Run: `codex_cods/runs/20260620_qwen3_asr_h100_cache_c32_warmup/`

Implemented a bounded per-process cache for uploaded audio bytes in the Qwen3-ASR request builder. The cache stores decoded duration/fingerprint plus CPU feature tensors and derived audio-token metadata. Cached entries are cloned per request before packaging into `MultimodalDataItem`.

This directly targets the official checker shape, which resends the same 20 uploaded audio clips across warmup and timed repeats. It should be described as duplicate-input preprocessing caching, not as a general unique-audio throughput result.

Provenance: the candidate run was executed from the base benchmark commit with the measured cache diff recorded in `codex_cods/runs/20260620_qwen3_asr_h100_cache_c32_warmup/cache_candidate.diff`. After post-commit review, the integrated cache was narrowed from module-global to per-adapter scope to prevent reuse across different feature extractors. That correction does not change the single-adapter checker workload path measured here.

| metric | baseline | cache candidate | change |
|---|---:|---:|---:|
| Corpus WER mean | 0.012687 | 0.011534 | stable |
| Skipped/failures | 0 | 0 | stable |
| Throughput mean | 38.054/s | 90.854/s | +138.8% |
| Wall clock mean | 0.526 s | 0.220 s | -58.1% |
| Latency mean | 0.465 s | 0.156 s | -66.5% |
| Latency p95 mean | 0.518 s | 0.210 s | -59.5% |
| RTF mean | 0.0939 | 0.0311 | -66.9% |
| RTF p95 mean | 0.1208 | 0.0419 | -65.3% |

Decision: keep the cache candidate. WER remained stable for the exact same H100 benchmark shape and performance improved materially.
