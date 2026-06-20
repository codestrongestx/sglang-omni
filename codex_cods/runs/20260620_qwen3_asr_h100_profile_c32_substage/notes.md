# Qwen3-ASR H100 request profile, ASR substages

Same workload shape as the clean baseline, with temporary ASR-specific request-builder markers:

- Model: `Qwen/Qwen3-ASR-1.7B`
- Dataset: SeedTTS EN reference clips
- Samples: 20
- Concurrency: 32
- Repeats: 3
- Warmup: enabled
- GPU: 1x NVIDIA H100 80GB HBM3

The profile captured 80 requests, including the warmup pass. The table below uses
`asr_substage_timed_only_table.txt`, which excludes the discarded warmup request
group and matches the timed benchmark repeats:

| interval | avg ms | p95 ms |
|---|---:|---:|
| stage total | 551.387 | 608.953 |
| stage pre-build wait | 208.065 | 392.352 |
| request build total | 20.989 | 27.378 |
| audio decode | 8.442 | 9.084 |
| feature extraction | 7.963 | 11.275 |
| scheduler queue wait | 191.441 | 373.150 |
| model to result adapter | 129.117 | 186.342 |
| result adapter | 0.618 | 1.715 |

Bottleneck ranking:

1. Serial request construction under c=32 burst load. The stage receives many requests quickly, then each request waits for earlier request builds before its own `scheduler_request_build_start`. The timed-repeat average pre-build wait is 208 ms.
2. Scheduler queue wait after request build. This averages 191 ms and likely reflects batching/backpressure after the request objects enter the scheduler queue.
3. Scheduler/model execution after prefill start. `model_to_result_adapter` averages 129 ms and has p95 186 ms.
4. Individual ASR request build cost. The build averages 21.0 ms, mostly audio decode plus feature extraction, but that cost becomes a large burst-level delay because the build path is serial.

Candidate next move for critique:

- Consider a low-risk adapter-scoped cache keyed by raw uploaded audio bytes for repeated ASR inputs, caching loaded audio plus feature-extractor output and derived token counts.
- This should preserve request semantics for duplicate audio and directly targets audio decode plus feature extraction, the largest concrete request-build substages.
- Risk: the checker-sized benchmark repeats the same 20 clips across warmup/repeats, so a cache may improve the dev benchmark more than unique-audio production traffic. It should be treated as valid only if bounded, transparent, and measured against the exact same H100 shape with stable WER.
