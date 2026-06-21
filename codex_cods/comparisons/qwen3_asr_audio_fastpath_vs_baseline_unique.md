# Qwen3-ASR PCM WAV Fast Path vs Unique Baseline

## Scope

- Baseline trace: `codex_cods/runs/20260621_h100_unique_baseline`
- Candidate trace: `codex_cods/runs/20260621_h100_unique_audio_fastpath`
- Hardware: RunPod H100 SXM 80GB, 1 GPU, `US-CA-2`
- Workload: `benchmark_qwen3_asr_concurrency`, `concurrency=32`, `repeats=3`, `--warmup`, `--unique-timed-samples`, `--max-samples 1088`, `--unique-samples-per-repeat 96`
- Timed samples: 288 evaluated, 0 skipped in both runs
- Code audit: exact measured heads resolve at `refs/heads/runpod/issue831-baseline_unique` and `refs/heads/runpod/issue831-audio_fastpath`; each trace also includes `code_head.patch.b64`.
- Router/worker balance: not applicable for this 1-GPU direct ASR server benchmark; `/workers` returned 404 and metric worker objects are empty.

## Result

| metric | baseline | PCM WAV fast path | delta |
| --- | ---: | ---: | ---: |
| wall mean | 3.671s | 2.913s | 1.26x faster |
| throughput mean | 26.154 samples/s | 32.989 samples/s | 1.26x higher |
| latency mean | 1.184s | 0.925s | 1.28x lower |
| latency p95 | 1.895s | 1.468s | 1.29x lower |
| RTF mean | 0.2555 | 0.1996 | 1.28x lower |
| max corpus WER | 0.0189 | 0.0172 | same band |
| skipped | 0 | 0 | stable |

## Interpretation

The PCM WAV fast path is a meaningful secondary candidate. It produces a 1.26x throughput improvement on the representative unique-audio H100 shape and keeps WER and skipped counts stable.

## Caveats

- The stale duplicate artifact `20260621T040539Z_audio_fastpath_1088` was interrupted and should be ignored.
- Use `20260621T040448Z_audio_fastpath_1088b` and the committed trace directory for this comparison.
- The raw pulled `nvidia-smi` captures predate the final wrapper ordering fix. The committed `nvidia-smi.txt` files are post-hoc hardware snapshots from the same RunPod H100 pod.
