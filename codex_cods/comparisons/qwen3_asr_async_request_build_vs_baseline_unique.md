# Qwen3-ASR Async Request Build vs Unique Baseline

## Scope

- Baseline trace: `codex_cods/runs/20260621_h100_unique_baseline`
- Candidate trace: `codex_cods/runs/20260621_h100_unique_async_request_build`
- Hardware: RunPod H100 SXM 80GB, 1 GPU, `US-CA-2`
- Workload: `benchmark_qwen3_asr_concurrency`, `concurrency=32`, `repeats=3`, `--warmup`, `--unique-timed-samples`, `--max-samples 1088`, `--unique-samples-per-repeat 96`
- Timed samples: 288 evaluated, 0 skipped in both runs
- Code audit: exact measured heads resolve at `refs/heads/runpod/issue831-baseline_unique` and `refs/heads/runpod/issue831-async_request_build`; each trace also includes `code_head.patch.b64`.
- Router/worker balance: not applicable for this 1-GPU direct ASR server benchmark; `/workers` returned 404 and metric worker objects are empty.

## Result

| metric | baseline | async request build | delta |
| --- | ---: | ---: | ---: |
| wall mean | 3.671s | 2.058s | 1.78x faster |
| throughput mean | 26.154 samples/s | 46.661 samples/s | 1.78x higher |
| latency mean | 1.184s | 0.638s | 1.85x lower |
| latency p95 | 1.895s | 0.934s | 2.03x lower |
| RTF mean | 0.2555 | 0.1380 | 1.85x lower |
| max corpus WER | 0.0189 | 0.0224 | same band |
| skipped | 0 | 0 | stable |

## Interpretation

The async request-build direction is the clear winner among the three explored candidates. It improves representative unique-audio throughput by 1.78x on the matched H100 shape while preserving completion count and keeping WER in the same band.

## Caveats

- The candidate `runner.log` has a post-completion shell parse warning from the earlier wrapper revision. Metrics files were already written.
- The raw pulled `nvidia-smi` captures predate the final wrapper ordering fix. The committed `nvidia-smi.txt` files are post-hoc hardware snapshots from the same RunPod H100 pod.
