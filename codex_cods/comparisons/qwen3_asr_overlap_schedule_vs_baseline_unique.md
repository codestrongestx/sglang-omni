# Qwen3-ASR Overlap Schedule vs Unique Baseline

## Scope

- Baseline trace: `codex_cods/runs/20260621_h100_unique_baseline`
- Candidate trace: `codex_cods/runs/20260621_h100_unique_overlap_schedule`
- Hardware: RunPod H100 SXM 80GB, 1 GPU, `US-CA-2`
- Workload: `benchmark_qwen3_asr_concurrency`, `concurrency=32`, `repeats=3`, `--warmup`, `--unique-timed-samples`, `--max-samples 1088`, `--unique-samples-per-repeat 96`
- Timed samples: 288 evaluated, 0 skipped in both runs
- Code audit: exact measured heads resolve at `refs/heads/runpod/issue831-baseline_unique` and `refs/heads/runpod/issue831-overlap_schedule`; each trace also includes `code_head.patch.b64`.
- Router/worker balance: not applicable for this 1-GPU direct ASR server benchmark; `/workers` returned 404 and metric worker objects are empty.

## Result

| metric | baseline | overlap schedule | delta |
| --- | ---: | ---: | ---: |
| wall mean | 3.671s | 3.544s | 1.04x faster |
| throughput mean | 26.154 samples/s | 27.091 samples/s | 1.04x higher |
| latency mean | 1.184s | 1.134s | 1.04x lower |
| latency p95 | 1.895s | 1.724s | 1.10x lower |
| RTF mean | 0.2555 | 0.2454 | 1.04x lower |
| max corpus WER | 0.0189 | 0.0215 | same band |
| skipped | 0 | 0 | stable |

## Interpretation

The overlap-schedule direction does not move the representative H100 unique-audio workload enough to prioritize by itself. It is useful evidence because it rules out this scheduling hook as the primary issue 831 optimization path under the measured shape.

## Caveats

- Treat this as a measurement result for the tested hook, not as proof that all scheduling changes are unhelpful.
- The raw pulled `nvidia-smi` captures predate the final wrapper ordering fix. The committed `nvidia-smi.txt` files are post-hoc hardware snapshots from the same RunPod H100 pod.
