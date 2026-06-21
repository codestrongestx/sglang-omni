# 2026-06-21 RunPod H100 Unique-Audio Issue 831 Measurements

## Setup

- Pod: `28xh4e0kd3aomf`, RunPod H100 SXM 80GB in `US-CA-2`, network volume mounted at `/workspace`.
- Pod status after runs: stopped by user action.
- Final balance check after stop: `1307.8468794151`; current spend returned to `0.049/hr`.
- Benchmark shape: `benchmark_qwen3_asr_concurrency`, `concurrency=32`, `repeats=3`, `--warmup`, `--unique-timed-samples`, `--max-samples 1088`, `--unique-samples-per-repeat 96`.
- Unique-window shape: 96 samples/window x 4 windows; benchmark reported 704 loaded samples unused. Each result evaluated 288 timed samples with 0 skips.
- Pulled artifacts root: `results/runpod/28xh4e0kd3aomf/results/qwen3_asr_831/experiments/`.

## Results

| run | branch head | artifact dir | wall mean | throughput mean | speedup vs baseline | latency mean | latency p95 | rtf mean | max corpus WER | skipped |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline unique harness | `fdd10c4` | `20260621T035858Z_baseline_unique_1088` | 3.671s | 26.154 samples/s | 1.00x | 1.184s | 1.895s | 0.2555 | 0.0189 | 0 |
| async request build | `e332193` | `20260621T035935Z_async_request_build_1088` | 2.058s | 46.661 samples/s | 1.78x | 0.638s | 0.934s | 0.1380 | 0.0224 | 0 |
| overlap schedule | `eed0ef9` | `20260621T040448Z_overlap_schedule_1088b` | 3.544s | 27.091 samples/s | 1.04x | 1.134s | 1.724s | 0.2454 | 0.0215 | 0 |
| PCM WAV load fast path | `2441f70` | `20260621T040448Z_audio_fastpath_1088b` | 2.913s | 32.989 samples/s | 1.26x | 0.925s | 1.468s | 0.1996 | 0.0172 | 0 |

## Interpretation

- The async request-build branch is the clear winner in this matched H100 unique-audio shape: 46.661 samples/s vs 26.154 baseline, a 1.78x throughput improvement.
- The PCM WAV load fast path is useful but smaller: 32.989 samples/s, a 1.26x throughput improvement.
- The overlap schedule hook is close to baseline: 27.091 samples/s, a 1.04x improvement, not enough to prioritize by itself.
- WER remained in the same band across runs and all runs evaluated 288/288 timed samples with 0 skips.

## Operational Notes

- Initial RunPod attempts failed before benchmarking because the pod image did not expose a `ninja` executable for SGLang JIT compilation. Installing `ninja-build` fixed startup.
- Initial `--max-samples 384 --unique-samples-per-repeat 96` was invalid for the unique-window harness: 384 loaded rows yielded only 244 unique audio paths but needed 384 unique clips. The measured runs use `--max-samples 1088`.
- The first wrapper version killed only the parent server process, leaving ASR child processes holding GPU memory after the baseline and async runs. The final `runpod_scripts/run_issue831_experiment.sh` version starts the server in its own process group, attempts process-group cleanup even when the parent exits, and captures `nvidia-smi` after acquiring the GPU lock.
- A stale duplicate audio job (`20260621T040539Z_audio_fastpath_1088`) was externally interrupted during cleanup/requeue work and should be ignored. Use `20260621T040448Z_audio_fastpath_1088b`.
- The async request-build run wrote valid `results.json` and `summary.txt`, but its `runner.log` ends with a post-completion shell parse warning from the earlier wrapper revision. The warning happened after metric files were written and does not affect the JSON metrics in the table.
- In the historical pulled artifacts for the measured runs, the initial `nvidia_smi.txt` capture was taken before the GPU lock in the wrapper version used for those jobs. Use the benchmark logs, summaries, and result JSON for the measured comparison; the committed wrapper now captures after lock acquisition.
