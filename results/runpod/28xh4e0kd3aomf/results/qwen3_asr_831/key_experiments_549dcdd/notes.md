# Qwen3-ASR Issue 831 Key ASR Benchmark

Run root:
`/workspace/results/qwen3_asr_831/key_experiments_549dcdd`

Local mirror:
`results/runpod/28xh4e0kd3aomf/results/qwen3_asr_831/key_experiments_549dcdd`

## Run metadata

- Pod: `28xh4e0kd3aomf`
- Hardware: `NVIDIA H100 80GB HBM3, 81559 MiB`
- Baseline SHA: `1ad75cc1093cac1290b04be2c518f8b7f7b8745b`
- Candidate SHA: `549dcdd65bd2fd461a4469e2fb68df13aa7c2e85`
- ASR model: `Qwen/Qwen3-ASR-1.7B`
- Samples: `1088`
- ASR concurrency: `32`
- Repeats: `3`
- Candidate override: `stages.0.factory_args.request_build_max_workers=4`, `stages.0.factory_args.request_build_max_pending=16`

## Results

| Config | Mean wall, seconds | Mean throughput, samples/s | Mean latency, seconds | p95 latency, seconds | Mean RTF | Max corpus WER | Skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_main_1ad75cc` | `21.6903` | `50.1609` | `0.6320` | `0.8304` | `0.1365` | `0.0133172` | `0` |
| `candidate_549dcdd_workers4_pending16` | `11.3816` | `95.6029` | `0.3281` | `0.4388` | `0.0710` | `0.0133172` | `0` |

Speedup by mean wall time: `1.9057x`.

## Notes

This is the supported `benchmark_qwen3_asr_concurrency` path available in the
current checkout. The benchmark parser in both current `origin/main` and
candidate did not accept the `--unique-timed-samples` flags, so the temporary
wrapper removed those two arguments.

The pod virtualenv did not include `pip`, so `pip_freeze.txt` records
`No module named pip` for these two lightweight ASR benchmark runs. Python
version, GPU metadata, exact SHAs, server/client commands, logs, and result
JSON are present. The generated-audio PR evidence bundle below includes the
same code paths and full per-repeat benchmark outputs.
