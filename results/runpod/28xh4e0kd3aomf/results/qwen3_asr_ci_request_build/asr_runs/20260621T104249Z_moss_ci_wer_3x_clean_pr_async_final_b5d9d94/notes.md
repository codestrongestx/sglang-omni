# Qwen3-ASR async request-build evidence

Run root:
`/workspace/results/qwen3_asr_ci_request_build/asr_runs/20260621T104249Z_moss_ci_wer_3x_clean_pr_async_final_b5d9d94`

Local mirror:
`results/runpod/28xh4e0kd3aomf/results/qwen3_asr_ci_request_build/asr_runs/20260621T104249Z_moss_ci_wer_3x_clean_pr_async_final_b5d9d94`

Tracked evidence bundle:
`results/runpod/28xh4e0kd3aomf/results/qwen3_asr_ci_request_build/asr_runs/20260621T104249Z_moss_ci_wer_3x_clean_pr_async_final_b5d9d94`

This tracked bundle is the PR audit source. The performance-tested code SHA is
`b5d9d946fa07c402f86dff1a5bf7c231298a356b`; later commits on the PR branch may
add only evidence/docs around that tested code.

## Run metadata

- Pod: `28xh4e0kd3aomf`
- Hardware: `NVIDIA H100 80GB HBM3, 81559 MiB`
- Baseline SHA: `8a41fc4da31f68e0515cc81475af1feeaa9c8929`
- Candidate SHA: `b5d9d946fa07c402f86dff1a5bf7c231298a356b`
- ASR model: `Qwen/Qwen3-ASR-1.7B`
- TTS model/artifacts: `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`
- Input artifact dir: `/workspace/results/qwen3_asr_ci_request_build/artifacts/2_seedtts_moss_nonstream_c16_8a41fc4_single_h100/vc_nonstream_c16`
- Samples: `1088` generated audio files
- ASR concurrency: `32`
- Repeats: `3`
- Launch command is recorded in `launch_command.txt`.
- Per-config server commands are recorded in each `server_command.txt`.
- Per-repeat transcribe commands are recorded in each `repeat_*/command.txt`.
- `input/generated.json` and `input/source_tts_speed_results.json` preserve one
  copy of the shared generated-audio artifact metadata used across all repeats.

## End-to-end ASR plus WER results

Warm average uses repeats 2 and 3, after the cold first pass.

| Config | Workers | Max pending | Times, seconds | Warm avg, seconds | Speedup vs baseline warm | WER corpus | Skipped |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| `baseline_main_8a41fc4_sync` | sync | n/a | `39.4055`, `25.8226`, `25.5611` | `25.6918` | `1.0000x` | `0.0262916`, `0.0258729`, `0.0256217` | `0`, `0`, `0` |
| `candidate_b5d9d94_workers2_pending8` | 2 | 8 | `26.7688`, `14.2625`, `14.1911` | `14.2268` | `1.8059x` | `0.0250356`, `0.0264590`, `0.0264590` | `0`, `0`, `0` |
| `candidate_b5d9d94_workers4_pending16` | 4 | 16 | `26.6591`, `12.8965`, `13.1550` | `13.0257` | `1.9724x` | `0.0283848`, `0.0255380`, `0.0262916` | `0`, `0`, `0` |
| `candidate_b5d9d94_workers8_pending32` | 8 | 32 | `29.2861`, `13.7750`, `13.3072` | `13.5411` | `1.8973x` | `0.0272126`, `0.0265427`, `0.0258729` | `0`, `0`, `0` |

Baseline WER range width: `0.0006698`.

WER notes:
- Warm 2-worker WER is `0.0264590`, which is `0.0001675` above the baseline max and inside baseline run-to-run width.
- Warm 4-worker WER range is `0.0255380..0.0262916`, within baseline range.
- Warm 8-worker WER range is `0.0258729..0.0265427`; the high point is `0.0002512` above baseline max and inside baseline run-to-run width.
- Candidate 4-worker repeat 1 has a cold WER spike (`0.0283848`) that did not repeat on warm runs.

## Resource bounds

Aggregated from each `repeat_*/resource_samples.jsonl`.

| Config | Samples | Request-build workers | Max pending configured | Max pending observed | Max backlog configured | Max backlog sampled | Max GPU MiB sampled | Max server RSS MiB sampled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_main_8a41fc4_sync` | 24 | n/a | n/a | n/a | n/a | n/a | 73165 | 1300.1 |
| `candidate_b5d9d94_workers2_pending8` | 17 | 2 | 8 | 8 | 64 | 15 | 72093 | 1294.6 |
| `candidate_b5d9d94_workers4_pending16` | 17 | 4 | 16 | 16 | 64 | 16 | 72321 | 1301.8 |
| `candidate_b5d9d94_workers8_pending32` | 18 | 8 | 32 | 32 | 128 | 0 | 73037 | 1307.5 |

Log scan over `*.log` and `*stderr*` found no:
- `backlog is full`
- `Traceback`
- `ERROR`
- `500 Internal`
- `HTTP/1.1.* 500`

## Verdict

The candidate demonstrates a meaningful end-to-end speedup on the exact generated-audio to Qwen3-ASR transcription to WER workflow.

The best setting in this run is 4 workers with max pending 16:
- Warm end-to-end ASR time improved from `25.6918s` to `13.0257s`.
- Warm speedup is `1.9724x`.
- Skipped samples remained zero.
- Warm WER stayed within baseline run-to-run variance.
- Pending request builds were bounded at the configured cap.
- Sampled backlog remained below its configured cap.

8 workers remains faster than baseline but slower than 4 workers in this sweep, so 4 workers should be treated as the supported default for this candidate unless further runs show otherwise.
