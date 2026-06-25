# Qwen3-ASR Issue 831 Generated-Audio PR Evidence

Run root:
`/workspace/results/qwen3_asr_ci_request_build/asr_runs/20260625T130900Z_current_549dcdd_generated_audio`

Local mirror:
`results/runpod/28xh4e0kd3aomf/results/qwen3_asr_ci_request_build/asr_runs/20260625T130900Z_current_549dcdd_generated_audio`

## Run metadata

- Pod: `28xh4e0kd3aomf`
- Hardware: `NVIDIA H100 80GB HBM3, 81559 MiB`
- Baseline SHA: `1ad75cc1093cac1290b04be2c518f8b7f7b8745b`
- Candidate SHA: `549dcdd65bd2fd461a4469e2fb68df13aa7c2e85`
- ASR model: `Qwen/Qwen3-ASR-1.7B`
- TTS model/artifacts: `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`
- Input artifact dir: `/workspace/results/qwen3_asr_ci_request_build/artifacts/2_seedtts_moss_nonstream_c16_8a41fc4_single_h100/vc_nonstream_c16`
- Samples: `1088` generated audio files
- ASR concurrency: `32`
- TTS concurrency metadata: `16`
- Repeats: `3`
- Candidate override: `stages.0.factory_args.request_build_max_workers=4`, `stages.0.factory_args.request_build_max_pending=16`

## End-to-end ASR plus WER results

Warm average uses repeats 2 and 3, after the cold first pass.

| Config | Workers | Max pending | Times, seconds | Warm avg, seconds | Warm throughput, samples/s | Speedup vs baseline warm | WER corpus | Skipped |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `baseline_main_1ad75cc_sync` | sync | n/a | `37.4587`, `23.9902`, `24.0719` | `24.0311` | `45.2749` | `1.0000x` | `0.0251193`, `0.0248681`, `0.0257054` | `0`, `0`, `0` |
| `candidate_549dcdd_workers4_pending16` | 4 | 16 | `24.7505`, `13.2618`, `12.9537` | `13.1078` | `83.0157` | `1.8333x` | `0.0259566`, `0.0253705`, `0.0246169` | `0`, `0`, `0` |

## Verdict

The final candidate demonstrates a matched generated-audio transcription speedup
on the current rebased branch:

- Warm ASR total time improved from `24.0311s` to `13.1078s`.
- Warm speedup is `1.8333x`.
- Warm throughput improved from `45.2749` to `83.0157` samples/s.
- Skipped samples remained zero.
- Candidate warm WER range `0.0246169..0.0253705` overlaps the baseline warm
  range `0.0248681..0.0257054`.
