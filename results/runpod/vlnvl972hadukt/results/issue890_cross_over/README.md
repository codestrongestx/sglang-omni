# Issue 890 Cross-Over Evidence Mirror

This directory is the pulled RunPod evidence mirror for the issue #890 ASR
audio/harness cross-over run on commit
`4eff258e4a279c61c2b68a06e4949f1f5a17b972`.

The compact metric summary and source-file checksums are recorded in
`codex_cods/causes/issue890_cross_over_evidence.json`.

## Included

- `20260626T134741Z_4eff258_direct_h100/`
  - `direct_asr_server.log`
  - `reference_audio_standalone.json` for A1
  - `reference_audio_tts_wer_harness/{generated.json,asr_speed_results.json,wer_results.csv,wer_results.json}` for A2
  - `generated_audio_tts_wer_harness/{generated.json,speed_results.json}` copied from the generated-audio source artifact for comparison
- `20260626T135711Z_4eff258_direct_h100_generated_only/`
  - `direct_asr_server.log`
  - `generated_audio_standalone.json` for B1
  - `generated_audio_tts_wer_harness/{generated.json,speed_results.json,asr_speed_results.json,wer_results.csv,wer_results.json}` for B2
  - `cross_over_summary.json`, `run_stdout.log`, and `run_error.txt`
  - `reference_audio_tts_wer_harness/generated.json` created from SeedTTS reference audio for the runner setup

## Not Included

- Generated WAV/audio blobs.
- Separate command, environment, or hardware manifest files.
- First-pass stdout capture. The first pass is represented by its server log
  and result JSON/CSV files.

The temporary remote runner did not emit a complete command/provenance bundle.
For reproduction, use the cleaned script committed at
`codex_cods/scripts/issue890_cross_over_remote.py`; treat this mirror as
evidence for the measured outputs, not as a full rerun package.
