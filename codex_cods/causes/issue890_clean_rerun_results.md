# Issue 890 Clean Rerun Results

This note summarizes one fresh-server-per-cell run of the issue #890 matrix.
Raw logs and JSON/CSV outputs were pulled to:

```text
results/runpod/mssmsnhw57zetj/results/issue890_clean_rerun/20260627T004538Z_4eff258_h100_fresh_server/
```

The generated-audio input WAV tree was not committed because it is 921,629,805
bytes. A checksum manifest for that external artifact is included at:

```text
results/runpod/mssmsnhw57zetj/results/issue890_clean_rerun/20260627T004538Z_4eff258_h100_fresh_server/generated_audio_source_artifact/
```

## Run Setup

- Repository commit: `4eff258e4a279c61c2b68a06e4949f1f5a17b972`
- GPU: `NVIDIA H100 80GB HBM3`
- ASR model: `Qwen/Qwen3-ASR-1.7B`
- ASR concurrency: `32`
- Cell order: `A1`, `B1`, `A2`, `B2`
- Generated-audio source path:
  `/workspace/results/qwen3_asr_ci_request_build/artifacts/2_seedtts_moss_nonstream_c16_8a41fc4_single_h100/vc_nonstream_c16`
- Generated-audio source manifest: 1092 files, 921,629,805 bytes,
  `manifest.json` SHA-256 `bbe8cab0c046a2d341a34515beaf857c8121bb82be7251784a8859b4cd0755df`
- Each cell used a fresh ASR server process.
- All four cells exited with code `0`.
- Every pre/post GPU snapshot showed `0 MiB` allocated.

## Raw Metrics

| cell | audio | harness | samples | audio seconds | samples/s | audio seconds/s | p99 latency | WER |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A1 | SeedTTS reference | standalone ASR | 1088 | 5152.879 | 54.080 | 256.130 | 0.906 | 1.407% |
| B1 | generated TTS | standalone ASR | 1088 | 4788.640 | 46.939 | 206.592 | 1.090 | 2.654% |
| A2 | SeedTTS reference | TTS WER ASR path | 1088 | 5152.879 | 49.348 | 233.716 | 2.182 | 1.347% |
| B2 | generated TTS | TTS WER ASR path | 1088 | 4788.640 | 45.096 | 198.480 | 2.461 | 2.604% |

## Controlled Ratios

Audio source effect:

- `B1 / A1`: `0.868x` samples/s, `0.807x` audio seconds/s.
- `B2 / A2`: `0.914x` samples/s, `0.849x` audio seconds/s.

Harness effect:

- `A2 / A1`: `0.912x` samples/s, `0.912x` audio seconds/s.
- `B2 / B1`: `0.961x` samples/s, `0.961x` audio seconds/s.

## Reading

This run supports treating the original `A1 -> B2` comparison as a mixed
comparison: it changes both audio source and client harness. In this controlled
run, generated audio is slower than reference audio under both harnesses, and
the TTS WER ASR path is slower than the standalone ASR harness under both audio
sets.

To rerun the generated-audio cells exactly, use the same generated-audio source
artifact or files matching `generated_audio_source_artifact/sha256sums.txt`.

This is one clean run, not a repeated statistical study. It should be used as
attribution evidence for issue #890, not as a final performance threshold.
