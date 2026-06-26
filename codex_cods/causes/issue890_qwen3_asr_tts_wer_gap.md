# Issue 890: Qwen3-ASR TTS WER Throughput Gap

This note records our working interpretation for issue #890:
Qwen3-ASR standalone SeedTTS evaluation reported about 71 samples/s, while
post-TTS WER transcription in the TTS CI reported about 63 samples/s.

The gap should be investigated as an attribution problem, not treated as a
confirmed server regression. The two measurements are useful to compare as a CI
smoke signal, but they are not an apples-to-apples ASR benchmark.

## What Is Being Compared

The observed run used the same ASR model family, ASR concurrency 32, H100 80GB
hardware, and roughly the same number of SeedTTS utterances. The audio inputs
were different:

- Stage 1 transcribed fixed SeedTTS reference audio.
- Stage 2/3 transcribed generated TTS audio written by the model under test.

The clip count is effectively the same. The difference is not "more clips but
shorter clips". From the issue numbers:

```text
stage 1: 5152.9 audio seconds / 4.74 seconds mean ~= 1088 clips
stage 3: 4788.6 audio seconds / 4.40 seconds mean ~= 1088 clips
```

The generated TTS audio is shorter in aggregate, but it is also a different
workload. Generated audio can have different sample rate, channel count,
container characteristics, silence, pronunciation artifacts, clipping, or other
properties that change preprocessing cost and decoder tail behavior.

Shorter audio alone does not explain the whole observation. If ASR cost scaled
only with audio duration, stage 3 should generally get faster because it has
fewer total audio seconds. Instead the reported audio-normalized throughput is
also lower:

```text
stage 1: 5152.9 audio seconds / 15.33 wall seconds ~= 336 audio seconds/s
stage 3: 4788.6 audio seconds / 17.29 wall seconds ~= 277 audio seconds/s
```

That points to another contributor: fixed per-request overhead, generated-audio
format/content effects, client harness differences, warmup differences, or tail
latency. The p99 latency reported in the issue is especially suspicious
(`1.337s` in stage 3 vs `0.710s` in stage 1).

## Current Thesis

The most likely explanation is generated-audio workload difference, not #885
request-builder configuration failing to propagate.

Static code inspection of the merged #885 path suggests:

- The Qwen3-ASR config defaults use asynchronous request building with
  `request_build_max_workers=2` and `request_build_max_pending=16`.
- The TTS WER fixture launches Qwen3-ASR with empty `worker_extra_args`, so it
  should pick up the model defaults.
- The TTS WER fixture waits for the TTS server to stop and for GPU memory to be
  released before launching the ASR router.
- The TTS WER path should still confirm `tp_size=1`, request-build config,
  worker count, router split, and artifact provenance from runtime logs before
  drawing a final conclusion.

The major known benchmark differences are:

- The standalone ASR CI uses the shared async `aiohttp` benchmark runner.
- The standalone ASR CI excludes warmup requests from timing.
- The TTS WER path uses a blocking `requests` client in a thread pool.
- The TTS WER path currently times the full transcription pass without an
  equivalent excluded warmup window.
- The TTS WER path transcribes generated TTS WAVs instead of the fixed SeedTTS
  reference audio.

The 2026-06-26 cross-over run below makes the harness-only explanation less
likely: when audio is held fixed in a direct single-worker ASR server shape, the
TTS WER client path was not slower than the standalone client path. The stable
directional signal was audio source: generated TTS audio was slower than
SeedTTS reference audio in both client paths.

## Evidence Gathered So Far

Current CI observation for issue #890 gives two live data points, but both
audio source and harness change at the same time:

```text
stage 1:   SeedTTS reference audio + standalone ASR harness ~= 71 samples/s
stage 2/3: generated TTS audio    + TTS WER harness        ~= 63 samples/s
```

That comparison establishes the symptom, but it does not attribute the cause.
The missing information is the two cross-over cases:

```text
SeedTTS reference audio + TTS WER harness
generated TTS audio    + standalone ASR harness
```

Existing pulled RunPod artifacts provide additional but non-final evidence:

- `results/runpod/28xh4e0kd3aomf/results/qwen3_asr_831/key_experiments_549dcdd`
  covers standalone ASR benchmark runs on the fixed SeedTTS reference audio.
  The candidate there used `workers4_pending16` and reached about
  `95.6 samples/s` mean throughput on one direct ASR server.
- `results/runpod/28xh4e0kd3aomf/results/qwen3_asr_ci_request_build/asr_runs/20260625T130900Z_current_549dcdd_generated_audio`
  covers generated TTS audio through the TTS WER transcribe path. The warm
  `workers4_pending16` repeats reached about `82-84 samples/s` on one direct
  ASR server.
- An older generated-audio worker sweep in
  `results/runpod/28xh4e0kd3aomf/results/qwen3_asr_ci_request_build/asr_runs/20260621T104249Z_moss_ci_wer_3x_clean_pr_async_final_b5d9d94`
  showed `workers2_pending8` warm repeats around `76 samples/s`, still well
  above the issue's `~63 samples/s`.

These artifacts show that generated audio plus the TTS WER harness can be fast
when measured on a direct single ASR server with the experimental request-build
candidate. They do not close issue #890 because the reported CI comparison used
the merged defaults, a managed-router shape with two worker replicas, and job
artifacts/logs that are not fully available locally.

New cross-over evidence on the merged issue commit is summarized in
`codex_cods/causes/issue890_cross_over_evidence.json`:

- Hardware: one `NVIDIA H100 80GB HBM3`; the pod was stopped after results were
  pulled and summarized.
- Commit: `4eff258e4a279c61c2b68a06e4949f1f5a17b972`.
- Runner: temporary uploaded copies of
  `codex_cods/scripts/issue890_cross_over_remote.py` were used while the runner
  was being hardened. The committed script is the cleaned reproduction copy.
- Remote results:
  `/workspace/results/issue890_cross_over/20260626T134741Z_4eff258_direct_h100`
  and
  `/workspace/results/issue890_cross_over/20260626T135711Z_4eff258_direct_h100_generated_only`.
- Source-file checksums and compact metrics are committed in
  `issue890_cross_over_evidence.json`; the relevant pulled RunPod logs and
  JSON/CSV result files are tracked under
  `results/runpod/vlnvl972hadukt/results/issue890_cross_over/`.
- ASR server shape: direct `sgl-omni serve`, one ASR stage process on one GPU.
  This is not exact CI parity with the two-worker managed router fixture.

Cross-over matrix, ASR concurrency 32:

| cell | audio | harness | wall s | samples/s | audio s/s | p95 s | p99 s | corpus WER |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A1 | SeedTTS reference | standalone async, warmup 64 | 20.364 | 53.427 | 253.0 | 0.796 | 0.936 | 0.0162 |
| A2 | SeedTTS reference | TTS WER thread pool, no excluded warmup | 12.340 | 88.168 | 417.6 | 0.473 | 0.544 | 0.0133 |
| B1 | generated TTS | standalone async, warmup 64 | 24.040 | 45.257 | 199.2 | 1.040 | 1.405 | 0.0264 |
| B2 | generated TTS | TTS WER thread pool, no excluded warmup | 13.660 | 79.646 | 350.5 | 0.584 | 0.692 | 0.0255 |

Useful ratios:

```text
standalone generated/reference: 45.257 / 53.427 = 0.847  (~15.3% slower)
TTS WER generated/reference:    79.646 / 88.168 = 0.903  (~9.7% slower)
issue reported generated/ref:   63 / 71 ~= 0.887        (~11.3% slower)
```

Interpretation:

- The generated-audio slowdown survives in both harnesses, so generated TTS WAVs
  are a plausible primary explanation for the issue's `~63` vs `~71` gap.
- The issue's observed ratio sits between the two direct-server cross-over
  ratios, which makes an audio-workload explanation quantitatively plausible.
- Holding audio fixed did not show the TTS WER harness as intrinsically slower in
  this run. Do not over-interpret A2 > A1 or B2 > B1 as proof that the TTS path
  is faster; those cells ran after a preceding pass on a direct server and are
  affected by ordering/warm-server effects. The safer conclusion is narrower:
  the harness-only-slower thesis is not supported by this controlled run.
- Generated audio also had worse tail latency and higher WER in matching
  harnesses, which supports a content/workload difference rather than a pure
  sample-count or duration explanation.
- The generated-only pass completed B2 and wrote valid
  `generated_audio_tts_wer_harness/{wer_results.json,asr_speed_results.json}`;
  `run_error.txt` records only a post-run summary typo in the temporary uploaded
  runner. The committed runner has that summary bug fixed.

RunPod status checked on 2026-06-26:

- Account balance was above the `1230 USD` stop threshold before and after the
  cross-over run.
- Existing pod `28xh4e0kd3aomf` was stopped.
- Starting that pod failed twice with RunPod reporting no free GPUs on the host.
- The local mirror includes logs, commands, JSON summaries, CSVs, and metadata,
  but not the generated WAV files. Re-running the cross-over matrix therefore
  needs either the original pod artifacts, GitHub artifact access, or fresh TTS
  audio generation.
- A replacement H100 pod in the network-volume region was launched because the
  original stopped pod could not be restarted. It was stopped after pulling
  results, leaving no H100 workload running for this investigation.

## Minimal Experiment Matrix

Run the smallest matrix that separates audio workload from harness behavior:

```text
Audio set:
A = SeedTTS reference audio
B = generated TTS audio from the stage 2/3 artifacts

Harness:
1 = standalone ASR harness: async aiohttp runner, warmup excluded
2 = TTS WER harness: requests thread pool, current timing behavior
```

Expected cells:

```text
A + 1 = current stage-1 shape
B + 2 = current stage-2/3 shape
A + 2 = harness effect with reference audio
B + 1 = generated-audio effect with standalone harness
```

Interpretation:

- If `A + 2` drops near the post-TTS result, the harness/warmup path explains
  most of the gap.
- If `B + 1` drops near the post-TTS result, the generated TTS audio explains
  most of the gap.
- If both `A + 2` and `B + 1` are only partly slower, the gap is additive.
- If all controlled cells are near the standalone result, the CI observation is
  likely an artifact provenance, runtime environment, or co-residency issue.
- If only generated audio has much worse p99, inspect the slow generated
  samples before changing server code.

Before running the matrix, do the cheap provenance checks:

- Record the exact git commit, image, and model path used by each stage.
- Confirm the ASR router worker count and routing split.
- Confirm `tp_size=1`.
- Confirm runtime request-build settings match the merged defaults.
- Confirm stage 2/3 WER reads the intended `generated.json` and WAV directory.

## Stop Rule

Stop once the gap is attributable to harness differences, generated-audio
workload differences, or a specific runtime/config artifact. Only turn the issue
into an optimization task if the matrix shows a server-side or routing behavior
that remains slower after controlling audio set and client harness.

Current stop-rule status: the direct-server cross-over attributes the direction
and approximate magnitude of the mismatch to generated-audio workload effects.
The remaining open item is exact CI-parity confirmation under the two-worker
managed router, or access to the original GitHub job logs/artifacts. That is a
stronger provenance check, not a reason to suspect #885 configuration
propagation by default.
