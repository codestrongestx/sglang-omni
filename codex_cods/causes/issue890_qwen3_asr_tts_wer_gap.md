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

The most likely explanation is benchmark-path mismatch rather than #885 request
builder configuration failing to propagate.

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
