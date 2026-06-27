# Issue 890 Clean Rerun Protocol

This note is a concise protocol for rerunning the issue #890 attribution matrix
without the cache/order ambiguity in the first cross-over check.

## Where We Are

- The original issue compares two diagonal cells:
  - `A1`: SeedTTS reference audio with the standalone ASR harness.
  - `B2`: generated TTS audio with the TTS WER ASR path.
- That comparison changes both audio source and client harness at the same time.
- Our first cross-over check is useful directional evidence, but it reused the
  same ASR server across cells. Do not use its `A1 -> A2` or `B1 -> B2`
  differences as a clean harness-speed conclusion.

## Clean Environment Rule

Each matrix cell should run against a fresh ASR server process.

The goal is to clear server-side runtime state between cells, including any
in-memory request, prefix, KV, scheduler, or warmed execution state. Do not try
to achieve this by deleting model or dataset caches; those are download/input
caches and should stay stable across cells.

For each cell:

- Stop any existing TTS or ASR server.
- Confirm no old server process is still holding GPU memory.
- Start one fresh Qwen3-ASR server for that cell.
- Run exactly one cell.
- Save the raw logs and result JSON/CSV files.
- Stop the server before moving to the next cell.

## Matrix

Use the same ASR model, commit, hardware class, sample count, ASR concurrency,
and generated-audio artifact for all cells.

| cell | audio source | client harness | purpose |
|---|---|---|---|
| A1 | SeedTTS reference audio | standalone ASR harness | stage-1-like cell |
| B1 | generated TTS audio | standalone ASR harness | audio effect under standalone harness |
| A2 | SeedTTS reference audio | TTS WER ASR path | harness effect with reference audio |
| B2 | generated TTS audio | TTS WER ASR path | stage-2/3-like cell |

Run each cell from a fresh ASR server. If time permits, repeat the full matrix
or rotate the cell order so the conclusion is not tied to one run order.

## What To Report

Report the raw numbers first:

- samples/s
- audio seconds/s
- wall time
- p95 and p99 ASR latency
- sample count and total audio seconds
- corpus WER

Then report only same-harness ratios:

```text
B1 / A1 = generated/reference under standalone harness
B2 / A2 = generated/reference under TTS WER harness
```

Do not use `A2 / A1` or `B2 / B1` as a harness-speed conclusion unless each
cell was run from a fresh server and the result repeats.

## Interpretation Boundary

The clean rerun can support one of these conclusions:

- The original issue numbers are not directly comparable because the audio
  source changes.
- The client harness contributes after controlling the audio source.
- Both audio source and client harness contribute.
- The mismatch does not reproduce under clean conditions and needs CI artifact
  or runner investigation.

Keep the conclusion limited to what the controlled cells show. Avoid explaining
why generated audio differs unless a separate sample-level analysis proves it.
