# Issue 890 Clean Rerun Protocol

This note is a high-level handoff for rerunning the issue #890 attribution
matrix in a clean environment. It is intentionally concise: the goal is to make
the comparison reproducible without over-prescribing implementation details.

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

In this context, "clean" means clean runtime state, not redownloading models or
regenerating inputs. The input artifacts should be fixed before the matrix
starts, and the ASR server should be restarted between cells.

Keep:

- the same repository commit
- the same model weights
- the same reference audio set
- the same generated TTS artifact
- the same hardware class and concurrency setting

Reset:

- ASR/TTS server processes
- GPU memory held by old server processes
- per-cell runtime logs and result directories
- any per-process warmed state created by a previous cell

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

The important controlled comparisons are:

- `A1 -> B1`: change audio source only, keep the standalone harness.
- `A2 -> B2`: change audio source only, keep the TTS WER ASR path.
- `A1 -> A2`: change harness only, keep reference audio.
- `B1 -> B2`: change harness only, keep generated audio.

## What To Report

Report the raw numbers first, with one row per matrix cell:

- samples/s
- audio seconds/s
- wall time
- p95 and p99 ASR latency
- sample count and total audio seconds
- corpus WER
- repository commit, GPU type, and cell order

Then report the controlled ratios:

```text
B1 / A1 = generated/reference under standalone harness
B2 / A2 = generated/reference under TTS WER harness
A2 / A1 = TTS WER ASR path / standalone harness on reference audio
B2 / B1 = TTS WER ASR path / standalone harness on generated audio
```

Use `A2 / A1` and `B2 / B1` as harness comparisons only when each cell was run
from a fresh server. If the run was not fresh-per-cell, keep those as raw
numbers instead of a harness-speed conclusion.

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

## Handoff Checklist

Before sharing the results, make sure the rerun package contains:

- the four per-cell result files
- server logs for each cell
- pre/post GPU state for each cell
- the exact commit and artifact paths used
- a short summary table with the raw metrics and controlled ratios

The issue comment should state what was controlled, what changed in each row,
and which comparisons are valid. It should not speculate about audio difficulty
or model behavior unless that has been measured separately.
