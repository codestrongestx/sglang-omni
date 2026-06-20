# Issue 831 Optimization Instructions

Use this document to keep Qwen3-ASR optimization runs comparable and auditable.

## Artifact Layout

Keep stable planning and comparison notes in the repo:

```text
codex_cods/
  plan.md
  instructions.md
  baseline.md
  runs/
    <timestamp>_<label>_<gpu>_<mode>/
      meta.json
      command.sh
      git.txt
      env.txt
      nvidia-smi.txt
      logs/
      metrics.json
      notes.md
  comparisons/
    <candidate>_vs_<baseline>.md
```

Keep raw pod pulls under the existing result path:

```text
results/runpod/<pod-id>/
```

Use `runpod_scripts/runpod_jupyter_exec.py` for pod commands. Pull `/workspace/results`
with:

```bash
runpod_scripts/runpod_pull_results.sh <pod-id> <jupyter-password>
```

## Active Baseline

For now, use **H100 as the baseline and development GPU**. The baseline is valid
only for the exact H100 shape we run, so keep the GPU count fixed across
before/after comparisons.

The immediate development baseline is:

- H100 hardware
- same GPU count for baseline and candidate runs
- same benchmark command, dataset, concurrency, warmup, and environment
- same metric set: WER, throughput, latency, RTF, and any profiler evidence

Do not spend time on final CI-shape verification yet. The 2-GPU CI checker shape
is deferred until we have a candidate optimization worth validating.

## Deferred CI Verification

When needed, verify the final candidate with the current `test_qwen3_asr_ci.py`
path: 2-worker router, 2 GPUs, same CI concurrency and dataset. Treat that as a
separate CI-equivalence pass, not as the day-to-day optimization loop.

## Per-Run Capture

Every run directory must include:

```bash
git rev-parse HEAD
git diff --stat
nvidia-smi
python -V
python -m pip freeze  # or equivalent uv/pip package snapshot
```

Also save:

- exact benchmark command
- full server, router, and client logs
- `qwen3_asr_results.json` or equivalent metrics JSON
- short notes on workload shape, GPU type/count, concurrency, samples, and known anomalies

## Iteration Loop

1. Run the baseline on the fixed H100 development shape.
2. Inspect logs/profiler output and select one suspected bottleneck.
3. Make one optimization.
4. Commit the change with a clear message.
5. Run the same benchmark shape again.
6. Compare WER, throughput, latency, RTF, and router worker balance.
7. Write `codex_cods/comparisons/<candidate>_vs_<baseline>.md`.
8. Keep, refine, or revert the change based on measured evidence.

## Proof Standard

A performance claim is valid only when it compares matched runs:

- same GPU type and count
- same code baseline discipline
- same dataset and sample count
- same concurrency and warmup
- same command and environment

For current optimization evidence, matched H100 before/after runs are the source
of truth. For later CI-threshold claims, use the CI repro shape and label it
separately.
