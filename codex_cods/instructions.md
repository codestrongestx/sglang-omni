# Issue 831 Optimization Instructions

Use this document to keep Qwen3-ASR optimization runs comparable and auditable.

## Artifact Layout

Keep stable planning and comparison notes in the repo:

```text
codex_cods/
  plan.md
  instructions.md
  baseline.md
  comparisons/
    <candidate>_vs_<baseline>.md
```

Track compact RunPod evidence and logs under the pulled remote mirror path:

```text
results/
  runpod/
    <pod-id>/
      results/
        <experiment-name>/
          <evidence files>
```

`<pod-id>` is always the RunPod pod ID. Do not rename it to a human label; the
pod ID is part of the audit trail. Inside each pod folder, keep the pulled
remote path under `results/runpod/<pod-id>/results/`, mirroring
`/workspace/results` from the pod. The pull script preserves this shape.

Example:

```text
results/runpod/kjrl6b2adbofl9/results/day13_dreamzero_warm_dit_attribution_h100_nsys_2026_nccl_followup/
```

Prefer these standard file names for compact evidence:

```text
summary.txt
status.code
*.status
preflight.txt
env.txt
env_and_permissions.txt
repo_revisions.txt
latency.json
server_breakdown.json
trace_summary.json
correctness_comparison.json
action_summary.json
action_hashes.json
server_trace.jsonl
server.log
client.log
launcher.log
day<N>_run.log
```

Use profiler-specific subfolders:

```text
torch_profiler/
  profiler/
    *_table.txt
    *_operators.json

nsys/
  nsight/
    *.nsys-rep
  nsys_stats.txt

ncu/
  nsight/
    *.ncu-rep
  ncu_*_stdout.txt
  *.status
```

Track compact evidence and logs under `results/runpod/...`. Do not track by
default:

```text
generated_videos/
torch_profiler/profiler/*_trace.json
nsight/*.sqlite
*.tar.gz
*.pid
```

Same pod plus same experiment name means refresh or complete the same evidence
record. A new question, hardware, backend, rank, profiler, or meaningful config
means creating a new experiment folder with a variant suffix.

Any hard-to-reproduce artifact needed to verify a performance or correctness
claim must be committed under this `results/runpod/<pod-id>/results/...`
mirror. This includes exact commands, hardware/config metadata, SHAs, logs,
metrics, WER outputs, model-info snapshots, and resource/backpressure samples.
Large generated media may stay out of Git when the committed metadata records
the immutable source artifact path and sample manifest needed to audit it.

Use `runpod_scripts/runpod_jupyter_exec.py` for pod commands. Pull `/workspace/results`
with:

```bash
runpod_scripts/runpod_pull_results.sh <pod-id> <jupyter-password>
```

## Meaningful Optimization Target

Issue 831 is about making Qwen3-ASR faster on the TTS WER evaluation hot path.
A candidate is meaningful only if it helps representative CI traffic, where ASR
usually transcribes generated audio that is unique per sample.

Do not count a win as meaningful when the only improvement comes from repeated
benchmark inputs, warmup cache hits, or reusing the same uploaded audio across
timed repeats. Such results may be kept as secondary evidence, but they do not
satisfy the issue by themselves.

A PR-worthy candidate must show:

- final integrated commit, not only a dirty diff
- before/after on a representative unique-audio workload
- no cache-miss or cold-path regression
- WER and failure count remain stable
- exact command, hardware, sample count, concurrency, and environment

## Active Baseline

For now, use **H100 as the baseline and development GPU**. The baseline is valid
only for the exact H100 shape we run, so keep the GPU count fixed across
before/after comparisons.

The immediate development baseline is:

- H100 hardware
- same GPU count for baseline and candidate runs
- same benchmark command, dataset, concurrency, warmup, and environment
- same metric set: WER, throughput, latency, RTF, and any profiler evidence
- at least one unique-audio run, or actual generated-audio TTS WER run, before claiming a general speedup

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
- WER output JSON/CSV for runs that make WER claims
- resource samples and model-info snapshots for runs that make bounded-memory,
  queue-depth, pending-future, or backpressure claims
- launch manifest and per-config server command files for multi-config sweeps
- short notes on workload shape, GPU type/count, concurrency, samples, and known anomalies

When a run is used as PR evidence, commit the tracked `results/runpod/...`
evidence bundle in the same PR branch as the code. A PR description may
summarize the results, but the raw evidence needed to audit the claim should be
available from Git.

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

Cache-hit-only claims must be labeled as such. Do not describe them as general
Qwen3-ASR speedups, and do not use them alone to justify a PR.

For current optimization evidence, matched H100 before/after runs are the source
of truth. For later CI-threshold claims, use the CI repro shape and label it
separately.
