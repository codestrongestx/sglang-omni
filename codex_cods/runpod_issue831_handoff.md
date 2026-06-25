# RunPod Issue 831 Handoff

This records the reusable steps and traps from verifying Qwen3-ASR Issue 831
work on the `qwen3-asr-issue831` RunPod pod.

## Pod

- Name: `qwen3-asr-issue831`
- Pod ID: `28xh4e0kd3aomf`
- Shape observed: `1 x NVIDIA H100 80GB HBM3`, datacenter `US-CA-2`
- Jupyter URL shape: `https://28xh4e0kd3aomf-8888.proxy.runpod.net`
- Use `runpod_scripts/runpod_jupyter_exec.py`; do not rely on SSH.

The local environment did not have `runpodctl`. A temporary verified CLI worked:

```bash
curl -fL -o /tmp/runpodctl-v2.6.0-linux-arm64 \
  https://github.com/runpod/runpodctl/releases/download/v2.6.0/runpodctl-linux-arm64
chmod +x /tmp/runpodctl-v2.6.0-linux-arm64
```

Persist API auth with `runpodctl doctor`, then tighten permissions because the
CLI wrote its config world-readable in this environment:

```bash
/tmp/runpodctl-v2.6.0-linux-arm64 doctor
chmod 600 /home/agent/.runpod/config.toml
```

Wake the pod:

```bash
/tmp/runpodctl-v2.6.0-linux-arm64 pod list --all --name qwen3-asr-issue831
/tmp/runpodctl-v2.6.0-linux-arm64 pod start 28xh4e0kd3aomf
```

Extract the Jupyter password from pod metadata into a `0600` temp file, but do
not commit it or paste it into tracked notes.

## Verification Checkout

The existing pod repo at `/workspace/repos/sglang-omni` was dirty and behind, so
use a separate checkout:

```bash
VERIFY=/workspace/repos/sglang-omni-verify-5560337
git clone --reference-if-able /workspace/repos/sglang-omni \
  https://github.com/codestrongestx/sglang-omni.git "$VERIFY"
cd "$VERIFY"
git checkout --detach b0f94dd817eb1f816dd6b0bd4fb4a0989b6f8cda
git config user.name codestrongestx
git config user.email 238223145+codestrongestx@users.noreply.github.com
git am --3way /workspace/sglang_omni_5560337.patch
```

The recreated commit hash on the pod can differ, but verify the tree hash
matches local `5560337`:

```bash
git rev-parse HEAD^{tree}
```

For this run, the tree was `57fab4f05c1b095184fbeb3f5b6e6ce0b3e084ac`.

## Tests Run

Focused unit tests passed on the H100 pod:

```bash
cd /workspace/repos/sglang-omni-verify-5560337
PYTHONPATH=$PWD /workspace/repos/sglang-omni/.venv/bin/python -m pytest \
  tests/unit_test/qwen3_asr/test_pipeline.py \
  tests/unit_test/qwen3_asr/test_request_builders.py -q
```

Result: `11 passed` in about 91 seconds.

## Benchmark Flags Gotcha

`runpod_scripts/run_issue831_experiment.sh` in this checkout passes these flags:

```text
--unique-timed-samples
--unique-samples-per-repeat
```

But `benchmarks/eval/benchmark_qwen3_asr_concurrency.py` at the verified commit
does not accept them. The supported benchmark flags are the basic parser set:
`--host`, `--port`, `--meta`, `--lang`, `--max-samples`, `--concurrencies`,
`--repeats`, `--model-path`, `--warmup`, and `--output`.

Use a temporary copy of the harness for this commit:

```bash
sed '/--unique-timed-samples/d; /--unique-samples-per-repeat/d' \
  runpod_scripts/run_issue831_experiment.sh \
  > /workspace/tmp/run_issue831_experiment_no_unique_5560337.sh
chmod +x /workspace/tmp/run_issue831_experiment_no_unique_5560337.sh
```

Then run with request-build workers enabled through the existing dotted config
overrides for the ASR stage kwargs, not via env:

```bash
export PYTHONPATH=/workspace/repos/sglang-omni-verify-5560337
export ISSUE831_VENV_PYTHON=/workspace/repos/sglang-omni/.venv/bin/python
export ISSUE831_VENV_SGL_OMNI=/workspace/repos/sglang-omni/.venv/bin/sgl-omni
export ISSUE831_RESULTS_ROOT=/workspace/results/qwen3_asr_831/experiments
export ISSUE831_PORT=18000
export ISSUE831_MAX_SAMPLES=1088
export ISSUE831_CONCURRENCIES=32
export ISSUE831_REPEATS=3
bash /workspace/tmp/run_issue831_experiment_no_unique_5560337.sh \
  candidate_5560337_supported_benchmark_w4p16_fg \
  /workspace/repos/sglang-omni-verify-5560337 \
  stages.0.factory_args.request_build_max_workers=4 \
  stages.0.factory_args.request_build_max_pending=16
```

Observed run directory:

```text
/workspace/results/qwen3_asr_831/experiments/<timestamp>_candidate_5560337_supported_benchmark_w4p16
```

## Operational Traps

- The pod has one H100. The stock `tests/test_model/test_qwen3_asr_ci.py`
  managed-router helper defaults to two one-GPU workers, so it is not the right
  shape for this pod.
- Commands backgrounded inside the Jupyter terminal can be killed when that
  terminal/session closes. Prefer foreground runs or a fully detached launcher.
- A killed background run left an orphan ASR stage process holding about 72 GB
  of GPU memory. Check and clean before retrying:

```bash
nvidia-smi
ps -ef | grep -E 'sgl-omni|sglang_omni|stage_workers|benchmark_qwen3' | grep -v grep
kill <pid>
sleep 5
kill -9 <pid>
```

- Port `8000` may be left in use; the server can auto-select another port, but
  the harness still probes the configured port. Prefer setting
  `ISSUE831_PORT` to a known-free high port for serious timing runs.
- `runpod_scripts/runpod_pull_results.sh` expects `@password-file` or stdin
  (`-`) for the Jupyter password, not a raw password argument.
