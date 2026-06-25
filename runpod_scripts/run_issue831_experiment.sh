#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <label> <worktree> [serve-extra-arg ...]" >&2
  exit 2
fi

label="$1"
worktree="$2"
shift 2

venv_python="${ISSUE831_VENV_PYTHON:-/workspace/repos/sglang-omni/.venv/bin/python}"
venv_sgl_omni="${ISSUE831_VENV_SGL_OMNI:-/workspace/repos/sglang-omni/.venv/bin/sgl-omni}"
venv_bin="$(dirname "$venv_python")"
results_root="${ISSUE831_RESULTS_ROOT:-/workspace/results/qwen3_asr_831/experiments}"
setup_env="${ISSUE831_SETUP_ENV:-/workspace/results/qwen3_asr_831/setup/sync_env.sh}"
model_path="${ISSUE831_MODEL_PATH:-Qwen/Qwen3-ASR-1.7B}"
host="${ISSUE831_HOST:-127.0.0.1}"
port="${ISSUE831_PORT:-8000}"
max_samples="${ISSUE831_MAX_SAMPLES:-1088}"
unique_samples_per_repeat="${ISSUE831_UNIQUE_SAMPLES_PER_REPEAT:-96}"
concurrencies="${ISSUE831_CONCURRENCIES:-32}"
repeats="${ISSUE831_REPEATS:-3}"
startup_timeout_s="${ISSUE831_STARTUP_TIMEOUT_S:-900}"
lock_path="${ISSUE831_GPU_LOCK:-/workspace/tmp/issue831_gpu.lock}"

if [[ ! -d "$worktree" ]]; then
  echo "missing worktree: $worktree" >&2
  exit 2
fi

if [[ -f "$setup_env" ]]; then
  # shellcheck disable=SC1090
  source "$setup_env"
fi
export PATH="$venv_bin:$PATH"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${results_root}/${timestamp}_${label}"
mkdir -p "$run_dir"

exec > >(tee -a "$run_dir/runner.log") 2>&1

echo "run_dir=$run_dir"
echo "label=$label"
echo "worktree=$worktree"
echo "model_path=$model_path"
echo "max_samples=$max_samples"
echo "unique_samples_per_repeat=$unique_samples_per_repeat"
echo "concurrencies=$concurrencies"
echo "repeats=$repeats"
echo "serve_extra_args=$*"

capture() {
  local name="$1"
  shift
  echo "+ $*" | tee "$run_dir/${name}.cmd"
  "$@" >"$run_dir/${name}.txt" 2>&1 || true
}

(
  flock -x 9
  echo "acquired_gpu_lock=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  stale_pids="$(pgrep -f 'sgl-omni serve|sglang_omni.cli serve' || true)"
  if [[ -n "$stale_pids" ]]; then
    echo "killing stale server pids: $stale_pids"
    kill $stale_pids || true
    sleep 5
  fi

  capture git_head git -C "$worktree" rev-parse HEAD
  capture git_status git -C "$worktree" status --short --branch
  capture git_diff_stat git -C "$worktree" diff --stat
  capture nvidia_smi nvidia-smi
  capture python_version "$venv_python" --version
  capture pip_freeze "$venv_python" -m pip freeze

  cat >"$run_dir/server_command.sh" <<EOF
PYTHONPATH="$worktree" "$venv_sgl_omni" serve --model-path "$model_path" --host 0.0.0.0 --port "$port" --log-level info $*
EOF

  cat >"$run_dir/benchmark_command.sh" <<EOF
PYTHONPATH="$worktree" "$venv_python" -m benchmarks.eval.benchmark_qwen3_asr_concurrency --host "$host" --port "$port" --max-samples "$max_samples" --concurrencies "$concurrencies" --repeats "$repeats" --warmup --unique-timed-samples --unique-samples-per-repeat "$unique_samples_per_repeat" --output "$run_dir/results.json"
EOF

  cd "$worktree"
  echo "starting_server=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  setsid env PYTHONPATH="$worktree" "$venv_sgl_omni" serve \
    --model-path "$model_path" \
    --host 0.0.0.0 \
    --port "$port" \
    --log-level info \
    "$@" >"$run_dir/server.log" 2>&1 &
  server_pid=$!
  echo "$server_pid" >"$run_dir/server.pid"

  cleanup() {
    if [[ -n "${server_pid:-}" ]]; then
      echo "stopping_server=$server_pid"
      kill -- "-$server_pid" 2>/dev/null || kill "$server_pid" || true
      sleep 2
      kill -9 -- "-$server_pid" 2>/dev/null || true
      kill -9 "$server_pid" 2>/dev/null || true
      wait "$server_pid" || true
    fi
  }
  trap cleanup EXIT

  deadline=$((SECONDS + startup_timeout_s))
  until curl -fsS "http://${host}:${port}/v1/models" >"$run_dir/models.json" 2>"$run_dir/health.err"; do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "server exited before ready" >&2
      tail -200 "$run_dir/server.log" >&2 || true
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "server readiness timed out after ${startup_timeout_s}s" >&2
      tail -200 "$run_dir/server.log" >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo "server_ready=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  PYTHONPATH="$worktree" "$venv_python" -m benchmarks.eval.benchmark_qwen3_asr_concurrency \
    --host "$host" \
    --port "$port" \
    --max-samples "$max_samples" \
    --concurrencies "$concurrencies" \
    --repeats "$repeats" \
    --warmup \
    --unique-timed-samples \
    --unique-samples-per-repeat "$unique_samples_per_repeat" \
    --output "$run_dir/results.json" 2>&1 | tee "$run_dir/benchmark.log"

  "$venv_python" - <<'PY' "$run_dir/results.json" "$run_dir/summary.txt"
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
data = json.loads(results_path.read_text())
lines = []
for entry in data.get("results", []):
    lines.append(
        "concurrency={concurrency} wall_mean={wall:.3f}s "
        "throughput_mean={throughput:.3f}samples/s wer_max={wer:.4f} "
        "evaluated={evaluated}/{total} skipped={skipped}".format(
            concurrency=entry["concurrency"],
            wall=entry["wall_clock_s"]["mean"],
            throughput=entry["throughput_samples_per_s"]["mean"],
            wer=entry["corpus_wer"]["max"],
            evaluated=entry["evaluated"],
            total=entry["total"],
            skipped=entry["skipped"],
        )
    )
summary_path.write_text("\n".join(lines) + ("\n" if lines else ""))
print(summary_path.read_text(), end="")
PY

  echo "completed=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
) 9>"$lock_path"
