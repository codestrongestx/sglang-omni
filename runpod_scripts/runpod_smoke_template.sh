#!/usr/bin/env bash
# Portable RunPod launcher template.
#
# Required env:
#   RUNPOD_WORKLOAD_COMMAND  bash command to run after optional repo checkout
#
# Optional env:
#   RUN_NAME                 default: runpod_smoke
#   WORKSPACE_HOME           default: /workspace
#   EXPECTED_GPU_SUBSTRING   default: RTX 4090; empty disables GPU name check
#   ALLOW_GPU_MISMATCH       default: 0
#   REPO_URL                 optional git repo to clone/fetch
#   REPO_REF                 default: main
#   REPO_DIR                 default: $WORKSPACE_HOME/repos/workload
#   UPDATE_SUBMODULES        default: 0
set -euo pipefail

export WORKSPACE_HOME="${WORKSPACE_HOME:-/workspace}"
export VLA_HOME="${VLA_HOME:-$WORKSPACE_HOME}"
export HF_HOME="${HF_HOME:-$WORKSPACE_HOME/caches/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE_HOME/caches/pip}"
export TORCH_HOME="${TORCH_HOME:-$WORKSPACE_HOME/caches/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE_HOME/caches/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE_HOME/caches/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE_HOME/caches/uv}"
export TMPDIR="${TMPDIR:-$WORKSPACE_HOME/tmp}"
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

RUN_NAME="${RUN_NAME:-runpod_smoke}"
EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING-RTX 4090}"
ALLOW_GPU_MISMATCH="${ALLOW_GPU_MISMATCH:-0}"
REPO_URL="${REPO_URL:-}"
REPO_REF="${REPO_REF:-main}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_HOME/repos/workload}"
UPDATE_SUBMODULES="${UPDATE_SUBMODULES:-0}"
RESULT_DIR="${RESULT_DIR:-$WORKSPACE_HOME/results/$RUN_NAME}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"

export RUN_NAME EXPECTED_GPU_SUBSTRING REPO_URL REPO_REF REPO_DIR RESULT_DIR PYTHON_BIN

mkdir -p \
  "$RESULT_DIR" \
  "$WORKSPACE_HOME/repos" \
  "$WORKSPACE_HOME/checkpoints" \
  "$HF_HOME" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$UV_CACHE_DIR" \
  "$TMPDIR"

rm -f \
  "$RESULT_DIR/${RUN_NAME}_run.log" \
  "$RESULT_DIR/env.txt" \
  "$RESULT_DIR/repo_revisions.txt" \
  "$RESULT_DIR/summary.txt" \
  "$RESULT_DIR/status.code"

exec > >(tee -a "$RESULT_DIR/${RUN_NAME}_run.log") 2>&1

finish() {
  local status=$?
  echo "$status" > "$RESULT_DIR/status.code"
  exit "$status"
}
trap finish EXIT

echo "=== start ==="
date -u
hostname

echo "=== system ==="
df -h / "$WORKSPACE_HOME" || true
free -h || true
nvidia-smi || true
if [[ -n "${PYTHON_BIN}" ]]; then
  "$PYTHON_BIN" --version || true
  "$PYTHON_BIN" -m pip --version || true
else
  echo "python missing"
fi

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${EXPECTED_GPU_SUBSTRING}" ]]; then
  if [[ -z "${GPU_NAMES}" ]]; then
    echo "No visible GPUs; expected ${EXPECTED_GPU_SUBSTRING}" >&2
    exit 3
  fi
  if ! grep -q "${EXPECTED_GPU_SUBSTRING}" <<<"${GPU_NAMES}"; then
    if [[ "${ALLOW_GPU_MISMATCH}" != "1" ]]; then
      echo "Refusing GPU mismatch. Expected substring: ${EXPECTED_GPU_SUBSTRING}" >&2
      echo "Visible GPUs:" >&2
      printf '%s\n' "${GPU_NAMES}" >&2
      exit 3
    fi
    echo "ALLOW_GPU_MISMATCH=1 set; continuing despite visible GPUs:"
    printf '%s\n' "${GPU_NAMES}"
  fi
fi

{
  echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname $(hostname)"
  echo "run_name ${RUN_NAME}"
  echo "workspace_home ${WORKSPACE_HOME}"
  echo "result_dir ${RESULT_DIR}"
  echo "expected_gpu_substring ${EXPECTED_GPU_SUBSTRING}"
  echo "repo_url ${REPO_URL}"
  echo "repo_ref ${REPO_REF}"
  echo "repo_dir ${REPO_DIR}"
  echo
  df -h / "$WORKSPACE_HOME" || true
  echo
  free -h || true
  echo
  nvidia-smi || true
  echo
  if [[ -n "${PYTHON_BIN}" ]]; then
    "$PYTHON_BIN" --version || true
    "$PYTHON_BIN" -m pip --version || true
  else
    echo "python missing"
  fi
  echo
  env | sort | grep -E '^(CUDA|HF_HOME|PIP_CACHE_DIR|REPO_|RESULT_DIR|RUN_NAME|TORCH|TRITON|UV_CACHE_DIR|VLA_HOME|WORKSPACE_HOME|XDG_CACHE_HOME)=' || true
} > "$RESULT_DIR/env.txt" 2>&1

echo "=== optional repo checkout ==="
if [[ -n "${REPO_URL}" ]]; then
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone "$REPO_URL" "$REPO_DIR"
  fi
  if ! git -C "$REPO_DIR" diff --quiet || ! git -C "$REPO_DIR" diff --cached --quiet; then
    echo "Refusing to overwrite tracked local changes in ${REPO_DIR}" >&2
    git -C "$REPO_DIR" status --short >&2
    exit 4
  fi
  if git -C "$REPO_DIR" fetch origin "$REPO_REF"; then
    git -C "$REPO_DIR" checkout --detach FETCH_HEAD
  else
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout --detach "$REPO_REF" || git -C "$REPO_DIR" checkout "$REPO_REF"
  fi
  if [[ "${UPDATE_SUBMODULES}" == "1" ]]; then
    git -C "$REPO_DIR" submodule update --init --recursive
  fi
fi

{
  date -u
  if [[ -d "$REPO_DIR/.git" ]]; then
    echo "repo ${REPO_DIR}"
    git -C "$REPO_DIR" status --short
    git -C "$REPO_DIR" log -1 --decorate --stat
  else
    echo "no git repo checked out at ${REPO_DIR}"
  fi
} > "$RESULT_DIR/repo_revisions.txt" 2>&1

echo "=== workload ==="
if [[ -z "${RUNPOD_WORKLOAD_COMMAND:-}" ]]; then
  echo "RUNPOD_WORKLOAD_COMMAND is required" >&2
  exit 2
fi

if [[ -d "$REPO_DIR" ]]; then
  cd "$REPO_DIR"
fi

bash -lc "$RUNPOD_WORKLOAD_COMMAND"

{
  echo "run_name: ${RUN_NAME}"
  echo "status_code: 0"
  echo "result_dir: ${RESULT_DIR}"
  echo "completed_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RESULT_DIR/summary.txt"

echo "=== summary ==="
cat "$RESULT_DIR/summary.txt"
