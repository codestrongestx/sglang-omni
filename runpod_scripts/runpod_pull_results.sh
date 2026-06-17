#!/usr/bin/env bash
# Pull /workspace/results/ from a Runpod pod into results/runpod/<pod-id>/ locally.
# Uses scripts/runpod_jupyter_exec.py — no SSH dependency.
#
# Usage:
#   scripts/runpod_pull_results.sh <pod-id> <jupyter-password|@password-file> [remote-subdir]
#
# Default remote-subdir is "results". Pulls the whole subtree as a tarball, then
# extracts under results/runpod/<pod-id>/<remote-subdir>/. Idempotent — re-run to
# refresh; existing files are overwritten.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <pod-id> <jupyter-password> [remote-subdir]" >&2
  exit 2
fi

POD_ID="$1"
JUPYTER_PASSWORD_ARG="$2"
REMOTE_SUBDIR="${3:-results}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXEC_SCRIPT="${REPO_ROOT}/scripts/runpod_jupyter_exec.py"
BASE_URL="https://${POD_ID}-8888.proxy.runpod.net"
LOCAL_ROOT="${REPO_ROOT}/results/runpod/${POD_ID}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_CMD=("${REPO_ROOT}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run --with requests --with websocket-client python)
else
  PYTHON_CMD=(python3)
fi

mkdir -p "${LOCAL_ROOT}"

REMOTE_TAR_ABS="/workspace/_pull/${REMOTE_SUBDIR//\//_}.tar.gz"
# Jupyter serves files relative to its root_dir (=/ on Runpod's pytorch image),
# so the fetch path is the absolute path with the leading slash stripped.
REMOTE_TAR_FETCH="${REMOTE_TAR_ABS#/}"
LOCAL_TAR="${LOCAL_ROOT}/.${REMOTE_SUBDIR//\//_}.tar.gz"

if [[ "${JUPYTER_PASSWORD_ARG}" == @* ]]; then
  PASSWORD_ARGS=(--password-file "${JUPYTER_PASSWORD_ARG#@}")
else
  PASSWORD_ARGS=(--password "${JUPYTER_PASSWORD_ARG}")
fi

echo "[pull] tarring /workspace/${REMOTE_SUBDIR} on pod ${POD_ID} ..."
"${PYTHON_CMD[@]}" "${EXEC_SCRIPT}" exec \
  --base-url "${BASE_URL}" \
  "${PASSWORD_ARGS[@]}" \
  --request-timeout 90 \
  --timeout 120 \
  "mkdir -p /workspace/_pull && tar czf ${REMOTE_TAR_ABS} -C /workspace ${REMOTE_SUBDIR} 2>/dev/null && ls -l ${REMOTE_TAR_ABS}"

echo "[pull] fetching tarball ..."
"${PYTHON_CMD[@]}" "${EXEC_SCRIPT}" fetch \
  --base-url "${BASE_URL}" \
  "${PASSWORD_ARGS[@]}" \
  --request-timeout 600 \
  --remote-path "${REMOTE_TAR_FETCH}" \
  --output "${LOCAL_TAR}"

echo "[pull] extracting to ${LOCAL_ROOT} ..."
tar xzf "${LOCAL_TAR}" -C "${LOCAL_ROOT}"
rm -f "${LOCAL_TAR}"

echo "[pull] done. local files:"
find "${LOCAL_ROOT}/${REMOTE_SUBDIR}" -maxdepth 3 -type f -printf "  %p (%s bytes)\n" 2>/dev/null \
  || find "${LOCAL_ROOT}/${REMOTE_SUBDIR}" -maxdepth 3 -type f -exec ls -la {} \;
