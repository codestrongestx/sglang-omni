#!/usr/bin/env bash
# Pull /workspace/results/ from a Runpod pod into results/runpod/<pod-id>/ locally.
# Uses runpod_scripts/runpod_jupyter_exec.py — no SSH dependency.
#
# Usage:
#   scripts/runpod_pull_results.sh <pod-id> <@jupyter-password-file|-> [remote-subdir]
#
# Default remote-subdir is "results". Pulls the whole subtree as a tarball, then
# extracts under results/runpod/<pod-id>/<remote-subdir>/. Idempotent — re-run to
# refresh; existing files are overwritten.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <pod-id> <@jupyter-password-file|-> [remote-subdir]" >&2
  exit 2
fi

POD_ID="$1"
JUPYTER_PASSWORD_ARG="$2"
REMOTE_SUBDIR="${3:-results}"

if [[ ! "${REMOTE_SUBDIR}" =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]]; then
  echo "remote-subdir must be a simple relative path under /workspace" >&2
  exit 2
fi
IFS="/" read -r -a REMOTE_SUBDIR_PARTS <<< "${REMOTE_SUBDIR}"
for part in "${REMOTE_SUBDIR_PARTS[@]}"; do
  if [[ "${part}" == "." || "${part}" == ".." ]]; then
    echo "remote-subdir must not contain . or .. path components" >&2
    exit 2
  fi
  if [[ "${part}" == -* ]]; then
    echo "remote-subdir components must not start with -" >&2
    exit 2
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXEC_SCRIPT="${REPO_ROOT}/runpod_scripts/runpod_jupyter_exec.py"
BASE_URL="https://${POD_ID}-8888.proxy.runpod.net"
LOCAL_ROOT="${REPO_ROOT}/results/runpod/${POD_ID}"

if [[ -n "${RUNPOD_PULL_PYTHON:-}" ]]; then
  # shellcheck disable=SC2206
  PYTHON_CMD=(${RUNPOD_PULL_PYTHON})
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
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

PASSWORD_TMP=""
cleanup() {
  if [[ -n "${PASSWORD_TMP}" ]]; then
    rm -f "${PASSWORD_TMP}"
  fi
}
trap cleanup EXIT

if [[ "${JUPYTER_PASSWORD_ARG}" == @* ]]; then
  PASSWORD_ARGS=(--password-file "${JUPYTER_PASSWORD_ARG#@}")
elif [[ "${JUPYTER_PASSWORD_ARG}" == "-" ]]; then
  PASSWORD_TMP="$(mktemp)"
  chmod 600 "${PASSWORD_TMP}"
  cat > "${PASSWORD_TMP}"
  PASSWORD_ARGS=(--password-file "${PASSWORD_TMP}")
else
  echo "raw password arguments are not supported; pass @password-file or - on stdin" >&2
  exit 2
fi

echo "[pull] tarring /workspace/${REMOTE_SUBDIR} on pod ${POD_ID} ..."
"${PYTHON_CMD[@]}" "${EXEC_SCRIPT}" exec \
  --base-url "${BASE_URL}" \
  "${PASSWORD_ARGS[@]}" \
  --request-timeout 90 \
  --timeout 120 \
  "mkdir -p /workspace/_pull && tar czf ${REMOTE_TAR_ABS} -C /workspace -- ${REMOTE_SUBDIR} 2>/dev/null && ls -l ${REMOTE_TAR_ABS}"

echo "[pull] fetching tarball ..."
"${PYTHON_CMD[@]}" "${EXEC_SCRIPT}" fetch \
  --base-url "${BASE_URL}" \
  "${PASSWORD_ARGS[@]}" \
  --request-timeout 600 \
  --remote-path "${REMOTE_TAR_FETCH}" \
  --output "${LOCAL_TAR}"

echo "[pull] extracting to ${LOCAL_ROOT} ..."
"${PYTHON_CMD[@]}" - "${LOCAL_TAR}" "${LOCAL_ROOT}" "${REMOTE_SUBDIR}" <<'PY'
from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

tar_path = Path(sys.argv[1])
local_root = Path(sys.argv[2]).resolve()
remote_subdir = sys.argv[3].strip("/")
target = (local_root / remote_subdir).resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


with tempfile.TemporaryDirectory(dir=local_root) as tmp_dir:
    tmp_root = Path(tmp_dir).resolve()
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            name = member.name
            parts = Path(name).parts
            if name.startswith("/") or ".." in parts:
                raise SystemExit(f"unsafe tar member path: {name!r}")
            if not (name == remote_subdir or name.startswith(remote_subdir + "/")):
                raise SystemExit(f"unexpected tar member outside {remote_subdir!r}: {name!r}")
            if not (member.isdir() or member.isfile()):
                raise SystemExit(f"unsupported tar member type: {name!r}")
            destination = (tmp_root / name).resolve()
            if not is_relative_to(destination, tmp_root):
                raise SystemExit(f"unsafe tar member destination: {name!r}")
        archive.extractall(tmp_root)

    extracted = (tmp_root / remote_subdir).resolve()
    if not is_relative_to(extracted, tmp_root) or not extracted.exists():
        raise SystemExit(f"expected extracted directory missing: {remote_subdir!r}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    os.replace(extracted, target)
PY
rm -f "${LOCAL_TAR}"

echo "[pull] done. local files:"
find "${LOCAL_ROOT}/${REMOTE_SUBDIR}" -maxdepth 3 -type f -printf "  %p (%s bytes)\n" 2>/dev/null \
  || find "${LOCAL_ROOT}/${REMOTE_SUBDIR}" -maxdepth 3 -type f -exec ls -la {} \;
