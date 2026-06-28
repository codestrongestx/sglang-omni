from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

VENV_PY = "/workspace/repos/sglang-omni/.venv/bin/python"
BASE = "/workspace/repos/sglang-omni-lazy-base-5ef6341-20260628"
CAND = "/workspace/repos/sglang-omni-lazy-candidate-5999ce5-20260628"
MODEL = os.environ.get("STARTUP_PROBE_MODEL", "Qwen/Qwen3-ASR-1.7B")
RESULT_ROOT = Path("/workspace/results/lazy_factory_startup_probe")
RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_asr_h100_5999ce5"
RUN_DIR = RESULT_ROOT / RUN_ID
PORT_BASE = int(os.environ.get("STARTUP_PROBE_PORT_BASE", "18100"))
READY_TIMEOUT_S = float(os.environ.get("STARTUP_PROBE_TIMEOUT_S", "360"))


def run(cmd: list[str], **kwargs):
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def git_value(worktree: str, *args: str) -> str:
    res = run(["git", "-C", worktree, *args], check=True)
    return res.stdout.strip()


def kill_stale() -> None:
    pattern = "sgl-omni serve|sglang_omni.cli serve|sglang.launch_server|stage_workers"
    res = run(["pgrep", "-f", pattern])
    pids = [int(x) for x in res.stdout.split() if x.strip().isdigit()]
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if pids:
        time.sleep(5)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def check_ready(port: int) -> tuple[str, int] | None:
    for endpoint in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{endpoint}", timeout=1.0
            ) as resp:
                code = int(resp.status)
                if 200 <= code < 300:
                    return endpoint, code
        except (urllib.error.URLError, TimeoutError):
            pass
    return None


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def run_once(label: str, worktree: str, ordinal: int, counted: bool) -> dict:
    port = PORT_BASE + ordinal
    log_path = RUN_DIR / f"{ordinal:02d}_{label}.server.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": worktree,
            "HF_HOME": env.get("HF_HOME", "/workspace/.cache/huggingface"),
            "UV_CACHE_DIR": env.get("UV_CACHE_DIR", "/workspace/.cache/uv"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    cmd = [
        VENV_PY,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    kill_stale()
    start = time.perf_counter()
    start_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log_path.open("w", encoding="utf-8") as log:
        log.write("cmd=" + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=worktree,
            preexec_fn=os.setsid,
            text=True,
        )
        ready = None
        error = None
        try:
            while time.perf_counter() - start < READY_TIMEOUT_S:
                poll = proc.poll()
                if poll is not None:
                    error = f"server exited before ready rc={poll}"
                    break
                ready = check_ready(port)
                if ready is not None:
                    break
                time.sleep(0.25)
        finally:
            elapsed = time.perf_counter() - start
            stop_process(proc)
            time.sleep(4)
    result = {
        "ordinal": ordinal,
        "label": label,
        "counted": counted,
        "worktree": worktree,
        "head": git_value(worktree, "rev-parse", "HEAD"),
        "tree": git_value(worktree, "rev-parse", "HEAD^{tree}"),
        "port": port,
        "start_wall_utc": start_wall,
        "ready_s": elapsed if ready is not None else None,
        "ready_endpoint": ready[0] if ready else None,
        "ready_status": ready[1] if ready else None,
        "error": error,
        "server_log": str(log_path),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for label in ("base", "candidate"):
        vals = [
            r["ready_s"]
            for r in results
            if r["counted"] and r["label"] == label and r["ready_s"] is not None
        ]
        vals = [float(v) for v in vals]
        if vals:
            summary[label] = {
                "n": len(vals),
                "values_s": vals,
                "mean_s": sum(vals) / len(vals),
                "min_s": min(vals),
                "max_s": max(vals),
            }
        else:
            summary[label] = {"n": 0, "values_s": []}
    if summary["base"].get("n") and summary["candidate"].get("n"):
        delta = summary["base"]["mean_s"] - summary["candidate"]["mean_s"]
        summary["delta_mean_s"] = delta
        summary["delta_mean_pct"] = delta / summary["base"]["mean_s"] * 100.0
    return summary


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": RUN_ID,
        "model": MODEL,
        "venv_python": VENV_PY,
        "base": {
            "worktree": BASE,
            "head": git_value(BASE, "rev-parse", "HEAD"),
            "tree": git_value(BASE, "rev-parse", "HEAD^{tree}"),
        },
        "candidate": {
            "worktree": CAND,
            "head": git_value(CAND, "rev-parse", "HEAD"),
            "tree": git_value(CAND, "rev-parse", "HEAD^{tree}"),
        },
        "gpu": run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        ).stdout.strip(),
    }
    (RUN_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plan = [
        ("base", BASE, False),
        ("candidate", CAND, True),
        ("base", BASE, True),
        ("candidate", CAND, True),
        ("base", BASE, True),
    ]
    results = []
    try:
        for idx, (label, worktree, counted) in enumerate(plan):
            results.append(run_once(label, worktree, idx, counted))
            (RUN_DIR / "results.json").write_text(
                json.dumps(results, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        kill_stale()
    summary = summarize(results)
    (RUN_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [f"run_dir={RUN_DIR}", f"gpu={meta['gpu']}"]
    for label in ("base", "candidate"):
        item = summary[label]
        lines.append(
            f"{label}: n={item.get('n')} values={item.get('values_s')} "
            f"mean={item.get('mean_s')}"
        )
    if "delta_mean_s" in summary:
        lines.append(f"delta_mean_s={summary['delta_mean_s']:.3f}")
        lines.append(f"delta_mean_pct={summary['delta_mean_pct']:.2f}")
    text = "\n".join(lines) + "\n"
    (RUN_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
