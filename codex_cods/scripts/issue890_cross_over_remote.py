# SPDX-License-Identifier: Apache-2.0
"""Run the issue #890 ASR audio/harness cross-over experiment.

This script is intended to run on the RunPod checkout. It starts one direct
Qwen3-ASR server, then measures SeedTTS reference audio and previously
generated TTS audio through both available ASR client harnesses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import traceback
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.benchmark_qwen3_asr_concurrency import (
    build_asr_eval_results,
    run_asr_transcription,
)
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_transcribe,
)
from benchmarks.tasks.tts import QWEN3_ASR_MODEL_PATH


ASR_CONCURRENCY = 32
ASR_WARMUP_REQUESTS = ASR_CONCURRENCY * 2


@dataclass
class CellSummary:
    label: str
    audio_set: str
    harness: str
    total_samples: int
    evaluated: int
    skipped: int
    wall_clock_s: float
    throughput_samples_per_s: float
    latency_mean_s: float
    latency_p95_s: float
    latency_p99_s: float | None
    rtf_mean: float
    audio_duration_total_s: float
    audio_seconds_per_wall_second: float
    corpus_wer: float | None
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--model-path", default=QWEN3_ASR_MODEL_PATH)
    parser.add_argument("--repo", default=os.getcwd())
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--generated-output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=1088)
    parser.add_argument(
        "--cells",
        default="A1,A2,B1,B2",
        help="Comma-separated subset of A1,A2,B1,B2 to run.",
    )
    parser.add_argument("--startup-timeout-s", type=int, default=900)
    parser.add_argument("--server-log-name", default="direct_asr_server.log")
    return parser.parse_args()


def wait_for_health(host: str, port: int, proc: subprocess.Popen, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup: rc={proc.returncode}")
        try:
            response = requests.get(url, timeout=2, proxies={"http": None, "https": None})
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        except Exception as exc:  # noqa: BLE001 - startup diagnostics only
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def start_asr_server(args: argparse.Namespace, output_root: Path) -> subprocess.Popen:
    log_path = output_root / args.server_log_name
    log_file = log_path.open("w")
    command = [
        "sgl-omni",
        "serve",
        "--model-path",
        args.model_path,
        "--model-name",
        args.model_path,
        "--host",
        args.bind_host,
        "--port",
        str(args.port),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.repo}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.Popen(
        command,
        cwd=args.repo,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    wait_for_health(args.host, args.port, proc, args.startup_timeout_s)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=30)
    except Exception:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)


def load_generated_samples(generated_dir: Path, max_samples: int) -> list[SampleInput]:
    with (generated_dir / "generated.json").open() as handle:
        entries = json.load(handle)
    samples: list[SampleInput] = []
    for entry in entries[:max_samples]:
        if not entry.get("is_success", False):
            continue
        samples.append(
            SampleInput(
                sample_id=str(entry["sample_id"]),
                ref_text=str(entry["target_text"]),
                target_text=str(entry["target_text"]),
                ref_audio=str(entry["wav_path"]),
            )
        )
    return samples


def audio_total_s_from_samples(samples: list[SampleInput]) -> float:
    total = 0.0
    for sample in samples:
        with open(sample.ref_audio, "rb") as handle:
            total += get_wav_duration(handle.read())
    return total


def audio_total_s_from_entries(entries: list[dict]) -> float:
    return sum(float(entry.get("audio_duration_s") or 0.0) for entry in entries)


async def run_standalone_cell(
    *,
    label: str,
    audio_set: str,
    samples: list[SampleInput],
    host: str,
    port: int,
    model_path: str,
    output_path: Path,
) -> CellSummary:
    outputs, wall_clock_s = await run_asr_transcription(
        samples,
        host=host,
        port=port,
        model_path=model_path,
        lang="en",
        concurrency=ASR_CONCURRENCY,
        warmup=ASR_WARMUP_REQUESTS,
    )
    results = build_asr_eval_results(
        samples,
        outputs,
        wall_clock_s,
        "en",
        model_path=model_path,
        concurrency=ASR_CONCURRENCY,
    )
    output_path.write_text(json.dumps(results, indent=2))
    summary = results["summary"]
    speed = results["speed"]
    audio_total_s = sum(
        float(sample.get("audio_duration_s") or 0.0)
        for sample in results["per_sample"]
        if sample.get("is_success")
    )
    wall_time_s = float(speed["asr_total_time_s"])
    return CellSummary(
        label=label,
        audio_set=audio_set,
        harness="standalone_async_aiohttp_warmup64",
        total_samples=int(summary["total_samples"]),
        evaluated=int(summary["evaluated"]),
        skipped=int(summary["skipped"]),
        wall_clock_s=wall_time_s,
        throughput_samples_per_s=float(speed["throughput_samples_per_s"]),
        latency_mean_s=float(speed["latency_mean_s"]),
        latency_p95_s=float(speed["latency_p95_s"]),
        latency_p99_s=float(speed["latency_p99_s"]),
        rtf_mean=float(speed["rtf_mean"]),
        audio_duration_total_s=audio_total_s,
        audio_seconds_per_wall_second=audio_total_s / wall_time_s,
        corpus_wer=float(summary["corpus_wer"]),
        output_path=str(output_path),
    )


def make_generated_dir_from_reference(samples: list[SampleInput], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for sample in samples:
        with open(sample.ref_audio, "rb") as handle:
            duration_s = get_wav_duration(handle.read())
        entries.append(
            {
                "sample_id": sample.sample_id,
                "target_text": sample.ref_text,
                "wav_path": sample.ref_audio,
                "is_success": True,
                "latency_s": 0.0,
                "audio_duration_s": duration_s,
                "error": "",
            }
        )
    (output_dir / "generated.json").write_text(json.dumps(entries, indent=2))


def make_generated_dir_from_artifact(source_dir: Path, output_dir: Path, max_samples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "generated.json").open() as handle:
        entries = json.load(handle)[:max_samples]
    (output_dir / "generated.json").write_text(json.dumps(entries, indent=2))
    speed_path = source_dir / "speed_results.json"
    if speed_path.exists():
        shutil.copy2(speed_path, output_dir / "speed_results.json")


def run_tts_wer_cell(
    *,
    label: str,
    audio_set: str,
    output_dir: Path,
    port: int,
    model_path: str,
) -> CellSummary:
    config = TtsSeedttsBenchmarkConfig(
        model="issue890-audio-source",
        meta=DATASETS["seedtts"],
        output_dir=str(output_dir),
        lang="en",
        device="cuda:0",
        stream=False,
        concurrency=16,
        disable_tqdm=True,
        asr_concurrency=ASR_CONCURRENCY,
        asr_model_path=model_path,
    )
    result = run_tts_seedtts_transcribe(config, asr_router_port=port)
    summary = result["wer_summary"]
    speed = result["asr_speed"]
    audio_total_s = audio_total_s_from_entries(
        [asdict(output) for output in result["per_sample"] if output.is_success]
    )
    wall_time_s = float(speed["asr_total_time_s"])
    return CellSummary(
        label=label,
        audio_set=audio_set,
        harness="tts_wer_requests_threadpool_no_warmup",
        total_samples=int(summary["total_samples"]),
        evaluated=int(summary["evaluated"]),
        skipped=int(summary["skipped"]),
        wall_clock_s=wall_time_s,
        throughput_samples_per_s=float(speed["asr_throughput_samples_per_s"]),
        latency_mean_s=float(speed["asr_latency_mean_s"]),
        latency_p95_s=float(speed["asr_latency_p95_s"]),
        latency_p99_s=float(speed["asr_latency_p99_s"]),
        rtf_mean=float(speed["asr_rtf_mean"]),
        audio_duration_total_s=audio_total_s,
        audio_seconds_per_wall_second=audio_total_s / wall_time_s,
        corpus_wer=float(summary["wer_corpus"]),
        output_path=str(output_dir / "wer_results.json"),
    )


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    output_root = Path(args.output_root).resolve()
    generated_source = Path(args.generated_output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_cells = {
        cell.strip().upper() for cell in args.cells.split(",") if cell.strip()
    }

    print(f"[issue890] repo={repo}")
    print(f"[issue890] output_root={output_root}")
    print(f"[issue890] generated_source={generated_source}")

    samples = load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=args.max_samples,
        split="en",
    )
    generated_samples = load_generated_samples(generated_source, args.max_samples)
    print(f"[issue890] reference_samples={len(samples)}")
    print(f"[issue890] generated_samples={len(generated_samples)}")
    print(f"[issue890] reference_audio_total_s={audio_total_s_from_samples(samples):.3f}")
    print(
        "[issue890] generated_audio_total_s="
        f"{audio_total_s_from_samples(generated_samples):.3f}"
    )

    ref_tts_dir = output_root / "reference_audio_tts_wer_harness"
    gen_tts_dir = output_root / "generated_audio_tts_wer_harness"
    make_generated_dir_from_reference(samples, ref_tts_dir)
    make_generated_dir_from_artifact(generated_source, gen_tts_dir, args.max_samples)

    summaries: list[CellSummary] = []
    summary_path = output_root / "cross_over_summary.json"

    def write_summary() -> None:
        payload = {
            "config": {
                "repo": str(repo),
                "output_root": str(output_root),
                "generated_output_dir": str(generated_source),
                "model_path": args.model_path,
                "port": args.port,
                "asr_concurrency": ASR_CONCURRENCY,
                "standalone_warmup_requests": ASR_WARMUP_REQUESTS,
                "max_samples": args.max_samples,
                "cells": sorted(selected_cells),
            },
            "summaries": [asdict(summary) for summary in summaries],
        }
        summary_path.write_text(json.dumps(payload, indent=2))

    proc = start_asr_server(args, output_root)
    try:
        if "A1" in selected_cells:
            summaries.append(
                asyncio.run(
                    run_standalone_cell(
                        label="A1_reference_standalone",
                        audio_set="seedtts_reference",
                        samples=samples,
                        host=args.host,
                        port=args.port,
                        model_path=args.model_path,
                        output_path=output_root / "reference_audio_standalone.json",
                    )
                )
            )
            write_summary()
        if "A2" in selected_cells:
            summaries.append(
                run_tts_wer_cell(
                    label="A2_reference_tts_wer",
                    audio_set="seedtts_reference",
                    output_dir=ref_tts_dir,
                    port=args.port,
                    model_path=args.model_path,
                )
            )
            write_summary()
        if "B1" in selected_cells:
            summaries.append(
                asyncio.run(
                    run_standalone_cell(
                        label="B1_generated_standalone",
                        audio_set="generated_tts",
                        samples=generated_samples,
                        host=args.host,
                        port=args.port,
                        model_path=args.model_path,
                        output_path=output_root / "generated_audio_standalone.json",
                    )
                )
            )
            write_summary()
        if "B2" in selected_cells:
            summaries.append(
                run_tts_wer_cell(
                    label="B2_generated_tts_wer",
                    audio_set="generated_tts",
                    output_dir=gen_tts_dir,
                    port=args.port,
                    model_path=args.model_path,
                )
            )
            write_summary()
    except BaseException:
        (output_root / "run_error.txt").write_text(traceback.format_exc())
        raise
    finally:
        stop_server(proc)

    write_summary()
    print("[issue890] wrote", summary_path)
    for summary in summaries:
        print(
            "[issue890] "
            f"{summary.label}: throughput={summary.throughput_samples_per_s:.3f}/s "
            f"wall={summary.wall_clock_s:.3f}s "
            f"audio_s_per_wall={summary.audio_seconds_per_wall_second:.3f} "
            f"lat_p95={summary.latency_p95_s:.3f}s "
            f"lat_p99={summary.latency_p99_s:.3f}s "
            f"wer={summary.corpus_wer:.4f}"
        )


if __name__ == "__main__":
    main()
