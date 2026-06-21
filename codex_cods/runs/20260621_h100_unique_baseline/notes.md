# H100 Unique-Audio Baseline

- Purpose: baseline for issue 831 unique-audio optimization comparisons.
- Command: `command.sh`; server command: `server_command.sh`.
- Logs: `logs/runner.log`, `logs/server.log`, and `logs/benchmark.log`.
- Metrics: `metrics.json` and `qwen3_asr_results.json`.
- Code audit: exact measured head resolves locally at `refs/heads/runpod/issue831-baseline_unique`; `code_head.patch.b64` and `code_head_stat.txt` are included for portable review.
- Result: 26.154 samples/s mean throughput, 3.671s mean wall time, 1.184s mean latency, 0.2555 mean RTF, max corpus WER 0.0189, evaluated 288/288, skipped 0.
- Environment audit: `pip_freeze.txt` is a post-hoc `uv pip freeze` snapshot for the same venv because the original `python -m pip freeze` command failed on the pod image.
- Hardware audit: `nvidia-smi.txt` is a post-hoc snapshot from the same RunPod H100 pod after audit restart. The raw pulled artifact still contains the earlier pre-lock runtime capture.
- Router/worker balance: not applicable for this 1-GPU direct ASR server benchmark; the `/workers` endpoint returned 404 and worker objects in the metric JSON are empty.
