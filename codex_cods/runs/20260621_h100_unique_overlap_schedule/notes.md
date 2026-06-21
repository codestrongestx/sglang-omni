# H100 Unique-Audio Overlap-Schedule Candidate

- Direction: add an overlap scheduling measurement hook to test whether request scheduling overlap moves the ASR hot path.
- Baseline: `codex_cods/runs/20260621_h100_unique_baseline`.
- Command: `command.sh`; server command: `server_command.sh`.
- Logs: `logs/runner.log`, `logs/server.log`, and `logs/benchmark.log`.
- Metrics: `metrics.json` and `qwen3_asr_results.json`.
- Code audit: exact measured head resolves locally at `refs/heads/runpod/issue831-overlap_schedule`; `code_head.patch.b64` and `code_head_stat.txt` are included for portable review.
- Result: 27.091 samples/s mean throughput, 3.544s mean wall time, 1.134s mean latency, 0.2454 mean RTF, max corpus WER 0.0215, evaluated 288/288, skipped 0.
- Environment audit: `pip_freeze.txt` is a post-hoc `uv pip freeze` snapshot for the same venv because the original `python -m pip freeze` command failed on the pod image.
- Hardware audit: `nvidia-smi.txt` is a post-hoc snapshot from the same RunPod H100 pod after audit restart. The raw pulled artifact still contains the earlier pre-lock runtime capture.
- Router/worker balance: not applicable for this 1-GPU direct ASR server benchmark; the `/workers` endpoint returned 404 and worker objects in the metric JSON are empty.
- Caveat: the measured improvement is small relative to baseline and is not enough to prioritize this direction by itself.
