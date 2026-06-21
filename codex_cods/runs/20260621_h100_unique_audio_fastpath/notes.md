# H100 Unique-Audio PCM WAV Fast-Path Candidate

- Direction: bypass heavier audio loading work for PCM WAV input.
- Baseline: `codex_cods/runs/20260621_h100_unique_baseline`.
- Command: `command.sh`; server command: `server_command.sh`.
- Logs: `logs/runner.log`, `logs/server.log`, and `logs/benchmark.log`.
- Metrics: `metrics.json` and `qwen3_asr_results.json`.
- Code audit: exact measured head resolves locally at `refs/heads/runpod/issue831-audio_fastpath`; `code_head.patch.b64` and `code_head_stat.txt` are included for portable review.
- Result: 32.989 samples/s mean throughput, 2.913s mean wall time, 0.925s mean latency, 0.1996 mean RTF, max corpus WER 0.0172, evaluated 288/288, skipped 0.
- Environment audit: `pip_freeze.txt` is a post-hoc `uv pip freeze` snapshot for the same venv because the original `python -m pip freeze` command failed on the pod image.
- Hardware audit: `nvidia-smi.txt` is a post-hoc snapshot from the same RunPod H100 pod after audit restart. The raw pulled artifact still contains the earlier pre-lock runtime capture.
- Router/worker balance: not applicable for this 1-GPU direct ASR server benchmark; the `/workers` endpoint returned 404 and worker objects in the metric JSON are empty.
- Caveat: a stale duplicate run named `20260621T040539Z_audio_fastpath_1088` was interrupted and should be ignored. This trace uses the completed `20260621T040448Z_audio_fastpath_1088b` artifact.
