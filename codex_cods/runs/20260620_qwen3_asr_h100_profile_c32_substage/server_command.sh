source /workspace/results/qwen3_asr_831/setup/sync_env.sh
cd /workspace/repos/sglang-omni
.venv/bin/sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000 --log-level info
