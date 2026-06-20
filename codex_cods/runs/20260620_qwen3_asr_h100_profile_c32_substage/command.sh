source /workspace/results/qwen3_asr_831/setup/sync_env.sh
cd /workspace/repos/sglang-omni
.venv/bin/python -m benchmarks.eval.benchmark_qwen3_asr_concurrency \
  --port 8000 \
  --max-samples 20 \
  --concurrencies 32 \
  --repeats 3 \
  --warmup \
  --output /workspace/results/qwen3_asr_831/runs/20260620_qwen3_asr_h100_profile_c32_substage/qwen3_asr_results.json
