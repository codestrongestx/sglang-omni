# Qwen3-ASR H100 request profile, built-in events

Same workload shape as the clean baseline:

- Model: `Qwen/Qwen3-ASR-1.7B`
- Dataset: SeedTTS EN reference clips
- Samples: 20
- Concurrency: 32
- Repeats: 3
- Warmup: enabled
- GPU: 1x NVIDIA H100 80GB HBM3

The built-in request profiler captured 80 requests, including the warmup pass. The default rendered table was too coarse:

- `stage_input_received->stage_complete`: avg 464.744 ms, p95 513.739 ms
- `scheduler_request_build_start->scheduler_request_build_end`: avg 16.760 ms, p95 18.956 ms

The large unaccounted remainder required ASR-specific substage timing.
