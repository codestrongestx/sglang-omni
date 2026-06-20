# Qwen3-ASR H100 baseline, c=32

Official checker-sized workload:

- Model: `Qwen/Qwen3-ASR-1.7B`
- Dataset: SeedTTS EN reference clips
- Samples: 20
- Concurrency: 32
- Repeats: 3
- Warmup: enabled
- GPU: 1x NVIDIA H100 80GB HBM3

Summary from `metrics.json`:

- Evaluated: 20/20
- Skipped/failures: 0
- Corpus WER mean: 0.012687
- Corpus WER range: 0.010381 to 0.017301
- Throughput mean: 38.054 samples/s
- Throughput best: 38.884 samples/s
- Latency mean: 0.465 s
- Latency p95 mean: 0.518 s
- RTF mean: 0.0939
- RTF p95 mean: 0.1208
- Worker balance: not applicable; `/workers` returned 404 and benchmark worker fields were empty.

An earlier no-warmup attempt is retained only under the pulled RunPod results. It showed repeat 1 was dominated by cold overhead, so this warmup run is the clean steady-state baseline.
