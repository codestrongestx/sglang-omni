# Qwen3-ASR Async Request Build Candidate

## Direction

Move Qwen3-ASR request construction off the scheduler thread with an opt-in OmniScheduler request-build worker pool. This targets unique generated-audio ASR traffic where every request still needs byte decode, feature extraction, tokenization, multimodal item construction, and SGLang Req creation.

The Qwen3-ASR stage defaults to `request_build_max_workers=4`, while the shared scheduler keeps the existing synchronous behavior when `request_build_max_workers <= 1`.

## Review Fixes

- Pending async request builds are included in active request accounting, so admin abort/update paths can cancel them before they enqueue later.
- Completed builder futures drain only up to the first unfinished future, preserving request arrival order for waiting_queue admission.
- Qwen3-ASR factory tests verify `request_build_max_workers` is forwarded to OmniScheduler, not only present in the signature.

## Evidence

- `python3 -m py_compile sglang_omni/scheduling/omni_scheduler.py tests/unit_test/pipeline/test_scheduler.py sglang_omni/models/qwen3_asr/stages.py tests/unit_test/qwen3_asr/test_pipeline.py` passes.
- `git diff --check` passes.
- Local pytest could not be run on this aarch64 host because the project uv/torch environment fails during dependency build or torch CUDA preload import. H100 measurement is still required before claiming a speedup.
