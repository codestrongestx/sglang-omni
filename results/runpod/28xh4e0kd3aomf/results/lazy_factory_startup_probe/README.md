# Lazy Factory Startup Probe

This directory records the quick Qwen3-ASR `/health` startup probe for the
lazy stage-factory import experiment on pod `28xh4e0kd3aomf`.

## Compared Revisions

- Base: `5ef6341f5fe06ab9bd1156796f22b3c1ff5312bb`
  - Tree: `95c3a4eaca247be2b78652fc5d9ef63774d89536`
  - This is the issue 831 branch base and already includes merged PR 885.
- Candidate: `fa91b4cebffd45f04dbae66590cc0edc49f0f417`
  - Tree: `1e6e97324a50f077d6fb08917c499e1164e9b72d`
  - This is the RunPod-applied equivalent of local commit `5999ce5`.

## Method

The probe started `Qwen/Qwen3-ASR-1.7B` and measured elapsed time from process
launch until `/health` returned HTTP 200. Each run was stopped after readiness.

The corrected RunPod command used:

```bash
PATH=/workspace/repos/sglang-omni/.venv/bin:$PATH \
  /workspace/repos/sglang-omni/.venv/bin/python \
  /workspace/tmp/asr_lazy_factory_startup_probe.py
```

The `PATH` prefix matters because SGLang's JIT compile path needs the `ninja`
binary from the existing venv.

## Runs

- `20260628T014050Z_asr_h100_5999ce5/`
  - Invalid run.
  - Server startup exited before readiness because the process PATH did not
    include the venv `ninja` executable.
  - Kept for audit trail.
- `20260628T015422Z_asr_h100_5999ce5/`
  - Successful corrected run.
  - One uncounted base warmup, then two counted pairs.

Successful summary:

```text
base: n=2 values=[122.16378302592784, 114.697314822115] mean=118.43054892402142
candidate: n=2 values=[83.67632551118731, 91.40337156038731] mean=87.53984853578731
delta_mean_s=30.891
delta_mean_pct=26.08
```
