# Agent Instructions

SSH to RunPod pods is often fragile. Prefer
`~/code/runpod-lab/bin/runpod_jupyter` for remote pod commands. For ad hoc
commands, use:

```bash
~/code/runpod-lab/bin/runpod_jupyter exec \
  --base-url https://<pod-id>-8888.proxy.runpod.net \
  --password-file <jupyter-password-file> \
  "<command>"
```

To pull SGLang-Omni results from a pod, use:

```bash
~/code/runpod-lab/bin/runpod_pull.sh \
  --pod-id <pod-id> \
  --password-file <jupyter-password-file> \
  --project sglang-omni \
  --run-name <run-name>
```

It pulls `/workspace/results/sglang-omni/<run-name>/` into
`~/code/runpod-lab/results/sglang-omni/<pod-id>/<run-name>/`.

Always commit strategically so we can track what worked and what did not. Use
clear, complete commit messages.

Before every commit, run `uvx pre-commit run --all-files` and repeat until it
passes without modifying files.

After each commit, launch a headless agent with the latest best model, maximal
thinking effort, and fast mode to review the commit. Reason through its feedback,
fix issues if needed, and ask the same agent to review again. Repeat until no
major issues remain.

When renting GPUs from RunPod, prioritize GPUs that can attach to our network
volume, meaning the same region as the network volume. Stop GPUs when the task is
done to avoid billing.

Do not expose internal plan steps in code.

When committing issue-related work, avoid GitHub autolinks unless intentionally
updating the issue timeline. Write `issue 890` instead of `#890`, `fixes #890`,
or `sgl-project#890` in commit messages.

Make and review each change first as a granular commit on this `-dev` branch, then apply the finalized change to the corresponding branch without the `-dev` suffix and amend that branch's single PR-ready commit, keeping the implementations identical.
