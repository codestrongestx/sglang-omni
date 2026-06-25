SSH to the pod is likely to be fragile so use runpod_scripts/runpod_jupyter_exec.py to reach out to the pod.

To pull /workspace/results from a pod back to this repo, use runpod_scripts/runpod_pull_results.sh <pod-id> <jupyter-password>; it extracts to results/runpod/<pod-id>/.

Always commit stratgically so that we can keep a track of how things worked out or didn't. And write clear and complte commit message.

 After commit always launch a headless agent with the latest best model and the maximal thinking effort and fast mode to review the commit. Based on its feedback, reason and fix the issues if necessary. Then ask that same agent to review again. Repeat until no major issues found.

When renting gpus from runpod, always priotizie gpus that can be attached to our network volume, aka same region as the network volume.

Do not expose internal plan steps in code.
