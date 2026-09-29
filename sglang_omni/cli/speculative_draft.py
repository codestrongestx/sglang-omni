# SPDX-License-Identifier: Apache-2.0
"""Start the optional local speculative draft worker."""

import typer


def speculative_draft(
    model_path: str = typer.Option(...),
    target_model_path: str = typer.Option(...),
    socket_path: str = typer.Option(...),
    draft_tokens: int = typer.Option(4, min=1),
    max_sequence_length: int = typer.Option(8192, min=2),
    memory_fraction: float = typer.Option(0.9, min=0.01, max=0.99),
    gpu_id: int = typer.Option(0, min=0),
) -> None:
    # note (Codex): GPU dependencies remain optional for CLI discovery.
    from sglang_omni.models.minicpm_o.speculative_draft import run_draft_worker

    run_draft_worker(
        model_path,
        target_model_path,
        socket_path,
        draft_tokens,
        max_sequence_length,
        memory_fraction,
        gpu_id,
    )
