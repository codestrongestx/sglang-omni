# SPDX-License-Identifier: Apache-2.0
"""Public talker contract: condition embeddings match the reference math."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.minicpm_o.components.sglang_talker import (
    MiniCPMOTalkerForCausalLM,
    MiniCPMTTSProjector,
)
from sglang_omni.models.minicpm_o.talker_model_runner import MiniCPMOTalkerModelRunner

HIDDEN = 8
LLM_DIM = 16
NUM_TEXT = 20
TEXT_EOS = 5
AUDIO_BOS = 6


def _bare_model() -> MiniCPMOTalkerForCausalLM:
    model = object.__new__(MiniCPMOTalkerForCausalLM)
    nn.Module.__init__(model)
    model.text_eos_token_id = TEXT_EOS
    model.audio_bos_token_id = AUDIO_BOS
    model.normalize_projected_hidden = True
    model.emb_text = nn.Embedding(NUM_TEXT, HIDDEN)
    model.projector_semantic = MiniCPMTTSProjector(LLM_DIM, HIDDEN)
    return model


def test_condition_matches_reference_math():
    model = _bare_model()
    tokens = torch.tensor([3, 7, 1], dtype=torch.long)
    hidden = torch.randn(3, LLM_DIM)

    condition = model.build_condition_embeddings(tokens, hidden)

    # reference: emb_text(t) + l2norm(projector(h)), then [text_eos, audio_bos]
    ref = model.emb_text(tokens) + F.normalize(
        model.projector_semantic(hidden), p=2, dim=-1
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, torch.cat([ref, boundary], dim=0))
    assert condition.shape == (5, HIDDEN)


def test_condition_empty_span_is_boundary_only():
    model = _bare_model()
    condition = model.build_condition_embeddings(
        torch.empty(0, dtype=torch.long), torch.empty(0, LLM_DIM)
    )
    boundary = model.emb_text(torch.tensor([TEXT_EOS, AUDIO_BOS]))
    torch.testing.assert_close(condition, boundary)


def test_condition_length_mismatch_raises():
    model = _bare_model()
    with pytest.raises(ValueError, match="length mismatch"):
        model.build_condition_embeddings(torch.tensor([1, 2]), torch.randn(3, LLM_DIM))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runner_penalty_slices_positions_before_filtering(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        runner = object.__new__(MiniCPMOTalkerModelRunner)
        requests = [
            SimpleNamespace(
                data=SimpleNamespace(
                    req=SimpleNamespace(output_ids=history),
                    talker_model_inputs=inputs,
                )
            )
            for history, inputs in [
                ([0] * 16 + [-1] * 14 + [1, 1], {"rep_penalty": 2.0}),
                ([0] * 16, {"rep_penalty": 1.0}),
                ([], {"rep_penalty": 2.0}),
                ([-1, 4] * 8, {"rep_penalty": 2.0}),
                ([0] * 16, {}),
            ]
        ]
        logits = torch.tensor([[2.0, -3.0, 4.0, -5.0]], device=device).repeat(5, 1)
        expected = logits.clone()
        expected[0, 1] *= 4

        runner.process_sampling_logits(
            SimpleNamespace(next_token_logits=logits), requests
        )

        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
