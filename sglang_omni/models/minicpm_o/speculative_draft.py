# SPDX-License-Identifier: Apache-2.0
"""Persistent single-request draft worker with confirmed-prefix rollback."""

from multiprocessing.connection import Listener

import torch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from transformers import AutoTokenizer

from sglang_omni.models.minicpm_o.speculative import (
    DraftReady,
    DraftRequest,
    DraftResponse,
)
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)


def draft_forward_batch(
    token_ids: torch.Tensor,
    start_position: int,
    cache_slots: torch.Tensor,
    is_prefill: bool,
) -> ForwardBatch:
    sequence_length = start_position + token_ids.numel()
    mode = ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE
    lengths = torch.tensor([sequence_length], dtype=torch.int64, device="cuda")
    batch = ForwardBatch(
        forward_mode=mode,
        batch_size=1,
        input_ids=token_ids,
        req_pool_indices=torch.zeros(1, dtype=torch.int64, device="cuda"),
        seq_lens=lengths,
        seq_lens_cpu=torch.tensor([sequence_length], dtype=torch.int64),
        out_cache_loc=cache_slots[start_position:sequence_length],
        seq_lens_sum=sequence_length,
        orig_seq_lens=lengths,
        positions=torch.arange(start_position, sequence_length, device="cuda"),
        global_forward_mode=mode,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        can_run_decode_cuda_graph=not is_prefill,
    )
    if is_prefill:
        batch.extend_num_tokens = token_ids.numel()
        batch.extend_seq_lens = torch.tensor(
            [token_ids.numel()], dtype=torch.int32, device="cuda"
        )
        batch.extend_prefix_lens = torch.tensor(
            [start_position], dtype=torch.int32, device="cuda"
        )
        batch.extend_start_loc = torch.zeros(1, dtype=torch.int32, device="cuda")
        batch.extend_seq_lens_cpu = [token_ids.numel()]
        batch.extend_prefix_lens_cpu = [start_position]
        batch.extend_logprob_start_lens_cpu = [0]
        batch.is_extend_in_batch = True
    else:
        pass
    return batch


def run_draft_worker(
    model_path: str,
    target_model_path: str,
    socket_path: str,
    draft_tokens: int,
    max_sequence_length: int,
    memory_fraction: float,
    gpu_id: int,
) -> None:
    if draft_tokens < 1 or max_sequence_length <= draft_tokens:
        raise ValueError(
            "Draft token count must be positive and smaller than context capacity"
        )
    else:
        pass
    torch.set_num_threads(1)
    target_vocabulary = AutoTokenizer.from_pretrained(
        target_model_path, trust_remote_code=True
    ).get_vocab()
    draft_vocabulary = AutoTokenizer.from_pretrained(model_path).get_vocab()
    target_to_draft = {
        target_id: draft_vocabulary[token]
        for token, target_id in target_vocabulary.items()
        if token in draft_vocabulary
    }
    draft_to_target = {
        draft_id: target_id for target_id, draft_id in target_to_draft.items()
    }
    allowed_ids = sorted(draft_to_target)
    if not allowed_ids:
        raise ValueError("The target and draft tokenizers have no shared tokens")
    else:
        pass
    arguments = build_sglang_server_args(
        model_path,
        context_length=max_sequence_length,
        mem_fraction_static=memory_fraction,
        max_running_requests=1,
        max_total_tokens=max_sequence_length,
        cuda_graph_max_bs=1,
        disable_prefill_cuda_graph=True,
        disable_overlap_schedule=True,
        trust_remote_code=False,
    )
    worker, tree_cache, request_pool, allocator, configuration = (
        create_sglang_infrastructure(arguments, gpu_id)
    )
    runner = worker.model_runner
    allowed_tensor = torch.tensor(allowed_ids, dtype=torch.int64, device="cuda")
    contiguous_vocabulary = allowed_ids == list(range(len(allowed_ids)))
    cached_ids: list[int] = []
    current_request_id: str | None = None
    with torch.inference_mode(), Listener(socket_path, family="AF_UNIX") as listener:
        cache_slots = allocator.alloc(max_sequence_length)
        if cache_slots is None:
            raise RuntimeError("Draft cache allocation failed")
        else:
            pass
        request_pool.req_to_token[0, :max_sequence_length] = cache_slots
        try:
            with listener.accept() as connection:
                connection.send(
                    DraftReady(
                        draft_tokens=draft_tokens,
                        max_sequence_length=max_sequence_length,
                        target_model_path=target_model_path,
                    )
                )
                while True:
                    try:
                        request = connection.recv()
                    except EOFError:
                        break
                    if not isinstance(request, DraftRequest):
                        raise ValueError("Invalid draft request")
                    else:
                        pass
                    confirmed_ids = [
                        target_to_draft[token_id]
                        for token_id in request.input_ids
                        if token_id in target_to_draft
                    ]
                    if (
                        not confirmed_ids
                        or len(confirmed_ids) + draft_tokens > max_sequence_length
                    ):
                        raise ValueError(
                            "Draft context is empty or exceeds its allocation"
                        )
                    else:
                        pass
                    if current_request_id != request.request_id:
                        cached_ids = []
                        current_request_id = request.request_id
                    else:
                        pass
                    common_length = 0
                    for cached_token, confirmed_token in zip(cached_ids, confirmed_ids):
                        if cached_token != confirmed_token:
                            break
                        else:
                            common_length += 1
                    # note (Codex): Recompute the final confirmed token so rollback never reuses stale logits.
                    start_position = min(common_length, len(confirmed_ids) - 1)
                    correction_ids = confirmed_ids[start_position:]
                    is_prefill = len(correction_ids) > 2
                    proposals: list[torch.Tensor] = []
                    if is_prefill:
                        input_ids = torch.tensor(
                            correction_ids, device="cuda", dtype=torch.int64
                        )
                        result = runner.forward(
                            draft_forward_batch(
                                input_ids, start_position, cache_slots, True
                            )
                        )
                    else:
                        for offset, token_id in enumerate(correction_ids):
                            input_ids = torch.tensor(
                                [token_id], device="cuda", dtype=torch.int64
                            )
                            result = runner.forward(
                                draft_forward_batch(
                                    input_ids,
                                    start_position + offset,
                                    cache_slots,
                                    False,
                                )
                            )
                    for proposal_index in range(draft_tokens):
                        logits = result.logits_output.next_token_logits
                        if contiguous_vocabulary:
                            next_token = logits[:, : len(allowed_ids)].argmax(-1)
                        else:
                            next_token = allowed_tensor[
                                logits.index_select(-1, allowed_tensor).argmax(-1)
                            ]
                        proposals.append(next_token)
                        if proposal_index + 1 < draft_tokens:
                            result = runner.forward(
                                draft_forward_batch(
                                    next_token,
                                    len(confirmed_ids) + proposal_index,
                                    cache_slots,
                                    False,
                                )
                            )
                        else:
                            pass
                    proposed_ids = torch.cat(proposals).cpu().tolist()
                    cached_ids = confirmed_ids + proposed_ids[:-1]
                    connection.send(
                        DraftResponse(
                            token_ids=[
                                draft_to_target[token_id] for token_id in proposed_ids
                            ]
                        )
                    )
        finally:
            allocator.free(cache_slots)
            torch.distributed.destroy_process_group()
