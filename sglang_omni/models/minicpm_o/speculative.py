# SPDX-License-Identifier: Apache-2.0
"""Request-local protocol and retained tokens for target-verified draft decoding."""

import logging
from dataclasses import dataclass, replace
from multiprocessing.connection import Client, Connection
from pathlib import Path

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.model_runner.model_worker import ModelWorker

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DraftRequest:
    request_id: str
    input_ids: list[int]


@dataclass(kw_only=True)
class DraftResponse:
    token_ids: list[int]


@dataclass(kw_only=True)
class DraftReady:
    draft_tokens: int
    max_sequence_length: int
    target_model_path: str


@dataclass(kw_only=True)
class VerifiedTokens:
    token_ids: torch.Tensor
    cpu_token_ids: list[int]
    cache_slots: torch.Tensor
    expected_output_length: int
    expected_last_token: int


@dataclass(kw_only=True)
class SpeculationStatistics:
    rounds: int = 0
    accepted: int = 0
    buffered_consumed: int = 0
    buffered_discarded: int = 0
    ordinary_decodes: int = 0
    allocation_fallbacks: int = 0
    context_fallbacks: int = 0
    unsupported_fallbacks: int = 0


class MiniCPMOSpeculativeDecoder:
    """Expose verified lookahead through the scheduler's one-token contract."""

    def __init__(
        self,
        worker: ModelWorker,
        cache: BasePrefixCache,
        socket_path: str,
        draft_tokens: int,
    ) -> None:
        self.worker: ModelWorker = worker
        self.cache: BasePrefixCache = cache
        self.connection: Connection = Client(socket_path, family="AF_UNIX")
        try:
            ready = self.connection.recv()
        except (EOFError, OSError):
            self.connection.close()
            raise
        if (
            not isinstance(ready, DraftReady)
            or ready.draft_tokens != draft_tokens
            or ready.max_sequence_length < worker.server_args.context_length
            or Path(ready.target_model_path).resolve()
            != Path(worker.model_config.model_path).resolve()
        ):
            self.connection.close()
            raise ValueError("Draft worker configuration does not match the target")
        else:
            pass
        self.draft_tokens: int = draft_tokens
        self.max_sequence_length: int = worker.server_args.context_length
        self.pending: dict[str, VerifiedTokens] = {}
        self.statistics: dict[str, SpeculationStatistics] = {}

    def discard_pending(self, request_id: str) -> int:
        pending = self.pending.pop(request_id, None)
        if pending is not None:
            self.worker.model_runner.token_to_kv_pool_allocator.free(
                pending.cache_slots
            )
            return len(pending.cpu_token_ids)
        else:
            return 0

    def reset_request(self, request_id: str) -> None:
        discarded = self.discard_pending(request_id)
        statistics = self.statistics.pop(request_id, None)
        if statistics is not None:
            statistics.buffered_discarded += discarded
            logger.info(
                f"speculative_request request_id={request_id} rounds={statistics.rounds} "
                f"accepted={statistics.accepted} buffered_consumed={statistics.buffered_consumed} "
                f"buffered_discarded={statistics.buffered_discarded} ordinary_decodes={statistics.ordinary_decodes} "
                f"allocation_fallbacks={statistics.allocation_fallbacks} context_fallbacks={statistics.context_fallbacks} "
                f"unsupported_fallbacks={statistics.unsupported_fallbacks}"
            )
        else:
            pass

    def forward(
        self, batch: ForwardBatch, request: Req
    ) -> GenerationBatchResult | None:
        statistics = self.statistics.setdefault(request.rid, SpeculationStatistics())
        parameters = request.sampling_params
        supported = (
            batch.batch_size == 1
            and batch.capture_hidden_mode == CaptureHiddenMode.NULL
            and parameters.top_k == 1
            and parameters.frequency_penalty == 0
            and parameters.presence_penalty == 0
            and parameters.repetition_penalty == 1
            and parameters.min_new_tokens == 0
            and parameters.logit_bias is None
            and parameters.custom_params is None
            and not request.return_logprob
            and request.grammar is None
            and request.custom_logit_processor is None
        )
        if not supported:
            statistics.buffered_discarded += self.discard_pending(request.rid)
            statistics.ordinary_decodes += 1
            statistics.unsupported_fallbacks += 1
            return None
        else:
            pass
        runner = self.worker.model_runner
        allocator = runner.token_to_kv_pool_allocator
        pending = self.pending.get(request.rid)
        if pending is not None and (
            len(request.output_ids) != pending.expected_output_length
            or request.output_ids[-1] != pending.expected_last_token
        ):
            self.reset_request(request.rid)
            pending = None
            statistics = self.statistics.setdefault(
                request.rid, SpeculationStatistics()
            )
        else:
            pass
        if pending is not None:
            statistics.buffered_consumed += 1
            runner.token_to_kv_pool.move_kv_cache(
                batch.out_cache_loc, pending.cache_slots[:1]
            )
            allocator.free(pending.cache_slots[:1])
            next_token = pending.token_ids[:1]
            pending.expected_last_token = pending.cpu_token_ids.pop(0)
            pending.expected_output_length += 1
            pending.token_ids = pending.token_ids[1:]
            pending.cache_slots = pending.cache_slots[1:]
            if not pending.cpu_token_ids:
                self.pending.pop(request.rid)
            else:
                pass
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                next_token_ids=next_token,
                can_run_cuda_graph=False,
            )
        else:
            pass
        prefix_length = int(batch.seq_lens_cpu[0]) - 1
        span = self.draft_tokens + 1
        if prefix_length + span > self.max_sequence_length:
            statistics.ordinary_decodes += 1
            statistics.context_fallbacks += 1
            return None
        else:
            pass
        token_mapping = runner.req_to_token_pool.req_to_token
        saved_mapping = token_mapping[
            request.kv.req_pool_idx, prefix_length : prefix_length + span
        ].clone()
        scratch_slots = allocator.alloc(self.draft_tokens)
        if scratch_slots is None:
            statistics.ordinary_decodes += 1
            statistics.allocation_fallbacks += 1
            logger.info(
                f"speculative_allocation_fallback request_id={request.rid} prefix_length={prefix_length} "
                f"available_tokens={allocator.available_size()} evictable_tokens={self.cache.evictable_size()} "
                f"protected_tokens={self.cache.protected_size()}"
            )
            return None
        else:
            pass
        retained_count = 0
        try:
            self.connection.send(
                DraftRequest(
                    request_id=request.rid,
                    input_ids=request.origin_input_ids + request.output_ids,
                )
            )
            proposal = self.connection.recv()
            if (
                not isinstance(proposal, DraftResponse)
                or len(proposal.token_ids) != self.draft_tokens
                or any(
                    not isinstance(token_id, int)
                    or not 0 <= token_id < self.worker.model_config.vocab_size
                    for token_id in proposal.token_ids
                )
            ):
                raise ValueError("Invalid draft response")
            else:
                pass
            input_ids = torch.tensor(
                [request.output_ids[-1]] + proposal.token_ids,
                dtype=torch.int64,
                device=batch.input_ids.device,
            )
            positions = torch.arange(
                prefix_length, prefix_length + span, device=input_ids.device
            )
            cache_locations = torch.cat([batch.out_cache_loc, scratch_slots])
            token_mapping[
                request.kv.req_pool_idx, prefix_length : prefix_length + span
            ] = cache_locations
            prefix_lengths = torch.tensor(
                [prefix_length], dtype=torch.int64, device=input_ids.device
            )
            mask = (
                torch.ones(
                    (span, prefix_length + span),
                    dtype=torch.bool,
                    device=input_ids.device,
                )
                .tril(diagonal=prefix_length)
                .flatten()
            )
            verification = replace(
                batch,
                forward_mode=ForwardMode.TARGET_VERIFY,
                global_forward_mode=ForwardMode.TARGET_VERIFY,
                input_ids=input_ids,
                positions=positions,
                seq_lens=prefix_lengths,
                seq_lens_cpu=torch.tensor([prefix_length], dtype=torch.int64),
                seq_lens_sum=prefix_length,
                orig_seq_lens=prefix_lengths,
                out_cache_loc=cache_locations,
                out_cache_loc_virtual=None,
                spec_algorithm=SpeculativeAlgorithm.NGRAM,
                spec_info=NgramVerifyInput(
                    draft_token=input_ids,
                    custom_mask=mask,
                    positions=positions,
                    draft_token_num=span,
                ),
                can_run_decode_cuda_graph=True,
            )
            result = runner.forward(verification)
            predicted_ids = result.logits_output.next_token_logits.argmax(-1)
            predicted_cpu = predicted_ids.cpu().tolist()
            accepted_count = 0
            for predicted, proposed in zip(predicted_cpu, proposal.token_ids):
                if predicted != proposed:
                    break
                else:
                    accepted_count += 1
            statistics.rounds += 1
            statistics.accepted += accepted_count
            if accepted_count:
                self.pending[request.rid] = VerifiedTokens(
                    token_ids=predicted_ids[1 : accepted_count + 1].clone(),
                    cpu_token_ids=predicted_cpu[1 : accepted_count + 1],
                    cache_slots=scratch_slots[:accepted_count],
                    expected_output_length=len(request.output_ids) + 1,
                    expected_last_token=predicted_cpu[0],
                )
                retained_count = accepted_count
            else:
                pass
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                next_token_ids=predicted_ids[:1].clone(),
                can_run_cuda_graph=False,
            )
        finally:
            token_mapping[
                request.kv.req_pool_idx, prefix_length : prefix_length + span
            ] = saved_mapping
            allocator.free(scratch_slots[retained_count:])
