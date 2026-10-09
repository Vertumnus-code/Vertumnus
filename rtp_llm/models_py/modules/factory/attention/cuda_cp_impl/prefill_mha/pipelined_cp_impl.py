"""Pipelined prefill context-parallel attention: start attention on partial KV.

This is the "don't wait for the whole all-gather" design.  The key observation
that makes it cheap is a property of the zigzag layout:

  rank j owns LOW_j  = sequence positions [j*h, (j+1)*h)
        and  HIGH_j = sequence positions [N-(j+1)*h, N-j*h)

so an all-gather restricted to the LOW halves lands the *entire first half of the
sequence in sequence order already* (rank-major == position-major for LOW blocks),
and an all-gather restricted to the HIGH halves lands the second half in reverse
block order.  Two facts then remove all the data movement a pipelined scheme
normally pays:

  * Every remote contribution is NON-CAUSAL (the causal boundary only ever falls
    inside a rank's own two blocks), and non-causal attention is invariant to the
    order of the KV axis, so the reversed HIGH buffer needs no un-reversal.
  * Splitting the local contribution into "LOW_r attends to LOW_r" and
    "HIGH_r attends to HIGH_r" leaves remainders that are exactly contiguous
    sequence prefixes:
        low-half  Q remainder = [0, r*h)
        high-half Q remainder = [0, N-(r+1)*h)
    so the remote work is two ragged calls over prefixes of the gathered buffers,
    with no index_select anywhere.

FLOP accounting (exact, no redundancy):
    local        : 2 * h^2/2            = h^2
    q_low remote : h * r*h              = r*h^2
    q_high remote: h * (2cp-r-1)*h      = (2cp-r-1)*h^2
    total        = 2*cp*h^2 -- identical to the non-pipelined optimum.

Schedule (single main stream + torch-managed async collectives, no manual
wait_stream ping-pong -- the upstream overlap variant loses ~0.1 ms to exactly
that):

    issue AG_low (async)        NCCL stream: AG_low then AG_high back to back
    issue AG_high (async)
    local causal x2             <- overlaps AG_low
    wait AG_low ; append_low
    q_low  vs low[0, r*h)       <- overlaps AG_high
    q_high vs low[0, cp*h)      <- overlaps AG_high
    wait AG_high ; append_high
    q_high vs high[r+1 .. cp)
    merge

Restrictions: batch_size == 1 (for batch > 1 the per-sequence sub-ranges of the
gathered buffers stop being contiguous, which would reintroduce the gathers this
design exists to avoid).  Prefix cache is supported via one extra paged call.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed

from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.cascade import merge_state
from flashinfer.page import append_paged_kv_cache

from rtp_llm.models_py.distributed.collective_torch import Group, _get_group
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    get_py_flashinfer_workspace_buffer,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    ParamsBase,
    PyAttentionInputs,
    fill_mla_params,
)

logger = logging.getLogger(__name__)


class PCPPipelinedAttnOp:
    """Two-stage pipelined CP prefill (batch == 1)."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
        backend: str = "auto",
        causal: bool = True,
        kv_layout: str = "NHD",
    ):
        assert causal
        self.attn_inputs = attn_inputs
        self.attn_configs = attn_configs
        self.num_qo_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.seq_size_per_block = attn_configs.tokens_per_block
        self.device = torch.cuda.current_device()
        self.workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.cp_info = attn_inputs.context_parallel_info
        self.cp_rank = parallelism_config.tp_rank
        self.cp_size = parallelism_config.tp_size

        names = ("loc_lo", "loc_hi", "rem_lo", "rem_hi_a", "rem_hi_b")
        self.w = {
            n: BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer, kv_layout="NHD", backend=backend
            )
            for n in names
        }
        self.prefix_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, kv_layout="HND", backend=backend
        )

    @staticmethod
    def eligible(attn_inputs: PyAttentionInputs) -> bool:
        return attn_inputs.context_parallel_info.prefill_cp_chunk_lengths.numel() == 1

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        return attention_inputs.is_prefill and self.eligible(attention_inputs)

    # ----------------------------------------------------------------------- #

    def prepare(self, attention_inputs: PyAttentionInputs) -> ParamsBase:
        cp, r = self.cp_size, self.cp_rank
        chunk = int(self.cp_info.prefill_cp_chunk_lengths[0].item())
        assert self.cp_info.prefill_cp_chunk_lengths.numel() == 1, "batch==1 only"
        n_total = int(self.cp_info.prefill_actual_input_lengths_cpu[0].item())
        pfx = int(attention_inputs.prefix_lengths[0].item())
        h = chunk // 2
        self.h, self.n_total, self.pfx = h, n_total, pfx
        assert n_total == cp * chunk, "zigzag padding is not supported by this variant"

        hkv, d = self.num_kv_heads, self.head_dim
        dev = self.device

        # Two gathered buffers, one per zigzag half.  LOW lands in sequence order;
        # HIGH lands in reverse block order, which is fine (all uses non-causal).
        self._low = torch.empty(cp * h, 2, hkv, d, dtype=torch.bfloat16, device=dev)
        self._high = torch.empty(cp * h, 2, hkv, d, dtype=torch.bfloat16, device=dev)
        self._low_slot = self._low.narrow(0, r * h, h)
        self._high_slot = self._high.narrow(0, r * h, h)
        self._out = torch.empty(
            chunk, self.num_qo_heads, d, dtype=torch.bfloat16, device=dev
        )
        self._pg = _get_group(Group.TP)

        # in-sequence positions for the KV-cache append, per gathered buffer
        low_pos = torch.arange(cp * h, dtype=torch.int32) + pfx
        blk = torch.arange(cp, dtype=torch.int32).repeat_interleave(h)
        off = torch.arange(h, dtype=torch.int32).repeat(cp)
        high_pos = (pfx + n_total - (blk + 1) * h) + off
        self._low_pos = low_pos.to(dev)
        self._high_pos = high_pos.to(dev)
        self._zeros_low = torch.zeros(cp * h, dtype=torch.int32, device=dev)

        # ---- plans ---------------------------------------------------------- #
        common = dict(
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=hkv,
            head_dim_qk=d,
            q_data_type=torch.bfloat16,
        )

        def ind(*vals):
            return torch.tensor(vals, dtype=torch.int32, device=dev)

        # purely local, causal: LOW_r vs LOW_r and HIGH_r vs HIGH_r
        self.w["loc_lo"].plan(qo_indptr=ind(0, h), kv_indptr=ind(0, h),
                              causal=True, **common)
        self.w["loc_hi"].plan(qo_indptr=ind(0, h), kv_indptr=ind(0, h),
                              causal=True, **common)
        # remote, non-causal
        self.lo_rem = r * h  # low-half Q remainder  = sequence [0, r*h)
        self.hi_rem_a = cp * h  # high-half Q, first-half region [0, cp*h)
        self.hi_rem_b = (cp - 1 - r) * h  # high-half Q, HIGH blocks r+1..cp-1
        if self.lo_rem:
            self.w["rem_lo"].plan(qo_indptr=ind(0, h),
                                  kv_indptr=ind(0, self.lo_rem),
                                  causal=False, **common)
        self.w["rem_hi_a"].plan(qo_indptr=ind(0, h),
                                kv_indptr=ind(0, self.hi_rem_a),
                                causal=False, **common)
        if self.hi_rem_b:
            self.w["rem_hi_b"].plan(qo_indptr=ind(0, h),
                                    kv_indptr=ind(0, self.hi_rem_b),
                                    causal=False, **common)

        params = fill_mla_params(
            attention_inputs.prefix_lengths,
            attention_inputs.sequence_lengths,
            self.cp_info.prefill_actual_input_lengths_cpu,
            attention_inputs.kv_cache_kernel_block_id_host,
            self.attn_configs.kernel_tokens_per_block,
        )

        # prefix cache: one paged call, both halves, both over pages [0, pfx)
        self.has_prefix = pfx > 0
        if self.has_prefix:
            page = self.attn_configs.kernel_tokens_per_block
            npg = (pfx + page - 1) // page
            row = attention_inputs.kv_cache_kernel_block_id_host[0][:npg]
            self.prefix_wrapper.plan(
                qo_indptr=ind(0, h, 2 * h),
                paged_kv_indptr=ind(0, npg, 2 * npg),
                paged_kv_indices=torch.cat([row, row]).to(
                    dtype=torch.int32, device=dev
                ),
                paged_kv_last_page_len=ind(
                    pfx - (npg - 1) * page, pfx - (npg - 1) * page
                ),
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=hkv,
                head_dim_qk=d,
                page_size=page,
                causal=False,
                q_data_type=torch.bfloat16,
            )
        return params

    # ----------------------------------------------------------------------- #

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
        hq, hkv, d = self.num_qo_heads, self.num_kv_heads, self.head_dim
        h = self.h
        qkv = qkv.view(qkv.shape[0], -1)
        q = qkv[:, : hq * d].view(-1, hq, d)
        kv = qkv[:, hq * d :].view(-1, 2, hkv, d)
        q_lo, q_hi = q[:h], q[h:]
        kv_lo, kv_hi = kv[:h], kv[h:]

        # stage the two halves and fire both collectives; NCCL orders them
        # low-then-high on its own stream, which is the pipeline.
        self._low_slot.copy_(kv_lo)
        self._high_slot.copy_(kv_hi)
        work_lo = torch.distributed.all_gather_into_tensor(
            self._low, self._low_slot, group=self._pg, async_op=True
        )
        work_hi = torch.distributed.all_gather_into_tensor(
            self._high, self._high_slot, group=self._pg, async_op=True
        )

        # ---- overlaps AG_low: purely local causal work, zero data movement --
        o_loc_lo, l_loc_lo = self.w["loc_lo"].run(
            q_lo, kv_lo[:, 0], kv_lo[:, 1], return_lse=True
        )
        o_loc_hi, l_loc_hi = self.w["loc_hi"].run(
            q_hi, kv_hi[:, 0], kv_hi[:, 1], return_lse=True
        )

        cache_t = kv_cache.kv_cache_base.view(
            -1, 2, hkv, self.seq_size_per_block, d
        )
        if self.has_prefix:
            o_pfx, l_pfx = self.prefix_wrapper.run(q, cache_t, return_lse=True)
            o_loc_lo, l_loc_lo = merge_state(
                o_loc_lo, l_loc_lo, o_pfx[:h], l_pfx[:h]
            )
            o_loc_hi, l_loc_hi = merge_state(
                o_loc_hi, l_loc_hi, o_pfx[h:], l_pfx[h:]
            )

        work_lo.wait()
        append_paged_kv_cache(
            append_key=self._low[:, 0], append_value=self._low[:, 1],
            batch_indices=self._zeros_low, positions=self._low_pos,
            paged_kv_cache=cache_t, kv_indices=params.page_indice_d,
            kv_indptr=params.decode_page_indptr_d,
            kv_last_page_len=params.paged_kv_last_page_len_d, kv_layout="HND",
        )

        # ---- overlaps AG_high -------------------------------------------- #
        if self.lo_rem:
            o_r_lo, l_r_lo = self.w["rem_lo"].run(
                q_lo, self._low[: self.lo_rem, 0], self._low[: self.lo_rem, 1],
                return_lse=True,
            )
            o_lo, l_lo = merge_state(o_loc_lo, l_loc_lo, o_r_lo, l_r_lo)
        else:
            o_lo, l_lo = o_loc_lo, l_loc_lo

        o_r_hi_a, l_r_hi_a = self.w["rem_hi_a"].run(
            q_hi, self._low[:, 0], self._low[:, 1], return_lse=True
        )
        o_hi, l_hi = merge_state(o_loc_hi, l_loc_hi, o_r_hi_a, l_r_hi_a)

        work_hi.wait()
        append_paged_kv_cache(
            append_key=self._high[:, 0], append_value=self._high[:, 1],
            batch_indices=self._zeros_low, positions=self._high_pos,
            paged_kv_cache=cache_t, kv_indices=params.page_indice_d,
            kv_indptr=params.decode_page_indptr_d,
            kv_last_page_len=params.paged_kv_last_page_len_d, kv_layout="HND",
        )

        if self.hi_rem_b:
            # HIGH blocks r+1..cp-1, i.e. rows [(r+1)*h, cp*h) of the reversed
            # buffer.  Non-causal => the reversed KV order is irrelevant.
            seg = self._high[(self.cp_rank + 1) * h :]
            o_r_hi_b, l_r_hi_b = self.w["rem_hi_b"].run(
                q_hi, seg[:, 0], seg[:, 1], return_lse=True
            )
            o_hi, l_hi = merge_state(o_hi, l_hi, o_r_hi_b, l_r_hi_b)

        self._out[:h] = o_lo
        self._out[h:] = o_hi
        return self._out
