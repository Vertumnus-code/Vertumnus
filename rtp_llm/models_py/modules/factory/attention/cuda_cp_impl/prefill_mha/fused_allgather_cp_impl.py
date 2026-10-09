"""Optimized prefill context-parallel attention operators for RTP-LLM.

Drop-in replacements for ``PCPAllGatherAttnOp`` / ``PCPAllGatherOverlapAttnOp`` /
``PCPAll2AllAttnOp``: same constructor signature, same ``prepare``/``forward``
contract, same KV-cache side effect (the full sequence's new K/V is appended to
the local paged cache), same zigzag input layout.

Every design decision below is backed by a measurement on 8x H20-3e
(bf16, 64 q heads / 8 kv heads / head_dim 128, page 64):

1. *All-gather is the right primitive, not a p2p ring.*  Per layer at N=2048/cp=8,
   a fused all-gather costs 0.054 ms while cp-1 rounds of ring p2p cost 0.44 ms
   (0.31 ms even with ``batch_isend_irecv``).  NCCL's all-gather beats hand-rolled
   p2p by 6-8x at these message sizes, and every rank must receive the whole
   sequence's KV anyway (it has to write it all to its local paged cache), so the
   all-gather volume is a hard floor rather than an avoidable cost.

2. *Do not overlap comm on a second stream.*  The upstream "overlap" variant is a
   net regression (0.607 vs 0.575 ms at N=4096): it hides ~0.08 ms of NCCL but
   pays more than that in cross-stream ``wait_stream`` bubbles plus the extra
   non-causal attention launches and ``merge_state`` calls.  Minimising total work
   on one stream wins.

3. *Fuse K and V into one all-gather, into a persistent buffer.*  Two separate
   ``all_gather`` calls that each freshly ``torch.zeros`` their output cost
   0.0896 ms at N=2048; one fused all-gather into a preallocated buffer costs
   0.0544 ms -- 39% less, for free.

4. *Fold the zigzag unshuffle into ``append_paged_kv_cache`` positions.*
   ``positions`` may be an arbitrary non-monotonic permutation (verified), so the
   separate "restore" gather that upstream needs disappears.

5. *Let the paged KV cache be the KV source for attention.*  Rank r's low-half Q
   needs sequence prefix [0, prefix+(r+1)h) and its high-half Q needs
   [0, prefix+N-rh) -- both contiguous prefixes.  Two requests in one paged
   prefill call can point at *overlapping* page ranges (verified legal and
   bit-accurate), so the four ``index_select`` calls that build k0/k1/v0/v1, both
   ``merge_state`` calls and the separate prefix wrapper all disappear, and prefix
   caching is handled for free.

6. *q, k and v may be strided views* (verified bit-exact on both the ragged and
   paged wrappers), so q is read straight out of the packed qkv tensor with no
   ``contiguous()`` copy, and k/v are read straight out of the packed gathered
   buffer.

Net effect: 4 kernels per forward (1 small copy, 1 NCCL, 1 append, 1 attention)
against upstream's ~11-12.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional

import torch
import torch.distributed

from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
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


# --------------------------------------------------------------------------- #
# process-level scratch (grown on demand, reused across steps)
# --------------------------------------------------------------------------- #

_KV_GATHER_BUF: Optional[torch.Tensor] = None
_OUT_BUF: Optional[torch.Tensor] = None


def _kv_gather_buffer(slots: int, num_kv_heads: int, head_dim: int, device):
    """Packed [slots, 2, H_kv, D] bf16 gather destination, reused across steps."""
    if os.environ.get("RTP_CP_FRESH_GATHER") == "1":
        return torch.empty(
            slots, 2, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
    global _KV_GATHER_BUF
    need = slots * 2 * num_kv_heads * head_dim
    if _KV_GATHER_BUF is None or _KV_GATHER_BUF.numel() < need:
        _KV_GATHER_BUF = torch.empty(need, dtype=torch.bfloat16, device=device)
    return _KV_GATHER_BUF[:need].view(slots, 2, num_kv_heads, head_dim)


def _out_buffer(tokens: int, num_qo_heads: int, head_dim: int, device):
    """Attention output destination, so flashinfer does not torch.zeros one."""
    if os.environ.get("RTP_CP_FRESH_OUT") == "1":
        return None
    global _OUT_BUF
    need = tokens * num_qo_heads * head_dim
    if _OUT_BUF is None or _OUT_BUF.numel() < need:
        _OUT_BUF = torch.empty(need, dtype=torch.bfloat16, device=device)
    return _OUT_BUF[:need].view(tokens, num_qo_heads, head_dim)


# --------------------------------------------------------------------------- #
# shared prepare-time geometry
# --------------------------------------------------------------------------- #


class _CPGeometry:
    """Everything derivable once per batch from the CP metadata.

    Uses only fields the engine already populates: ``prefill_cp_chunk_lengths``,
    ``prefill_actual_input_lengths_cpu``, ``prefill_qkv_restore_indice``,
    ``prefill_qkv_padding_mask`` and ``prefix_lengths``.
    """

    def __init__(self, attn_inputs: PyAttentionInputs, cp_size: int, cp_rank: int,
                 page_size: int, device):
        cp_info = attn_inputs.context_parallel_info
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.page_size = page_size
        self.device = device

        self.chunk_lengths: List[int] = cp_info.prefill_cp_chunk_lengths.tolist()
        self.new_lengths: List[int] = cp_info.prefill_actual_input_lengths_cpu.tolist()
        self.prefix_lengths: List[int] = attn_inputs.prefix_lengths.tolist()
        self.batch_size = len(self.chunk_lengths)
        self.half_lengths = [c // 2 for c in self.chunk_lengths]
        self.tokens_local = sum(self.chunk_lengths)
        self.tokens_total = sum(self.new_lengths)
        self.has_prefix = any(p > 0 for p in self.prefix_lengths)

        # The gathered buffer is indexed in PADDED zigzag space: when the new token
        # count is not a multiple of 2*cp the engine pads it up to
        # chunk_lengths[b]*cp_size, and ships restore_indice / padding_mask over
        # that padded space.  We append *every* gathered slot, pads included, at
        # its padded in-sequence position.  Two reasons:
        #   1. Correctness.  The attention reads KV straight out of the paged
        #      cache over the padded range (see half_kv_lengths), so any slot in
        #      that range which is never written would be read uninitialised --
        #      verified to produce NaN output with a NaN-poisoned cache.  Writing
        #      the pads makes every slot that can be read a written slot.
        #   2. Cost.  It removes the compaction gather the padded path would
        #      otherwise need, so padded and aligned lengths cost the same.
        # The pad positions always fit inside pages the engine has already
        # allocated: padding < 2*cp, and 2*cp divides page_size, so the free tail
        # of the last page is always at least the padding (and padding is zero
        # whenever the real length is itself a multiple of page_size).
        self.restore = cp_info.prefill_qkv_restore_indice.to(torch.int64)
        self.gathered_slots = self.cp_size * self.tokens_local
        self.padded_lengths = [c * self.cp_size for c in self.chunk_lengths]
        assert self.restore.numel() == sum(self.padded_lengths), (
            f"restore_indice ({self.restore.numel()}) should cover the padded "
            f"sequence ({sum(self.padded_lengths)})"
        )

        # Build these ON DEVICE.  Constructed on the CPU they cost 311-355 ms per
        # prefill step inside the engine's worker thread at a 64k prompt (vs ~0.5 ms
        # for the identical code in a standalone process), which silently ate the
        # whole end-to-end win: prepare() runs once per prefill step, so an
        # operator-level benchmark never sees it.  On device it is ~0.1 ms.
        parts_b = [
            torch.full((p,), b, dtype=torch.int32, device=device)
            for b, p in enumerate(self.padded_lengths)
        ]
        parts_p = [
            torch.arange(p, dtype=torch.int32, device=device)
            + self.prefix_lengths[b]
            for b, p in enumerate(self.padded_lengths)
        ]
        batch_of_pos = parts_b[0] if len(parts_b) == 1 else torch.cat(parts_b)
        inseq_of_pos = parts_p[0] if len(parts_p) == 1 else torch.cat(parts_p)
        # scatter: gathered slot -> (batch, in-sequence position)
        self.ag_batch = torch.empty(
            self.gathered_slots, dtype=torch.int32, device=device
        )
        self.ag_pos = torch.empty(
            self.gathered_slots, dtype=torch.int32, device=device
        )
        self.ag_batch[self.restore] = batch_of_pos
        self.ag_pos[self.restore] = inseq_of_pos

        self.seq_batch = batch_of_pos
        self.seq_pos = inseq_of_pos

    # -- per-request KV lengths of the two zigzag halves ---------------------- #

    def half_kv_lengths(self, b: int):
        """(low-half kv length, high-half kv length) for sequence ``b``.

        The zigzag block boundaries live in the *padded* index space: when the new
        token count is not a multiple of 2*cp the engine pads it up to
        ``chunk_lengths[b] * cp_size`` before splitting, and rank r's high block
        covers padded positions [padded - (r+1)h, padded - r*h).  Its last query
        therefore needs KV up to padded - r*h, clamped to the number of tokens
        that actually exist.  Mixing the padded ``h`` with the real new length
        (which is what an earlier version did) makes every rank r >= 1 short by
        exactly the padding amount -- a small but real error that changes the
        output, and one that only shows up for lengths that are not a multiple of
        2*cp, i.e. essentially every real request.
        """
        h = self.half_lengths[b]
        pfx = self.prefix_lengths[b]
        padded = self.chunk_lengths[b] * self.cp_size
        # Both lengths are in PADDED index space and must NOT be clamped to the
        # real token count: flashinfer aligns the causal mask bottom-right, so
        # request i attends to j <= i + (kv_len - q_len), and correctness requires
        # kv_len - q_len == the block's padded start position.  Clamping kv_len to
        # the real length shifts that boundary by the padding amount and silently
        # drops the last few keys for every rank -- which is exactly the bug this
        # replaced.  Reading a few padded tail slots is harmless: they are
        # causally masked for every real query row (only the pad query rows, whose
        # output the engine discards, can see them), and the upstream all-gather
        # operator relies on the same property.
        lo = pfx + (self.cp_rank + 1) * h
        hi = pfx + padded - self.cp_rank * h
        return lo, hi

    def paged_plan_tensors(self, block_ids_host: torch.Tensor):
        """Page table for 2*batch requests (low half, high half per sequence).

        Requests are emitted in local-token order -- for each sequence the low
        block then the high block -- which is exactly the order the zigzag layout
        hands us q in, so no q permutation and no output scatter are needed.
        """
        qo = [0]
        indptr = [0]
        last_len: List[int] = []
        idx_parts: List[torch.Tensor] = []
        for b in range(self.batch_size):
            h = self.half_lengths[b]
            row = block_ids_host[b]
            if h == 0:
                continue  # nothing to compute for an empty sequence
            for kv_len in self.half_kv_lengths(b):
                npg = (kv_len + self.page_size - 1) // self.page_size
                if npg == 0:
                    # flashinfer reads garbage for a zero-page request on the FA3
                    # backend; such a request has no work anyway.
                    continue
                idx_parts.append(row[:npg])
                indptr.append(indptr[-1] + npg)
                last_len.append(kv_len - (npg - 1) * self.page_size)
                qo.append(qo[-1] + h)
        # qo_indptr / kv_indptr / last_page_len are consumed host-side by plan();
        # building them on CPU avoids a blocking D2H copy per plan.  The page
        # index list stays on device and must share kv_indptr's dtype.
        return (
            torch.tensor(qo, dtype=torch.int32),
            torch.tensor(indptr, dtype=torch.int32),
            torch.cat(idx_parts).to(dtype=torch.int32, device=self.device),
            torch.tensor(last_len, dtype=torch.int32),
        )


# --------------------------------------------------------------------------- #
# V1: fused all-gather + single paged prefill  (general: any batch / prefix)
# --------------------------------------------------------------------------- #


class PCPFusedPagedAttnOp:
    """Fused CP prefill: 1 copy + 1 all-gather + 1 append + 1 paged attention."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
        backend: str = "auto",
        causal: bool = True,
        kv_layout: str = "NHD",
    ):
        assert causal, "CP prefill is causal-only"
        self.attn_inputs = attn_inputs
        self.attn_configs = attn_configs
        self.num_qo_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        # One page size everywhere.  The page table is built from
        # kv_cache_kernel_block_id_host, so plan(page_size=...) and the cache view
        # must use kernel_tokens_per_block to match it.  Re-blocking an
        # [blocks, 2, H, page, D] HND cache to a different page size is not a
        # valid reinterpretation (H sits between the block and page axes), so a
        # mismatch cannot be supported -- fail loudly rather than silently
        # computing wrong attention.  (allgather_cp_impl.py carries the same
        # latent inconsistency: cache view uses tokens_per_block while
        # fill_mla_params uses kernel_tokens_per_block.)
        self.seq_size_per_block = (
            attn_configs.kernel_tokens_per_block or attn_configs.tokens_per_block
        )
        assert self.seq_size_per_block == attn_configs.tokens_per_block, (
            f"CP prefill needs kernel_tokens_per_block "
            f"({attn_configs.kernel_tokens_per_block}) == tokens_per_block "
            f"({attn_configs.tokens_per_block})"
        )
        self.device = torch.cuda.current_device()
        self.workspace_buffer = get_py_flashinfer_workspace_buffer()

        self.cp_info = attn_inputs.context_parallel_info
        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size

        self.paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, kv_layout="HND", backend=backend
        )
        self.geo: Optional[_CPGeometry] = None
        self._kv_ag: Optional[torch.Tensor] = None
        self._pg = None

    # -- introspection hooks used by the factory ----------------------------- #

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        return attention_inputs.is_prefill

    # ----------------------------------------------------------------------- #

    def prepare(self, attention_inputs: PyAttentionInputs) -> ParamsBase:
        geo = _CPGeometry(
            attention_inputs,
            self.prefill_cp_size,
            self.prefill_cp_rank,
            self.seq_size_per_block,
            self.device,
        )
        self.geo = geo

        # persistent gathered-KV buffer: [cp*T_local, 2, H_kv, D].
        # Packed so K and V ride in a single collective; sliced (strided) for use.
        self._kv_ag = _kv_gather_buffer(
            geo.gathered_slots, self.num_kv_heads, self.head_dim, self.device
        )
        self._kv_local = self._kv_ag.narrow(
            0, self.prefill_cp_rank * geo.tokens_local, geo.tokens_local
        )
        self._pg = _get_group(Group.TP)
        self._out = _out_buffer(
            geo.tokens_local, self.num_qo_heads, self.head_dim, self.device
        )

        params = fill_mla_params(
            attention_inputs.prefix_lengths,
            attention_inputs.sequence_lengths,
            self.cp_info.prefill_actual_input_lengths_cpu,
            attention_inputs.kv_cache_kernel_block_id_host,
            self.seq_size_per_block,
        )

        qo, kv_indptr, kv_indices, last_len = geo.paged_plan_tensors(
            attention_inputs.kv_cache_kernel_block_id_host
        )
        self.paged_wrapper.plan(
            qo_indptr=qo,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.seq_size_per_block,
            causal=True,
            q_data_type=torch.bfloat16,
        )
        return params

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
        geo = self.geo
        hq, hkv, d = self.num_qo_heads, self.num_kv_heads, self.head_dim
        qkv = qkv.view(qkv.shape[0], -1)

        # q stays a strided view of qkv -- no copy.
        q = qkv[:, : hq * d].view(-1, hq, d)
        # 1 copy: pack this rank's K and V into its slot of the gather buffer.
        self._kv_local.copy_(qkv[:, hq * d :].view(-1, 2, hkv, d))
        # 1 collective: K and V together, into the persistent buffer.
        torch.distributed.all_gather_into_tensor(
            self._kv_ag, self._kv_local, group=self._pg
        )

        kv_cache_tensor = kv_cache.kv_cache_base.view(
            -1, 2, hkv, self.seq_size_per_block, d
        )
        k_src, v_src = self._kv_ag[:, 0], self._kv_ag[:, 1]
        batch_idx, positions = geo.ag_batch, geo.ag_pos

        # 1 append: the zigzag unshuffle is carried by ``positions``.
        append_paged_kv_cache(
            append_key=k_src,
            append_value=v_src,
            batch_indices=batch_idx,
            positions=positions,
            paged_kv_cache=kv_cache_tensor,
            kv_indices=params.page_indice_d,
            kv_indptr=params.decode_page_indptr_d,
            kv_last_page_len=params.paged_kv_last_page_len_d,
            kv_layout="HND",
        )

        # 1 attention: both zigzag halves of every sequence, straight off the
        # paged cache, prefix included.  ``out=`` keeps flashinfer from
        # allocating and zero-filling an output tensor every layer.
        if self._out is None:  # bisect mode: let flashinfer allocate the output
            return self.paged_wrapper.run(q, kv_cache_tensor)
        return self.paged_wrapper.run(q, kv_cache_tensor, out=self._out)


# --------------------------------------------------------------------------- #
# V2: fused all-gather + sequence-ordered KV + 2 ragged prefills
#     (fast path for batch == 1 and no prefix, where the ragged kernel is
#      cheaper than the paged one at short sequence lengths)
# --------------------------------------------------------------------------- #


class PCPFusedRaggedAttnOp:
    """Sequence-ordered gathered KV so both causal prefixes are free slices."""

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
        # One page size everywhere.  The page table is built from
        # kv_cache_kernel_block_id_host, so plan(page_size=...) and the cache view
        # must use kernel_tokens_per_block to match it.  Re-blocking an
        # [blocks, 2, H, page, D] HND cache to a different page size is not a
        # valid reinterpretation (H sits between the block and page axes), so a
        # mismatch cannot be supported -- fail loudly rather than silently
        # computing wrong attention.  (allgather_cp_impl.py carries the same
        # latent inconsistency: cache view uses tokens_per_block while
        # fill_mla_params uses kernel_tokens_per_block.)
        self.seq_size_per_block = (
            attn_configs.kernel_tokens_per_block or attn_configs.tokens_per_block
        )
        assert self.seq_size_per_block == attn_configs.tokens_per_block, (
            f"CP prefill needs kernel_tokens_per_block "
            f"({attn_configs.kernel_tokens_per_block}) == tokens_per_block "
            f"({attn_configs.tokens_per_block})"
        )
        self.device = torch.cuda.current_device()
        self.workspace_buffer = get_py_flashinfer_workspace_buffer()
        self.cp_info = attn_inputs.context_parallel_info
        self.prefill_cp_rank = parallelism_config.tp_rank
        self.prefill_cp_size = parallelism_config.tp_size
        self.wrappers = {
            n: BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer, kv_layout="NHD", backend=backend
            )
            for n in ("lo", "hi")
        }
        self.geo = None
        self._kv_ag = None
        self._pg = None

    @staticmethod
    def eligible(attn_inputs: PyAttentionInputs) -> bool:
        cp_info = attn_inputs.context_parallel_info
        return (
            cp_info.prefill_cp_chunk_lengths.numel() == 1
            and not bool(attn_inputs.prefix_lengths.any().item())
        )

    def support(self, attention_inputs: PyAttentionInputs) -> bool:
        return attention_inputs.is_prefill and self.eligible(attention_inputs)

    def prepare(self, attention_inputs: PyAttentionInputs) -> ParamsBase:
        geo = _CPGeometry(
            attention_inputs,
            self.prefill_cp_size,
            self.prefill_cp_rank,
            self.seq_size_per_block,
            self.device,
        )
        assert geo.batch_size == 1 and not geo.has_prefix, (
            "PCPFusedRaggedAttnOp is the batch-1, no-prefix fast path"
        )
        self.geo = geo
        self._kv_ag = _kv_gather_buffer(
            geo.gathered_slots, self.num_kv_heads, self.head_dim, self.device
        )
        self._kv_local = self._kv_ag.narrow(
            0, self.prefill_cp_rank * geo.tokens_local, geo.tokens_local
        )
        self._kv_seq = torch.empty(
            geo.tokens_total, 2, self.num_kv_heads, self.head_dim,
            dtype=torch.bfloat16, device=self.device,
        )
        self._out = torch.empty(
            geo.tokens_local, self.num_qo_heads, self.head_dim,
            dtype=torch.bfloat16, device=self.device,
        )
        self._pg = _get_group(Group.TP)

        h = geo.half_lengths[0]
        lo_len, hi_len = geo.half_kv_lengths(0)
        self._h = h
        self._lo_len, self._hi_len = lo_len, hi_len
        common = dict(
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            causal=True,
            q_data_type=torch.bfloat16,
        )
        for name, kv_len in (("lo", lo_len), ("hi", hi_len)):
            self.wrappers[name].plan(
                qo_indptr=torch.tensor([0, h], dtype=torch.int32, device=self.device),
                kv_indptr=torch.tensor(
                    [0, kv_len], dtype=torch.int32, device=self.device
                ),
                **common,
            )

        return fill_mla_params(
            attention_inputs.prefix_lengths,
            attention_inputs.sequence_lengths,
            self.cp_info.prefill_actual_input_lengths_cpu,
            attention_inputs.kv_cache_kernel_block_id_host,
            self.seq_size_per_block,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        params: ParamsBase = None,
    ) -> torch.Tensor:
        geo = self.geo
        hq, hkv, d = self.num_qo_heads, self.num_kv_heads, self.head_dim
        h = self._h
        qkv = qkv.view(qkv.shape[0], -1)
        q = qkv[:, : hq * d].view(-1, hq, d)

        self._kv_local.copy_(qkv[:, hq * d :].view(-1, 2, hkv, d))
        torch.distributed.all_gather_into_tensor(
            self._kv_ag, self._kv_local, group=self._pg
        )
        # one gather into sequence order: both causal prefixes then become slices
        torch.index_select(self._kv_ag, 0, geo.restore, out=self._kv_seq)
        k_seq, v_seq = self._kv_seq[:, 0], self._kv_seq[:, 1]

        append_paged_kv_cache(
            append_key=k_seq,
            append_value=v_seq,
            batch_indices=geo.seq_batch,
            positions=geo.seq_pos,
            paged_kv_cache=kv_cache.kv_cache_base.view(
                -1, 2, hkv, self.seq_size_per_block, d
            ),
            kv_indices=params.page_indice_d,
            kv_indptr=params.decode_page_indptr_d,
            kv_last_page_len=params.paged_kv_last_page_len_d,
            kv_layout="HND",
        )

        self.wrappers["lo"].run(
            q[:h], k_seq[: self._lo_len], v_seq[: self._lo_len], out=self._out[:h]
        )
        self.wrappers["hi"].run(
            q[h:], k_seq[: self._hi_len], v_seq[: self._hi_len], out=self._out[h:]
        )
        return self._out


# --------------------------------------------------------------------------- #
# Which of the two to use
# --------------------------------------------------------------------------- #
#
# PCPFusedPagedAttnOp is the default and handles every shape.  The ragged variant
# is kept because it is measurably faster in one narrow window, but it is NOT
# dispatched to automatically.  Measured on 8x H20-3e (64/8/128, bf16, page 64),
# ms per layer, min over 3 independent runs with inter-rank skew <= 1.05:
#
#   cp=8  N=1024 (chunk 128):  paged 0.111  ragged 0.119   -> paged  +7%
#   cp=8  N=2048 (chunk 256):  paged 0.180  ragged 0.160   -> ragged +11%
#   cp=8  N=4096 (chunk 512):  paged 0.414  ragged 0.428   -> paged  +3%
#   cp=4  N=1024 (chunk 256):  paged 0.118  ragged 0.118   -> tie
#   cp=4  N=2048 (chunk 512):  paged 0.244  ragged 0.256   -> paged  +5%
#   cp=4  N=8192 (chunk 2048): paged 2.309  ragged 2.332   -> paged  +1%
#
# The ragged win is non-monotonic in chunk length -- it appears at chunk 256 and
# vanishes at 128 and 512 -- which points at attention-kernel tile quantisation
# rather than anything structural, and it would move with head count or page
# size.  Dispatching on a one-point window would be overfitting, and the ragged
# operator additionally only supports batch 1 with no prefix cache, so it would
# rarely fire in production anyway.  One unrestricted path is worth <=11% in a
# single shape; select it explicitly with RTP_CP_IMPL=ragged if you disagree.
