# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
from contextlib import suppress
from dataclasses import dataclass

import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard
from xformers.ops.fmha import merge_attentions
from xformers.ops.fmha.attn_bias import (
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)
from xformers.ops.fmha.flash3 import mha_fwd
from xformers.ops.rmsnorm import rms_norm

from cwm.model.transformer import PosEmbedImpl, Transformer

from .kernels import apply_rope, paged_memcpy
from .utils.cache import ModelCache, RawCache
from .utils.collectives import Collectives

with suppress(ModuleNotFoundError, ImportError):
    # in case we are running tests, do not crash
    # on import errors for CUDA libraries
    from flash_mla.flash_mla_interface import (
        flash_mla_with_kvcache,
        get_mla_metadata,
    )


def maybe_dist_to_local(x: torch.Tensor | DTensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        return x.to_local()
    return x


class ModelState:
    """
    Encapsulates the Transformer's input and state (e.g., caches)
    tensors to simplify using the model in cuda graphs.
    """

    _attn_bias: AttnBias
    _actual_batch_size: torch.Tensor
    tokens: torch.Tensor
    cache: ModelCache
    cache_shard: tuple[int, int]
    batch_size: int

    def __init__(
        self,
        batch_size: int,
        tokens: torch.Tensor,
        block_tbl: torch.Tensor,
        block_len: int,
        cache: RawCache,
        cache_shard: tuple[int, int],
        device: torch.device,
    ) -> None:
        """
        Initialize the model such that ``decode()`` can be
        called at once.
        """
        assert tokens.device == device
        assert cache.caches[0].device == device

        seqlen = block_tbl.shape[1] * block_len
        attn_bias = AttnBias.from_seqlens(
            q_seqlen=[1] * batch_size,
            kv_seqlen=[seqlen] * batch_size,
            block_tables=block_tbl[:batch_size],
            page_size=block_len,
        )
        attn_bias.q_seqinfo.to(device)
        attn_bias.k_seqinfo.to(device)
        self._attn_bias = attn_bias
        self._actual_batch_size = torch.tensor(
            batch_size,
            dtype=torch.int64,
            device=device,
        )

        self.tokens = tokens[:batch_size]
        self.batch_size = batch_size
        self.cache = cache
        self.cache_shard = cache_shard

    @property
    def seqlen(self) -> torch.Tensor:
        """
        The sequence lengths in the input batch. Taking into
        account the tokens to be processed by the next model
        call.
        """
        return self._attn_bias.k_seqinfo.seqlen

    @property
    def blktbl(self) -> torch.Tensor:
        """
        The block table mapping. That is, an integer tensor of
        shape [B, N] where B is the batch size and N is the
        number of blocks in the cache.

        Note that a block table mapping can address *more*
        tokens than are physically available in the caches;
        that is sensible because the same physical block can
        be reused in different batch lanes.

        The block table mapping can be seen as a tensor that
        tells for each logical lane in the batch what is its
        sequence of physical blocks.
        """
        return self._attn_bias.block_tables

    def set_actual_batch_size(self, n: int) -> None:
        self._actual_batch_size.fill_(n)

    def copy_inputs(
        self,
        blktbl: torch.Tensor,
        tokens: torch.Tensor,
        seqlen: torch.Tensor,
    ) -> None:
        if blktbl.data_ptr() != self.blktbl.data_ptr():
            self.blktbl.copy_(blktbl[: self.batch_size])
        if tokens.data_ptr() != self.tokens.data_ptr():
            self.tokens.copy_(tokens[: self.batch_size])
        if seqlen.data_ptr() != self.seqlen.data_ptr():
            self.seqlen.copy_(seqlen[: self.batch_size])


def vocab_parallel_embedding(
    embeddings: torch.Tensor,
    tokens: torch.Tensor,
    coll: Collectives,
) -> torch.Tensor:
    """
    Map the input token IDs to their embedding. The model
    arg must have its embedding weight evenly on the vocab
    dimension across the model parallel ranks.

    Args:
        embeddings (torch.Tensor):
            local shard of the embeddings weight, with
            vocabulary as first dimension
        tokens (torch.Tensor):
            1-D tensor of the token ids to embed
        coll (Collectives):
            collectives implementation

    Returns:
        torch.Tensor:
            the embeddings of the input tokens, of shape
            (S, D) with D the model dimension and dtype
            bfloat16.
    """
    local_vocab_size = embeddings.shape[0]
    start_index = coll.rank * local_vocab_size
    end_index = start_index + local_vocab_size
    mask = (tokens < start_index) | (tokens >= end_index)
    input = tokens - start_index
    input[mask] = 0
    h = F.embedding(input, embeddings)
    h[mask, :] = 0.0
    return coll.all_reduce(h)


def quantize_fp8_rowwise(
    x: torch.Tensor,
    scale_clip: float | None,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8 = torch.finfo(dtype)
    row_max = torch.max(torch.abs(x), dim=1)[0]
    if scale_clip is not None:
        row_max = torch.clamp(row_max, min=fp8.eps, max=scale_clip)
    else:
        row_max = torch.clamp(row_max, min=fp8.eps)
    x_scale = (fp8.max / row_max.to(torch.float32))[:, None]
    x_scale[x_scale == float("inf")] = 1.0
    x_fp8 = x * x_scale
    x_fp8 = torch.clamp(x_fp8, min=fp8.min, max=fp8.max)
    return x_fp8.to(dtype), 1 / x_scale


@torch.compile
def ffn_fp8_rowwise(
    h: torch.Tensor,
    ffn: torch.nn.Module,
    scale_clip: float | None,
) -> torch.Tensor:
    h_fp8, h_scale = quantize_fp8_rowwise(h, scale_clip)
    x1: torch.Tensor = torch._scaled_mm(  # type: ignore
        h_fp8,
        ffn.w1.weight.T,
        scale_a=h_scale,
        scale_b=ffn.w1.fp8_scale.unsqueeze(0),
        out_dtype=h.dtype,
    )
    x3: torch.Tensor = torch._scaled_mm(  # type: ignore
        h_fp8,
        ffn.w3.weight.T,
        scale_a=h_scale,
        scale_b=ffn.w3.fp8_scale.unsqueeze(0),
        out_dtype=h.dtype,
    )
    x = F.silu(x1, inplace=True).mul_(x3)
    x_fp8, x_scale = quantize_fp8_rowwise(x, scale_clip)
    out: torch.Tensor = torch._scaled_mm(  # type: ignore
        x_fp8,
        ffn.w2.weight.T,
        scale_a=x_scale,
        scale_b=ffn.w2.fp8_scale.unsqueeze(0),
        out_dtype=h.dtype,
    )
    return out


@dataclass
class MlaPrefillChunk:
    query_index: torch.Tensor
    """
    Index tensor to use for queries.
    """
    cache_index: torch.Tensor
    """
    Index tensor to query the paged
    latent cache.
    """
    q_seqstart: torch.Tensor
    "Sequence split for Q."
    k_seqstart: torch.Tensor
    "Sequence split for K and V."
    max_k_seqlen: int
    """
    Maximum sequence length of the split
    for K and V.
    """
    shard_index: torch.Tensor | None
    """
    Index tensor for the local shard of the
    full latent (xkv) tensor.
    """
    length: int
    "Length of the context chunk."


@dataclass
class MlaPrefill:
    """
    Prefill plan for MLA. The latent cache is split in chunks
    to avoid running out of memory. Each chunk yields a partial
    attention call. Partial attention results are combined with
    xformers' merge_attentions.
    """

    chunk: list[MlaPrefillChunk]
    causal_seqstart: torch.Tensor
    max_q_seqlen: int

    @staticmethod
    def prepare(
        k_seqlen: list[int],
        q_seqlen: list[int],
        q_cumsum: list[int],
        q_cumsum_cuda: torch.Tensor,
        block_tbl: torch.Tensor,
        block_len: int,
        cache_shard: tuple[int, int],
        chunk_size: int = 4096,
    ) -> "MlaPrefill":
        k_cumsum = [0]
        for n in k_seqlen:
            k_cumsum.append(n + k_cumsum[-1])
        k_size = k_cumsum[-1]

        _, nshards = cache_shard
        page_len = block_len * nshards
        page_idx = torch.arange(
            page_len,
            dtype=torch.int64,
            device=block_tbl.device,
        )

        # prepare context chunks
        chunk_list = []
        chunk_beg = 0
        chunk_end = min(chunk_size, k_size)
        lane_beg, lane_end = 0, 0

        while chunk_beg < chunk_end:
            while k_cumsum[lane_beg + 1] <= chunk_beg:
                lane_beg += 1
            lane_end = lane_beg
            while k_cumsum[lane_end] < chunk_end:
                lane_end += 1

            # query information
            query_index = torch.arange(
                q_cumsum[lane_beg],
                q_cumsum[lane_end],
                dtype=torch.int,
                device=block_tbl.device,
            )
            q_seqstart = q_cumsum_cuda[lane_beg : lane_end + 1]
            q_seqstart = q_seqstart - q_cumsum[lane_beg]

            # cache information
            shard_index = []
            cache_index = []
            seqstart = [0]
            max_seqlen = 0
            for n in range(lane_beg, lane_end):
                beg = max(chunk_beg, k_cumsum[n]) - k_cumsum[n]
                end = min(chunk_end, k_cumsum[n + 1]) - k_cumsum[n]
                assert beg < end
                shard: slice | torch.Tensor
                if nshards == 1:
                    shard = slice(beg, end)
                else:
                    i, c = cache_shard
                    shard = torch.tensor(
                        [j for j in range(beg, end) if j % c == i],
                        dtype=torch.int,
                        device=block_tbl.device,
                    )
                    shard_index.append(shard - beg + seqstart[-1])
                seqstart.append(seqstart[-1] + end - beg)
                max_seqlen = max(end - beg, max_seqlen)
                npage = (end + page_len - 1) // page_len
                block = block_tbl[n, :npage]
                idx = block.unsqueeze(-1) * page_len + page_idx
                cache_index.append(idx.flatten()[shard] // nshards)
            k_seqstart = torch.tensor(
                seqstart,
                dtype=torch.int,
                device=block_tbl.device,
            )

            chunk_list.append(
                MlaPrefillChunk(
                    query_index=query_index,
                    cache_index=torch.cat(cache_index),
                    q_seqstart=q_seqstart,
                    k_seqstart=k_seqstart,
                    max_k_seqlen=max_seqlen,
                    shard_index=torch.cat(shard_index) if nshards > 1 else None,
                    length=seqstart[-1],
                )
            )

            chunk_beg = chunk_end
            chunk_end += chunk_size
            chunk_end = min(chunk_end, k_size)

        return MlaPrefill(
            chunk=chunk_list,
            causal_seqstart=q_cumsum_cuda,
            max_q_seqlen=max(q_seqlen),
        )

    def run(
        self,
        xq: torch.Tensor,  # [B, NH, NOPE+ROPE]
        xkv: torch.Tensor,  # [B, LORA]
        xk_pe: torch.Tensor,  # [B, 1, ROPE]
        cache: torch.Tensor,  # [?, LORA+ROPE]
        wkvb: torch.Tensor,  # [NH * (NOPE+HEAD), LORA]
        head_dim: int,
        nope_dim: int,
        rope_dim: int,
        lora_dim: int,
        n_kv_heads: int,
        coll: Collectives | None,
    ) -> torch.Tensor:  # [B, NH, HEAD]
        xkv = F.linear(xkv, wkvb).unflatten(-1, (n_kv_heads, -1))
        xk_nope, xv = torch.split(xkv, [nope_dim, head_dim], dim=-1)
        xk_pe = xk_pe.expand(-1, n_kv_heads, -1)
        xk = torch.cat([xk_nope, xk_pe], dim=-1)

        softmax_scale = xq.shape[-1] ** -0.5
        attn, lse = mha_fwd(
            query=xq,
            key=xk,
            value=xv,
            cu_seqlens_q=self.causal_seqstart,
            cu_seqlens_k=self.causal_seqstart,
            seqused_k=None,
            leftpad_k=None,
            max_seqlen_q=self.max_q_seqlen,
            max_seqlen_k=self.max_q_seqlen,
            p=0,
            softmax_scale=softmax_scale,
            is_causal=True,
        )
        # attn: [B, NH, HEAD]
        # lse: [NH, B]

        nchunk = len(self.chunk)
        if nchunk == 0:
            # no need to process the context
            return attn

        chunk_attn = torch.zeros(
            (nchunk + 1, *attn.shape),
            dtype=attn.dtype,
            device=attn.device,
        )
        chunk_lse = torch.full(
            (nchunk + 1, *lse.shape),
            float("-inf"),
            dtype=lse.dtype,
            device=lse.device,
        )
        chunk_attn[nchunk] = attn
        chunk_lse[nchunk] = lse

        for n, c in enumerate(self.chunk):
            if c.shard_index is not None:
                assert coll is not None
                xkv = torch.zeros(
                    (c.length, cache.shape[-1]),
                    dtype=xq.dtype,
                    device=xq.device,
                )
                xkv[c.shard_index] = cache[c.cache_index]
                xkv = coll.all_reduce(xkv)
            else:
                xkv = cache[c.cache_index]
            xkv, xk_pe = torch.split(xkv, [lora_dim, rope_dim], dim=-1)
            xkv = F.linear(xkv, wkvb).unflatten(-1, (n_kv_heads, -1))
            xk_nope, xv = torch.split(xkv, [nope_dim, head_dim], dim=-1)
            xk_pe = xk_pe.unsqueeze(1).expand(-1, n_kv_heads, -1)
            xk = torch.cat([xk_nope, xk_pe], dim=-1)

            attn, lse = mha_fwd(
                query=xq[c.query_index],
                key=xk,
                value=xv,
                cu_seqlens_q=c.q_seqstart,
                cu_seqlens_k=c.k_seqstart,
                seqused_k=None,
                leftpad_k=None,
                max_seqlen_q=self.max_q_seqlen,
                max_seqlen_k=c.max_k_seqlen,
                p=0,
                softmax_scale=softmax_scale,
                is_causal=False,
            )
            # attn: [CHUNK_B, NH, HEAD]
            # lse: [NH, CHUNK_B]

            chunk_attn[n][c.query_index] = attn
            chunk_lse[n].T[c.query_index] = lse.T

        attn, _ = merge_attentions(
            attn_split=chunk_attn.unsqueeze(1),
            lse_split=chunk_lse.unsqueeze(1),
            write_lse=False,
        )
        return attn.squeeze(0)


@torch.inference_mode()
def _forward(
    model: Transformer,
    coll: Collectives | None,
    q_seqlen: list[int] | None,
    actual_batch_size: torch.Tensor | None,
    token_values: torch.Tensor,
    attn_bias: AttnBias,
    mla_attn: MlaPrefill | None,
    cache: ModelCache,
    cache_shard: tuple[int, int],
    logits_idx: torch.Tensor | None,
    prefill: bool = True,
) -> torch.Tensor:
    """
    Forward using the weights in ``model`` and efficient xformers
    kernels.
    """

    qk_head_dim = model.qk_head_dim
    head_dim = model.head_dim
    nope_dim = model.qk_nope_head_dim
    rope_dim = model.qk_rope_head_dim
    n_layers = model.n_layers
    n_local_heads = model.layers[0].attention.n_heads
    n_local_kv_heads = model.layers[0].attention.n_kv_heads
    assert model.norm is not None, "unsupported model"
    eps = model.norm.eps  # identical for all RMSNorm layers

    cache.page_in(0)

    # q_batch maps each token to its batch number
    # q_seqpos maps each token to its position
    # cache_len contains local cache sizes
    q_batch: torch.Tensor | None
    q_seqpos: torch.Tensor
    cache_len = attn_bias.k_seqinfo.seqlen

    if prefill:
        assert q_seqlen is not None
        q_batch = torch.tensor(
            sum(([i] * n for i, n in enumerate(q_seqlen)), []),
            dtype=torch.int,
            device=token_values.device,
        )
        k_seqlen = attn_bias.k_seqinfo.seqlen_py
        q_seqpos_list: list[int] = []
        for n, t in zip(k_seqlen, q_seqlen, strict=False):
            q_seqpos_list.extend(range(n - t, n))
        q_seqpos = torch.tensor(
            q_seqpos_list,
            dtype=torch.int,
            device=token_values.device,
        )
    else:
        q_batch = None
        # pull a tensor from the attention bias so that
        # the decoding remains graphable (i.e., does not
        # depend on host memory)
        q_seqpos = attn_bias.k_seqinfo.seqlen - 1
        if cache_shard[1] > 1:
            idx, cnt = cache_shard
            cache_len = (cache_len + cnt - 1 - idx) // cnt

    tok_embedding_weight = maybe_dist_to_local(model.tok_embeddings.weight)
    placement = (
        model.tok_embeddings.weight.placements[0]
        if isinstance(model.tok_embeddings.weight, DTensor)
        else None
    )
    if isinstance(placement, Shard) and placement.dim == 0:
        # vocab-parallel
        assert coll is not None
        h = vocab_parallel_embedding(tok_embedding_weight, token_values, coll)
    else:
        # else we have a parallel embedding on the
        # model dimension
        h_parallel = F.embedding(
            token_values,
            tok_embedding_weight,
        )
        if coll is not None:
            h = coll.all_gather(h_parallel)
        else:
            h = h_parallel

    if model.kv_lora_rank > 0 and not prefill:
        n_heads = n_local_heads
        if coll is not None:
            # MLA decoding does not shard heads
            n_heads *= coll.tp_size
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_len,
            num_heads_per_head_k=n_heads,
            num_heads_k=1,  # MQA during decoding
        )

    for i, layer in enumerate(model.layers):
        if i + 1 < n_layers:
            # request a cache page-in for the *next* layer so that
            # potential memory transfers may happen concurrently
            # with the current layer
            cache.page_in(i + 1)

        attention_norm_weight = maybe_dist_to_local(layer.attention_norm.weight)
        h_in_attn = rms_norm(h, attention_norm_weight, eps)

        if model.qkv_biases:
            wq_bias = maybe_dist_to_local(layer.attention.wq.bias)
            wk_bias = maybe_dist_to_local(layer.attention.wk.bias)
            wv_bias = maybe_dist_to_local(layer.attention.wv.bias)
        else:
            wq_bias, wk_bias, wv_bias = None, None, None

        attn = layer.attention

        if attn.q_lora_rank == 0:
            wq_weight = maybe_dist_to_local(attn.wq.weight)
        else:
            wq_weight = maybe_dist_to_local(attn.wq_b.weight)

        if attn.kv_lora_rank > 0 and not prefill:
            # during decoding, we want all heads for q
            # and parallelize on the kv-cache dimension
            wq_weight = attn._wq_full  # [NH * (NOPE + ROPE), DIM]

        if attn.q_lora_rank == 0:
            xq = F.linear(h_in_attn, wq_weight, wq_bias)
        else:
            wqa_weight = maybe_dist_to_local(attn.wq_a.weight)
            qnorm_weight = maybe_dist_to_local(attn.q_norm.weight)
            xq_lora = F.linear(h_in_attn, wqa_weight)
            xq_lora = rms_norm(xq_lora, qnorm_weight, eps)
            xq = F.linear(xq_lora, wq_weight)

        if attn.kv_lora_rank == 0:
            # GQA models
            wk_weight = maybe_dist_to_local(attn.wk.weight)
            wv_weight = maybe_dist_to_local(attn.wv.weight)
            xk = F.linear(h_in_attn, wk_weight, wk_bias)
            xv = F.linear(h_in_attn, wv_weight, wv_bias)

            xq = xq.view(xq.shape[0], n_local_heads, qk_head_dim)
            xk = xk.view(xk.shape[0], n_local_kv_heads, qk_head_dim)
            xv = xv.view(xv.shape[0], n_local_kv_heads, head_dim)

            if model.qk_norm:
                k_norm_weight = maybe_dist_to_local(layer.attention.k_norm.weight)
                q_norm_weight = maybe_dist_to_local(layer.attention.q_norm.weight)
                xk = rms_norm(xk, k_norm_weight, eps)
                xq = rms_norm(xq, q_norm_weight, eps)

            apply_rope(xq, q_seqpos, model.rope_freqs)
            apply_rope(xk, q_seqpos, model.rope_freqs)

            cache_k, cache_v = cache.cache_kv(i)
            for x, c in (xk, cache_k), (xv, cache_v):
                paged_memcpy(
                    src=x.flatten(1, 2),
                    dst=c,
                    page_tbl=attn_bias.block_tables,
                    dst_pos=q_seqpos,
                    dst_shard=None,
                    src_batch=q_batch,
                    batch_size=actual_batch_size,
                    page_size=attn_bias.page_size,
                )

            # request a cache page out for the current layer
            # now that it has been updated
            cache.page_out(i)

            kv_shape = (-1, attn_bias.page_size, n_local_kv_heads)
            attn_out, _ = mha_fwd(
                query=xq,
                key=cache_k.reshape(*kv_shape, qk_head_dim),
                value=cache_v.reshape(*kv_shape, head_dim),
                cu_seqlens_q=attn_bias.q_seqinfo.seqstart,
                cu_seqlens_k=None,
                seqused_k=attn_bias.k_seqinfo.seqlen,
                leftpad_k=None,
                max_seqlen_q=attn_bias.q_seqinfo.max_seqlen,
                max_seqlen_k=attn_bias.k_seqinfo.max_seqlen,
                p=0,
                block_table=attn_bias.block_tables,
                softmax_scale=xq.shape[-1] ** -0.5,
                is_causal=True,
                window_left=layer.window_size_left,
                window_right=-1,
            )
            attn_wo_weight = maybe_dist_to_local(attn.wo.weight)

        else:
            # MLA models
            assert not model.qk_norm, "mla + qk_norm unsupported"
            assert not model.qkv_biases, "mla + qkv_biases unsupported"

            lora_dim = attn.kv_lora_rank
            wkva_weight = maybe_dist_to_local(attn.wkv_a.weight)
            kvnorm_weight = maybe_dist_to_local(attn.kv_norm.weight)
            wkvb_weight = maybe_dist_to_local(attn.wkv_b.weight)
            n_heads = n_local_heads
            if not prefill and coll is not None:
                n_heads *= coll.tp_size

            # TODO(qcar): strides in apply_rope
            # TODO(qcar): strides in rms_norm

            xkv = F.linear(h_in_attn, wkva_weight)
            xkv, xk_pe = torch.split(xkv, [lora_dim, rope_dim], dim=-1)
            xk_pe = xk_pe.unsqueeze(1).contiguous()
            apply_rope(xk_pe, q_seqpos, model.rope_freqs)

            xkv = xkv.contiguous()
            xkv = rms_norm(xkv, kvnorm_weight, eps)

            (cache_xkv,) = cache.cache_kv(i)
            paged_memcpy(
                src=torch.cat([xkv, xk_pe.squeeze(1)], dim=-1),
                dst=cache_xkv,
                page_tbl=attn_bias.block_tables,
                dst_pos=q_seqpos,
                dst_shard=cache_shard,
                src_batch=q_batch,
                batch_size=actual_batch_size,
                page_size=attn_bias.page_size,
            )

            # request a cache page out for the current layer
            # now that it has been updated
            cache.page_out(i)

            xq = xq.unflatten(-1, (n_heads, -1))
            # xq: [B, NH, NOPE + ROPE]
            xq_nope, xq_pe = torch.split(xq, [nope_dim, rope_dim], dim=-1)
            xq_pe = xq_pe.contiguous()
            apply_rope(xq_pe, q_seqpos, model.rope_freqs)

            if prefill:
                assert mla_attn is not None
                attn_out = mla_attn.run(
                    xq=torch.cat([xq_nope, xq_pe], dim=-1),
                    xkv=xkv,
                    xk_pe=xk_pe,
                    cache=cache_xkv,
                    wkvb=wkvb_weight,
                    head_dim=head_dim,
                    nope_dim=nope_dim,
                    rope_dim=rope_dim,
                    lora_dim=lora_dim,
                    n_kv_heads=n_heads,
                    coll=coll,
                )
                attn_wo_weight = maybe_dist_to_local(attn.wo.weight)

            else:
                # _wkb: [NH, NOPE, LORA]
                # _wv: [NH, LORA, HEAD]
                xq_nope = xq_nope.transpose(0, 1)
                # [NH, B, NOPE] x [NH, NOPE, LORA] = [NH, B, LORA]
                xq_nope = torch.bmm(xq_nope, attn._wkb)
                xq_nope = xq_nope.transpose(0, 1)
                # xq_nope: [B, NH, LORA]
                xq = torch.cat([xq_nope, xq_pe], dim=-1)

                cache_xkv = cache_xkv.reshape(
                    -1,
                    attn_bias.page_size,
                    1,  # MQA during decoding
                    lora_dim + rope_dim,
                )
                attn_out, attn_lse = flash_mla_with_kvcache(
                    q=xq.unsqueeze(1),  # q_seqlen is 1, decoding
                    k_cache=cache_xkv,
                    block_table=attn_bias.block_tables,
                    cache_seqlens=cache_len,
                    head_dim_v=lora_dim,
                    tile_scheduler_metadata=tile_scheduler_metadata,
                    num_splits=num_splits,
                    softmax_scale=(nope_dim + rope_dim) ** -0.5,
                )
                # attn_out: [B, 1, NH, LORA]
                # attn_lse: [B, NH, 1]

                if coll is not None:
                    attn_out, _ = merge_attentions(
                        coll.all_gather(attn_out[None], 0),
                        coll.all_gather(attn_lse[None], 0),
                        write_lse=False,
                    )

                attn_out = attn_out.squeeze(1).transpose(0, 1)
                attn_out = torch.bmm(attn_out, attn._wv)  # [NH, B, HEAD]
                attn_out = attn_out.transpose(0, 1)
                attn_wo_weight = attn._wo_full

        # attn_out: [B, NH, HEAD]
        h_out = F.linear(attn_out.flatten(1, 2), attn_wo_weight)
        if (attn.kv_lora_rank == 0 or prefill) and coll is not None:
            # no need to reduce when decoding with MLA
            # since heads are not spread across the TP
            # group
            h_out = coll.all_reduce(h_out)
        h.add_(h_out)

        ffn_norm_weight = maybe_dist_to_local(layer.ffn_norm.weight)
        h_in_ffn = rms_norm(h, ffn_norm_weight, eps)

        feed_forward_w1_weight = maybe_dist_to_local(layer.feed_forward.w1.weight)
        feed_forward_w3_weight = maybe_dist_to_local(layer.feed_forward.w3.weight)
        x1 = F.linear(h_in_ffn, feed_forward_w1_weight)
        x3 = F.linear(h_in_ffn, feed_forward_w3_weight)

        feed_forward_w2_weight = maybe_dist_to_local(layer.feed_forward.w2.weight)
        # in-place operations to save 33% of activations
        x_mul = F.silu(x1, inplace=True).mul_(x3)
        h_out = F.linear(x_mul, feed_forward_w2_weight)

        if coll is not None:
            h_out = coll.all_reduce(h_out)
        h.add_(h_out)

    norm_weight = maybe_dist_to_local(model.norm.weight)
    h = rms_norm(h, norm_weight, eps)
    if logits_idx is not None:
        assert prefill
        h = h[logits_idx]
    if hasattr(model.output, "weight") and model.output.weight is not None:
        original_output_weight = model.output.weight
    elif hasattr(model.output, "tied_module") and model.output.tied_module is not None:
        # support tied embedding
        original_output_weight = model.output.tied_module.weight
    else:
        raise AttributeError(
            "model.output has neither 'weight' nor 'tied_module' attribute"
        )
    output_weight = maybe_dist_to_local(original_output_weight)
    if isinstance(original_output_weight, DTensor):
        assert coll is not None
        placement = original_output_weight.placements[0]
        assert isinstance(placement, Shard), f"{placement}"
        if placement.dim == 0:
            # column-parallel
            logits_parallel = F.linear(h, output_weight)
            logits = coll.all_gather(logits_parallel)
        elif placement.dim == 1:
            # row-parallel
            h = coll.local_split(h)
            logits_parallel = F.linear(h, output_weight)
            logits = coll.all_reduce(logits_parallel)
        else:
            raise ValueError(f"can this happen? {placement.dim}")
    else:
        # no output parallelism
        logits = F.linear(h, output_weight)
    return logits.float()


def prefill(
    model: Transformer,
    coll: Collectives | None,
    token_values: torch.Tensor,
    seq_info: list[tuple[int, int]],
    block_tbl: torch.Tensor,
    block_len: int,
    cache: ModelCache,
    cache_shard: tuple[int, int],
    logprobs: bool = False,
) -> tuple[torch.Tensor, list[list[float]] | None]:
    """
    Call the model for prompt processing.

    Args:
        model (Model):
            the model object from which weights are pulled.
        token_values (torch.Tensor):
            the concatenated sequence of tokens for the prompts
            to process; the sequence prompts do not have to be
            of same length (e.g., token_values could be of the
            form ``|prompt1|longerprompt2|ompt3|``).
        seq_info (list[tuple[int, int]]):
            sequence information for the prompts to process;
            for each prompt seq_info has a pair of integers
            ``(n0, n1)`` where ``n0`` is the size of the prompt
            prefix for which we already have a kv-cache and
            ``n1`` is the number of tokens to actually process
            (e.g., for the example token_values above we could
            have ``[(0, 7), (0, 13), (2, 5)]``, assuming that
            the last batch element already has a kv-cache for
            "pr").
        block_tbl (torch.Tensor): see ``ModelState.block_tbl``.
        block_len (int): the size of a cache block in tokens.
        cache (ModelCache): the kv-cache to read and write.
        cache_shard (tuple[int, int]):
            a tuple ``(idx, cnt)`` that specifies that which
            shard of the inference cache is held in ``cache``.
            The ``idx`` integer is the shard index and ``cnt``
            is the total number of shards. Cache sharding can
            be disabled by passing ``(0, 1)``.
        logprobs (bool, optional):
            whether to return logprobs for the prompt tokens.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            the logits for the last token of each prompt and,
            if ``logprobs`` is set, the logprobs of all non-
            predicted tokens.
    """
    q_seqlen = [n1 for _, n1 in seq_info]
    attn_bias = AttnBias.from_seqlens(
        q_seqlen=q_seqlen,
        kv_seqlen=[n0 + n1 for n0, n1 in seq_info],
        block_tables=block_tbl,
        page_size=block_len,
    )
    mla_attn: MlaPrefill | None = None
    if model.kv_lora_rank > 0:
        mla_attn = MlaPrefill.prepare(
            k_seqlen=[n0 for n0, _ in seq_info],
            q_seqlen=q_seqlen,
            q_cumsum=attn_bias.q_seqinfo.seqstart_py,
            q_cumsum_cuda=attn_bias.q_seqinfo.seqstart,
            block_tbl=block_tbl,
            block_len=block_len,
            cache_shard=cache_shard,
        )
    else:
        assert cache_shard == (0, 1)
    logits_idx = attn_bias.q_seqinfo.seqstart[1:] - 1
    logits = _forward(
        model,
        coll,
        q_seqlen,
        None,
        token_values,
        attn_bias,
        mla_attn,
        cache,
        cache_shard,
        None if logprobs else logits_idx,
        prefill=True,
    )
    if logprobs:
        seq_logprobs: list[list[float]] = []
        token_logprobs = (
            -F.cross_entropy(
                logits[:-1],
                token_values[1:].to(torch.int64),
                reduction="none",
            )
        ).tolist()
        beg = 0
        for sl in q_seqlen:
            end = beg + sl - 1
            lp = token_logprobs[beg:end]
            seq_logprobs.append(lp)
            beg += sl
        logits = logits[logits_idx]
        return logits, seq_logprobs
    else:
        return logits, None


def decode(
    model: Transformer,
    coll: Collectives | None,
    state: ModelState,
) -> torch.Tensor:
    """
    Call the model to decode one token.

    Args:
        model (Model):
            the model object from which weights are pulled.
        state (ModelState):
            the model inputs; use ``ModelState.copy_inputs``
            and ``ModelState.set_actual_batch_size`` to set
            the model inputs.

    Returns:
        torch.Tensor:
            the logits for each decoded token.
    """
    return _forward(
        model,
        coll,
        None,
        state._actual_batch_size,
        state.tokens,
        state._attn_bias,
        None,
        state.cache,
        state.cache_shard,
        logits_idx=None,
        prefill=False,
    )


def update_model(
    model: Transformer,
    coll: Collectives | None,
) -> None:
    assert not torch.cuda.is_current_stream_capturing()
    n_local_heads = model.layers[0].attention.n_heads
    nope_dim = model.qk_nope_head_dim

    def update_weight(
        obj: torch.nn.Module,
        attr: str,
        value: torch.Tensor,
    ) -> None:
        # copy in-place as the weights are
        # burnt in cuda graphs
        t = getattr(obj, attr, None)
        if t is not None:
            t.copy_(value)
        else:
            setattr(obj, attr, value)

    for layer in model.layers:
        attn = layer.attention

        # pre-compute/gather MLA weights
        if attn.kv_lora_rank > 0:
            wkvb = maybe_dist_to_local(attn.wkv_b.weight)
            if attn.q_lora_rank == 0:
                assert attn.wq.bias is None
                wq_full = maybe_dist_to_local(attn.wq.weight)
            else:
                wq_full = maybe_dist_to_local(attn.wq_b.weight)
            wo_full = maybe_dist_to_local(attn.wo.weight)
            n_heads = n_local_heads

            if coll is not None:
                wkvb = coll.all_gather(wkvb, 0)
                wq_full = coll.all_gather(wq_full, 0)
                wo_full = coll.all_gather(wo_full, 1)
                n_heads *= coll.tp_size

            wkvb = wkvb.unflatten(0, (n_heads, -1))
            wkb = wkvb[:, :nope_dim]
            wv = wkvb[:, nope_dim:].transpose(1, 2)
            # _wkb: [NH, NOPE, LORA]
            update_weight(attn, "_wkb", wkb)
            # _wv: [NH, LORA, HEAD]
            update_weight(attn, "_wv", wv)
            # _wq_full: [NH * (NOPE + ROPE), DIM or QLORA]
            update_weight(attn, "_wq_full", wq_full)
            # _wo_full: [DIM, NH * HEAD]
            update_weight(attn, "_wo_full", wo_full)


def rope_freqs(model: Transformer) -> torch.Tensor:
    """
    Precompute frequencies tensor used in RoPE computations.
    """
    head_dim = model.qk_rope_head_dim
    theta = model.pos_embed.rope_theta

    pows = torch.arange(0, head_dim, 2).float() / head_dim
    freqs = 1.0 / (theta**pows)

    if model.pos_embed.get_pos_embed_impl() != PosEmbedImpl.scaled_rope:
        return freqs

    low_freq_factor = model.pos_embed.scaled_rope.low_freq_factor
    high_freq_factor = model.pos_embed.scaled_rope.high_freq_factor
    old_ctx_len = model.pos_embed.scaled_rope.old_context_len
    low_freq_wavelen = old_ctx_len / low_freq_factor
    high_freq_wavelen = old_ctx_len / high_freq_factor
    scaling = model.pos_embed.scaled_rope.scale_factor

    for idx, freq in enumerate(freqs):
        wavelen = 2 * math.pi / freq
        if wavelen > low_freq_wavelen:
            freqs[idx] = freq / scaling
        if high_freq_wavelen <= wavelen <= low_freq_wavelen:
            x = old_ctx_len / wavelen - low_freq_factor
            x /= high_freq_factor - low_freq_factor
            freqs[idx] = (1 - x) * freq / scaling + x * freq

    return freqs
