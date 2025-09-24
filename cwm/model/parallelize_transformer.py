# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Standard parallelization strategy for inference with decoder-only transformer language model.
"""

import logging

import torch
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from cwm.model.transformer import Transformer

logger = logging.getLogger()


def parallelize_model(
    model: Transformer,
    world_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    vocab_parallel: bool = False,
    loss_parallel: bool = False,
) -> None:
    """Apply tensor parallelism parallelism to the model and cast parameters to the specified dtype.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if "tp" in world_mesh.mesh_dim_names and world_mesh["tp"].size() > 1:
        assert not model.weight_tying
        apply_tp(
            model,
            world_mesh["tp"],
            vocab_parallel=vocab_parallel,
            loss_parallel=loss_parallel,
        )

    cast_model_parameters(model, param_dtype)


def cast_model_parameters(model: Transformer, param_dtype: torch.dtype) -> None:
    """Cast model parameters to the specified dtype."""

    def cast_parameters(module: torch.nn.Module):
        for k, p in list(module.named_parameters(recurse=False)):
            # setting an attribute to a value of type torch.nn.Parameter will automatically
            # update the parameter as returned by named_parameters()/parameters()
            setattr(module, k, torch.nn.Parameter(p.to(param_dtype)))

    model.apply(cast_parameters)


SEQPAR_DIM = 0


def apply_tp_to_block(
    transformer_block: torch.nn.Module,
    tp_mesh: DeviceMesh,
) -> None:
    seq_placement = Shard(SEQPAR_DIM)

    layer_plan: dict[str, ParallelStyle] = {}
    col_parallel_op = ColwiseParallel
    row_parallel_op = RowwiseParallel

    layer_plan["attention_norm"] = SequenceParallel(sequence_dim=SEQPAR_DIM)
    layer_plan["ffn_norm"] = SequenceParallel(sequence_dim=SEQPAR_DIM)

    if transformer_block.qk_norm:
        # xq's shape is (bs, seq_len, n_heads, head_dim),
        # so the input is sharded across dim 2 (n_heads)
        layer_plan["attention.q_norm"] = SequenceParallel(
            sequence_dim=2, use_local_output=True
        )
        layer_plan["attention.k_norm"] = SequenceParallel(
            sequence_dim=2, use_local_output=True
        )

    layer_plan["attention.wq"] = col_parallel_op(input_layouts=seq_placement)
    layer_plan["attention.wk"] = col_parallel_op(input_layouts=seq_placement)
    layer_plan["attention.wv"] = col_parallel_op(input_layouts=seq_placement)

    layer_plan["attention.wo"] = row_parallel_op(output_layouts=seq_placement)

    layer_plan["feed_forward.w1"] = col_parallel_op(input_layouts=seq_placement)
    layer_plan["feed_forward.w2"] = row_parallel_op(output_layouts=seq_placement)
    layer_plan["feed_forward.w3"] = col_parallel_op(input_layouts=seq_placement)

    # Adjusting the number of heads and kv heads according to the tp size
    attn_layer = transformer_block.attention
    attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
    attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()
    attn_layer.heads_per_group = attn_layer.n_heads // attn_layer.n_kv_heads
    if transformer_block.layer_id == 0:
        logger.info(
            f"Adjusting transformer local n_heads to {attn_layer.n_heads} and local n_kv_heads to {attn_layer.n_kv_heads}"
        )

    parallelize_module(
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
    )


def apply_tp(
    model: torch.nn.Module,
    tp_mesh: DeviceMesh,
    *,
    vocab_parallel: bool,
    loss_parallel: bool,
) -> None:
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs
    # (which are the first transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallel_plan: dict[str, ParallelStyle] = {}

    if vocab_parallel:
        logger.info("Using vocab parallel")
        input_parallel_op = RowwiseParallel
        output_parallel_op = ColwiseParallel
    else:
        logger.info("Vocab parallel disabled")
        input_parallel_op = ColwiseParallel
        output_parallel_op = RowwiseParallel

    logger.info(f"Using TP with sequence parallel with sequence dim = {SEQPAR_DIM}")
    seq_placement = Shard(SEQPAR_DIM)

    parallel_plan["tok_embeddings"] = input_parallel_op(
        input_layouts=Replicate(),
        output_layouts=seq_placement,
    )
    parallel_plan["norm"] = SequenceParallel(sequence_dim=SEQPAR_DIM)
    parallel_plan["output"] = output_parallel_op(
        input_layouts=seq_placement,
        output_layouts=Shard(-1) if loss_parallel else Replicate(),
        use_local_output=not loss_parallel,
    )

    parallelize_module(model, tp_mesh, parallel_plan)

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we accelerated Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for transformer_block in model.layers:
        apply_tp_to_block(transformer_block, tp_mesh)

    logger.info(
        f"Applied Tensor Parallelism to the model with tp_size={tp_mesh.size()}"
    )
