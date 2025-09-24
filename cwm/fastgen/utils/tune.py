# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import torch.distributed
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from cwm.model.transformer import Transformer as CWMModel

    from .model import Transformer as FastgenModel

    TransformerModel: TypeAlias = CWMModel | FastgenModel


@dataclass
class _ModelParams:
    params: int
    per_token_activation: int
    per_token_cache: int


@dataclass
class MemParams:
    cache_tokens: int
    prefill_tokens: int

    def round_to(self, n: int) -> None:
        self.cache_tokens = _round_to(self.cache_tokens, n)
        self.prefill_tokens = _round_to(self.prefill_tokens, n)


def _round_to(x: int, n: int) -> int:
    return (x + n - 1) // n * n


def _model_params(
    device_mesh: DeviceMesh | None,
    model: "TransformerModel",
) -> _ModelParams:
    model_params = 0
    for p in model.parameters():
        if isinstance(p, DTensor):
            p = p.to_local()
        model_params += p.numel() * p.element_size()

    if device_mesh:
        t = torch.tensor(model_params, device="cuda")
        torch.distributed.all_reduce(
            t,
            op=torch.distributed.ReduceOp.MAX,
            group=device_mesh.get_group(),
        )
        model_params = int(t.item())

    ffn_w1 = model.layers[0].feed_forward.w1.weight
    if isinstance(ffn_w1, DTensor):
        ffn_w1 = ffn_w1.to_local()
    ffn_act = 2 * ffn_w1.shape[0]  # assumes bf16
    hidden_act = 2 * ffn_w1.shape[1]

    # Each layer keeps the skip connection live,
    # then in the ffn we have x1, x3, and h_in_ffn
    # live simultaneously.
    activation = 2 * hidden_act + 2 * ffn_act

    # Cut ourselves some slack.
    activation = int(activation * 1.2)

    if model.kv_lora_rank > 0:
        lora_dim = model.kv_lora_rank
        rope_dim = model.qk_rope_head_dim
        layer_cache = 2 * (lora_dim + rope_dim)
        if device_mesh is not None:
            layer_cache //= device_mesh.size()
    else:
        local_heads_dim = model.qk_head_dim + model.head_dim
        local_heads_dim *= model.layers[0].attention.n_kv_heads
        layer_cache = 2 * local_heads_dim

    return _ModelParams(
        params=model_params,
        per_token_activation=activation,
        per_token_cache=layer_cache * len(model.layers),
    )


def mem_params(
    device_mesh: DeviceMesh | None,
    model: "TransformerModel",
    prefill_gb: float,
    gpu_gb: float,
) -> MemParams:
    """
    Compute memory-related generation parameters.
    """
    p = _model_params(device_mesh, model)
    prefill = int(prefill_gb * 1e9)
    avail = max(prefill, int(gpu_gb * 1e9) - p.params)
    prefill_tokens = prefill // p.per_token_activation
    cache_tokens = (avail - prefill) // p.per_token_cache
    return MemParams(
        cache_tokens=cache_tokens,
        prefill_tokens=prefill_tokens,
    )
