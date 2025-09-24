# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Base configs and modules for Transformer-based decoder-only language model.
Refer to the fastgen implementation for forward pass implementation for inference.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import nn
from xformers.ops import fmha

logger = logging.getLogger()
fmha.dispatch._USE_FLASH_ATTENTION_3 = True


class PosEmbedImpl(Enum):
    rope = "rope"
    scaled_rope = "scaled_rope"


@dataclass
class ScaledRoPEArgs:
    scale_factor: int = 8
    old_context_len: int = 8192
    low_freq_factor: int = 1
    high_freq_factor: int = 4
    use_attn_scale: bool = False

    def __post_init__(self) -> None:
        self.low_freq_wavelen = self.old_context_len / self.low_freq_factor
        self.high_freq_wavelen = self.old_context_len / self.high_freq_factor

        if self.scale_factor == 1:
            raise ValueError("No scaling happens with scale_factor==1.")
        if self.low_freq_wavelen < self.high_freq_wavelen:
            raise ValueError(
                "Invalid self.low_freq_wavelen < self.high_freq_wavelen:"
                f"{self.low_freq_wavelen} < {self.high_freq_wavelen}"
            )


@dataclass
class PosEmbedArgs:
    pos_embed_impl: str
    rope_theta: float
    scaled_rope: ScaledRoPEArgs = field(default_factory=ScaledRoPEArgs)

    def get_pos_embed_impl(self) -> PosEmbedImpl:
        return PosEmbedImpl(self.pos_embed_impl)


@dataclass
class WindowAttentionArgs:
    # sliding window interleaving pattern
    # 6 corresponds to 5:1 local:global pattern, i.e. global attention every 6 layers
    # 1 corresponds to all layers using sliding window attention with 'global_window' size
    window_pattern: int = 6
    # size of the global sliding window, use full attention if not specified
    # the 'global_window' can be set directly when creating the attention masks
    # allowing for dynamic global window sizes (i.e. changing across batches)
    global_window: int | None = None
    # size of the local sliding window
    local_window: int | None = None
    # whether to enforce using global attention for the last layer
    global_last_layer: bool = True

    def __post_init__(self) -> None:
        assert self.window_pattern > 0
        assert (
            self.local_window is not None and self.local_window > 0
        ) or self.window_pattern == 1
        if self.global_window is not None:
            assert self.global_window > 0

    @staticmethod
    def is_global_attention_layer(
        layer_id: int, n_layers: int, global_last_layer: bool, window_pattern: int
    ) -> bool:
        if global_last_layer and layer_id == (n_layers - 1):
            return True
        return layer_id % window_pattern == 0


@dataclass
class TransformerArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None
    head_dim: int | None = None  # if None, use dim // n_heads

    max_seq_len: int = 1024
    vocab_size: int = -1

    ffn_dim_multiplier: float | None = None
    ffn_exp: int = 4

    # enforces that the SwiGLU hidden layer size
    # is a multiple of large power of 2.
    multiple_of: int = 256
    norm_eps: float = 1e-5

    # positional encoding parameter, increase for longer context
    pos_embed_impl: str = "rope"
    rope_theta: float = 10000.0
    scaled_rope: ScaledRoPEArgs = field(default_factory=ScaledRoPEArgs)

    # attention implementation and bias type
    # (xformers is the only supported implementation for now)
    attn_impl: str = "xformers"
    attn_bias_type: str | None = None
    window_attn: WindowAttentionArgs | None = None

    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 0  # > 0 to enable
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64

    # sharing input and output embedding weights
    weight_tying: bool = False

    # add biases in attention
    qkv_biases: bool = False

    # apply RMSNorm to xq, xk in attention
    qk_norm: bool = False

    def __post_init__(self) -> None:
        assert self.attn_impl == "xformers"
        assert not self.weight_tying, "Weight tying is not supported yet"
        if self.n_kv_heads is not None:
            assert self.n_heads % self.n_kv_heads == 0
        if self.qk_norm and self.kv_lora_rank:
            raise RuntimeError("MLA and QK norms are incompatible")

    def get_pos_embed_impl(self) -> PosEmbedImpl:
        return PosEmbedImpl(self.pos_embed_impl)

    def get_head_dim(self) -> int:
        return self.head_dim if self.head_dim is not None else self.dim // self.n_heads


class Attention(nn.Module):
    def __init__(self, args: TransformerArgs) -> None:
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = (
            args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        )
        assert self.n_heads % self.n_kv_heads == 0
        self.heads_per_group = self.n_heads // self.n_kv_heads
        self.qkv_biases = args.qkv_biases
        self.qk_norm = args.qk_norm
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.head_dim = args.get_head_dim()
        self.qk_head_dim = (
            self.head_dim
            if self.kv_lora_rank == 0
            else args.qk_nope_head_dim + args.qk_rope_head_dim
        )
        if self.kv_lora_rank > 0 or self.q_lora_rank > 0:
            raise NotImplementedError("MLA not implemented")
        if self.qk_norm:
            raise NotImplementedError("QK norm not implemented")

        self.wq = nn.Linear(
            self.dim,
            self.n_heads * self.qk_head_dim,
            bias=self.qkv_biases,
        )
        self.wk = nn.Linear(
            self.dim,
            self.n_kv_heads * self.qk_head_dim,
            bias=self.qkv_biases,
        )
        self.wv = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=self.qkv_biases,
        )

        self.wo = nn.Linear(
            self.n_heads * self.head_dim,
            self.dim,
            bias=False,
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
        mp_size: int = 1,
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        args: TransformerArgs,
        layer_id: int,
    ) -> None:
        super().__init__()
        n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        assert args.n_heads % n_kv_heads == 0
        assert args.dim % args.n_heads == 0
        self.layer_id = layer_id
        self.qkv_biases = args.qkv_biases
        self.qk_norm = args.qk_norm
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_exp * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        if window_attn := getattr(args, "window_attn", None):
            if window_attn.is_global_attention_layer(
                layer_id,
                args.n_layers,
                window_attn.global_last_layer,
                window_attn.window_pattern,
            ):
                if window_attn.global_window is None:
                    self.window_size_left = -1
                else:
                    self.window_size_left = window_attn.global_window - 1
            else:
                if window_attn.local_window is not None:
                    self.window_size_left = window_attn.local_window - 1
                else:
                    self.window_size_left = -1
        else:
            self.window_size_left = -1


class Transformer(nn.Module):
    def __init__(self, args: TransformerArgs) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.model_dim = args.dim
        self.head_dim = args.get_head_dim()
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_head_dim = (
            self.head_dim
            if args.kv_lora_rank == 0
            else args.qk_nope_head_dim + args.qk_rope_head_dim
        )
        self.qk_rope_head_dim = (
            args.qk_rope_head_dim if args.kv_lora_rank > 0 else self.head_dim
        )
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.weight_tying = args.weight_tying
        self.pos_embed = PosEmbedArgs(
            args.pos_embed_impl, args.rope_theta, args.scaled_rope
        )
        self.max_seq_len = args.max_seq_len
        self.window_attn = args.window_attn
        self.attn_bias_type = args.attn_bias_type or "causal"
        self.qkv_biases = args.qkv_biases
        self.qk_norm = args.qk_norm

        assert self.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(args, layer_id))

        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

    @property
    def n_layers(self) -> int:
        return len(self.layers)
