# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import time

import torch
from torch.distributed.device_mesh import DeviceMesh
from upath import UPath

from cwm.checkpoint.checkpointer import Checkpointer
from cwm.common.params import (
    dataclass_from_dict,
)
from cwm.model.parallelize_transformer import parallelize_model
from cwm.model.transformer import Transformer, TransformerArgs
from cwm.text.tokenizers import Tokenizer, build_tokenizer

logger = logging.getLogger()


def build_fastgen_model(
    world_mesh: DeviceMesh,
    checkpoint_dir: str | None = None,
    model_args: TransformerArgs | None = None,
    vocab_parallel: bool = False,
    loss_parallel: bool = False,
) -> Transformer:
    # NOTE that we apply tp when tp_size > 1 but never apply fsdp because fastgen doesn't support it
    assert (model_args is None) != (checkpoint_dir is None)

    start_time = time.time()
    if model_args is None:
        logger.info(f"Building model from checkpoint from: {checkpoint_dir}")
        ckpt_dir = UPath(checkpoint_dir)
        assert ckpt_dir.exists(), f"No checkpoint dir found at: {ckpt_dir}"

        logger.info("Loading model configuration")
        with UPath(ckpt_dir / "params.json").open("r") as f:
            params = json.load(f)
        model_args = dataclass_from_dict(TransformerArgs, params["model"])

    logger.info("Instantiating model")
    with torch.device("meta"):
        assert model_args is not None
        model = Transformer(model_args)

    parallelize_model(
        model,
        world_mesh,
        param_dtype=torch.bfloat16,
        vocab_parallel=vocab_parallel,
        loss_parallel=loss_parallel,
    )
    model = model.to_empty(device="cuda")

    if checkpoint_dir is not None:
        logger.info("Loading state dict from checkpoint")
        checkpoint = Checkpointer(model=model)
        checkpoint.load_from_path(ckpt_dir)

    # Make sure all ranks consolidated the checkpoints before moving on
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(f"Reloaded model in {time.time() - start_time:.2f} seconds")

    return model


def build_tokenizer_from_ckpt(checkpoint_dir: str) -> Tokenizer:
    ckpt_dir = UPath(checkpoint_dir)
    assert ckpt_dir.exists(), f"No checkpoint dir found at: {ckpt_dir}"
    logger.info("Loading model configuration")
    with UPath(ckpt_dir / "params.json").open("r") as f:
        params = json.load(f)
    if "data" in params:
        # pre-trained model
        tokenizer_name = params["data"]["tokenizer"]["name"]
        tokenizer_path = params["data"]["tokenizer"].get("path")
    else:
        # from our training run
        tokenizer_name = params["tokenizer"]["name"]
        tokenizer_path = params["tokenizer"].get("path")

    if tokenizer_path is None:
        tokenizer_path = str(UPath(ckpt_dir / "tokenizer.model"))
    logger.info(f"Loading tokenizer {tokenizer_name} from {tokenizer_path}")
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_path)
    return tokenizer
