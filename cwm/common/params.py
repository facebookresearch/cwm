# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
from pathlib import Path
from typing import Any, TypeVar

import torch
from omegaconf import OmegaConf
from upath import UPath

logger = logging.getLogger()

T = TypeVar("T")


TORCH_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


def nested_dict_from_flatten_dict(flat_dict: dict, prefix: str = ".") -> dict:
    """Converts a flattened dictionary to a nested dictionary, using the prefix to determine the nesting."""
    nested_dict: dict = {}
    for key, value in flat_dict.items():
        parts = key.split(prefix)
        current = nested_dict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return nested_dict


def nested_dict_to_flatten_dict(
    nested_dict: dict,
    prefix: str = ".",
    path: str = "",
    join_list: str | None = None,
) -> dict:
    """Converts a nested dictionary to a flattened dictionary, using the prefix to determine the nesting."""
    flat_dict: dict = {}
    for key, value in nested_dict.items():
        _key = f"{path}{prefix}{key}" if path else key
        if isinstance(value, dict):
            flat_dict.update(
                nested_dict_to_flatten_dict(value, path=_key, join_list=join_list),
            )
        else:
            flat_value = value
            if join_list is not None and isinstance(value, list):
                flat_value = join_list.join([str(v) for v in value])
            flat_dict[_key] = flat_value
    return flat_dict


def dataclass_from_dict(cls: type[T], data: dict) -> T:
    """Converts a dictionary to a dataclass instance, recursively for nested structures."""
    base = OmegaConf.structured(cls)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))


def dataclass_to_dict(dataclass_instance: T) -> dict:
    """Converts a dataclass instance to a dictionary, recursively for nested structures."""
    if isinstance(dataclass_instance, dict):
        return dataclass_instance
    return OmegaConf.to_container(
        OmegaConf.structured(dataclass_instance),
        resolve=True,
    )


def override_dataclass_with_dict(dataclass_instance: T, data: dict) -> T:
    """Override a dataclass instance with dict."""
    base = OmegaConf.structured(dataclass_instance)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))


def load_params(params_file: Path | UPath | str, dataclass_cls: type[T]) -> T:
    """Load a parameters file as instance of the specified data class."""
    params_file = UPath(params_file)
    params = OmegaConf.to_container(OmegaConf.load(params_file), resolve=True)
    return dataclass_from_dict(dataclass_cls, params)


def save_params(
    params: Any,
    path: Path | UPath | str,
    *,
    log_params: bool = True,
) -> None:
    """Dump the params dataclass into yaml parameters file."""
    yaml_dump = OmegaConf.to_yaml(OmegaConf.structured(params))
    if log_params:
        logger.info("Using the following params for this run:")
        logger.info(yaml_dump)
    UPath(path).write_text(yaml_dump)


def save_params_to_json(path: Path | UPath | str, params: T) -> None:
    """Dump the params dataclass into json parameters file."""
    with UPath(path).open("w") as f:
        params_dict = dataclass_to_dict(params)
        json.dump(params_dict, f, indent=4)


def load_params_from_json(
    path: Path | UPath | str, dataclass_cls: type[T], key: str | None = None
) -> T:
    """Load a json parameters file as instance of the specified dataclass."""
    with UPath(path).open("r") as f:
        params_dict = json.load(f)
        return dataclass_from_dict(dataclass_cls, params_dict.get(key, params_dict))


def load_from_cli(
    dataclass_cls: type[T], *, from_config_file: bool = False, with_preset: bool = False
) -> T:
    """Loading dataclass parameters or args from CLI.
    The config can be read either from a file-config specified through CLI with the config key
    or through cli args, defaulting to the default args from the data class.
    Our command line interface here uses OmegaConf
    https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    and therefore accepts arguments as a dot list and load parameters from dataclasses,
    e.g. with the following classes:

    .. code-block:: python

        @dataclass
        class DummyArgs:
            name: str
            mode: TransformerArgs

        @dataclass
        class ModelArgs:
            dim: int

    Then you can pass model.dim=32 to change values in ModelArgs
    or just name=foo for top level attributes.
    The parsing behaviour is as specified hereafter:
    1. We instantiate the parameters with its defaults from the dataclass
    2. If from_config_file = true, we override those default values with the ones provided in a config file
    3. We override the result with the additional arguments provided through command line
    If with_preset = true, we will load parameters from the preset configuration file
    specified either in the config file or the command line arguments and these parameters
    will override the defaults from the dataclass and themselves be overridden by the
    file config parameters and command line parameters.
    """

    default_cfg = OmegaConf.structured(dataclass_cls)
    cli_args = OmegaConf.from_cli()
    logger.info(f"CLI args: {cli_args}")

    file_cfg = None
    if from_config_file:
        file_cfg = OmegaConf.load(cli_args.config)
        # We remove the 'config' attribute from config as the underlying DataClass does not have them
        del cli_args.config

    preset_cfg = None
    # If with_preset, we allow loading presets from a dedicated config
    # this preset can either be specified in the config file or
    # dynamically through the cli args but will in any case be
    # overridden by the file config and the cli args
    if with_preset:
        preset_cfgs = []
        if "__preset_config" in cli_args:
            preset_cfg_from_cli = OmegaConf.load(cli_args.__preset_config)
            preset_cfgs.append(preset_cfg_from_cli)
            del cli_args.__preset_config

        if file_cfg and "__preset_config" in file_cfg:
            preset_cfg_from_file = OmegaConf.load(file_cfg.__preset_config)
            preset_cfgs.append(preset_cfg_from_file)
            del file_cfg.__preset_config

        if len(preset_cfgs) > 0:
            preset_cfg = OmegaConf.merge(*preset_cfgs)

    # building the configs by order:
    # first default, then from preset file, then from file, then from cli args
    ordered_cfgs = [default_cfg]
    if preset_cfg:
        ordered_cfgs.append(preset_cfg)
    if file_cfg:
        ordered_cfgs.append(file_cfg)
    ordered_cfgs.append(cli_args)
    cfg = OmegaConf.merge(*ordered_cfgs)

    return OmegaConf.to_object(cfg)
