# Copyright (c) Meta Platforms, Inc. and affiliates.

# flake8: noqa
from typing import NotRequired, TypedDict


class InitCmd(TypedDict):
    cmd: str
    shell: bool


class TestSpec(TypedDict):
    instance_id: str
    PASS_TO_PASS: str | list[str]
    FAIL_TO_PASS: str | list[str]
    image_path: str
    docker_url: str
    eval_script: str
    test_parser: NotRequired[str]
    init_commands: NotRequired[list[InitCmd]]
    version: NotRequired[str]
