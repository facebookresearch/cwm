# Copyright (c) Meta Platforms, Inc. and affiliates.

import importlib
from pathlib import Path

# Recursively import all python modules except those starting with an underscore
base_dir = Path(__file__).parent
for file in base_dir.rglob("*.py"):
    relative_path = file.relative_to(base_dir).with_suffix("")
    if str(relative_path.name).startswith("_"):
        continue
    module_path = ".".join(relative_path.parts)
    importlib.import_module(f"cwm.rl.envs.envs.{module_path}")
