# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Predefined environment constants
"""

from pathlib import Path

# ulimit is in KB
# 1792g is the max
# 224g per process (/ 8)
# let's set a 12G cap
MEM_LIMIT = 12 * 1024 * 1024
THREAD_LIMIT = 65536

SESSION_START_SCRIPT_BASE = f"\
export LC_ALL=en_US.UTF-8 && \
export LANG=en_US.UTF-8 && \
source ~/.bashrc && \
ulimit -v {MEM_LIMIT} && \
ulimit -d {MEM_LIMIT} && \
ulimit -u {THREAD_LIMIT}"

DEFAULT_PYTHON_PATH = "/opt/miniconda3/bin/python3"

PLUGIN_SETUP_SCRIPT = "export PYPLUGIN_PYTHON_PATH={plugin_python_path}"

SESSION_START_SCRIPT = f"{PLUGIN_SETUP_SCRIPT} && {SESSION_START_SCRIPT_BASE}"

# Plugins
PLUGIN_NAMES = ["edit", "create"]
PLUGIN_ROOT = (Path(__file__).parent / "plugins").resolve().as_posix()
PLUGIN_BIND_TARGET = "/swerl-plugins"
