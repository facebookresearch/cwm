# Copyright (c) Meta Platforms, Inc. and affiliates.

from .default_configs import (
    DEFAULT_PYTHON_PATH,
    PLUGIN_BIND_TARGET,
    PLUGIN_NAMES,
    PLUGIN_ROOT,
    SESSION_START_SCRIPT,
)
from .default_tools import DEFAULT_TOOLS
from .modal_backend import (
    ModalBackend,
    ModalBackend_NoTunnel,
    ModalBackend_Tunnel,
    ModalConfig,
)
from .tools import ToolType


def get_default_modal_backend(
    image_url: str,
    session_timeout: float,
    work_dir: str = "/testbed",
    use_tunnel: bool = False,
    background_mode: bool = False,
    startup_commands: str = "",
    # 40 minutes default timeout
    sandbox_timeout: int = 2400,
    tools: dict[str, ToolType] = DEFAULT_TOOLS,
    server_python_path: str = DEFAULT_PYTHON_PATH,
    block_network: bool = False,
    session_start_script: str | None = None,
    memory: tuple[int, int] | None = (1024, 16384),  # (1G, 16G)
    plugins: bool = True,
) -> ModalBackend:
    if session_start_script is None:
        session_start_script = SESSION_START_SCRIPT.format(
            plugin_python_path=server_python_path
        )
    start_script = f"{session_start_script}\n{startup_commands}\ncd {work_dir}"
    config = ModalConfig(
        image_url=image_url,
        session_timeout=session_timeout,
        plugin_root=PLUGIN_ROOT,
        bind_target=PLUGIN_BIND_TARGET,
        start_script=start_script,
        tools=tools,
        plugin_names=PLUGIN_NAMES if plugins else [],
        server_python_path=server_python_path,
        sandbox_timeout=sandbox_timeout,
        block_network=block_network,
        memory=memory,
    )

    if use_tunnel:
        return ModalBackend_Tunnel(config, background_mode=background_mode)
    else:
        return ModalBackend_NoTunnel(config)
