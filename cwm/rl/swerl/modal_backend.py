# Copyright (c) Meta Platforms, Inc. and affiliates.

import contextlib
import json
import logging
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Concatenate, ParamSpec, Protocol, TypeVar, runtime_checkable

import grpclib
import modal
import tenacity

from .errors import BackendInitError
from .remote.client import PersistentClient, ServerError, SessionOutput
from .tools import (
    BashResult,
    ToolBackend,
    ToolType,
    make_python_plugins_from_dir,
)


@dataclass(frozen=True)
class ModalConfig:
    image_url: str
    start_script: str
    session_timeout: float
    plugin_root: str
    bind_target: str
    tools: dict[str, ToolType]
    plugin_names: list[str]
    server_python_path: str = "python3"
    # 40 minutes hard timeout
    sandbox_timeout: int = 2400
    tunnel_creation_timeout: int = 360
    # (minimum, hard limit)
    cpu: tuple[float, float] | None = (0.125, 4.0)
    # in MiB
    memory: tuple[int, int] | None = (1024, 16384)  # (1G, 16G)
    block_network: bool = False


APP_NAME = "swe-remote"
_MODAL_APP: modal.app.App | None = None
_MODAL_APP_LOCK = threading.Lock()

# Silent heavy modal logging
logging.getLogger("hpack.hpack").setLevel(logging.WARNING)
logging.getLogger("hpack.table").setLevel(logging.WARNING)
logging.getLogger("modal-utils").setLevel(logging.WARNING)
logging.getLogger("modal-client").setLevel(logging.WARNING)


def global_modal_app() -> modal.app.App:
    """
    Get the global modal app.
    Modal has issues with handling concurrent app creation.
    So the app must exist before we start
    """
    global _MODAL_APP
    if _MODAL_APP is not None:
        return _MODAL_APP

    with _MODAL_APP_LOCK:
        # we need a second check because it's possible that another thread has created the app
        # and the lock just got released
        if _MODAL_APP is None:
            try:
                _MODAL_APP = modal.App.lookup(APP_NAME, create_if_missing=False)
            except Exception as e:
                creation_cmd = """
                python -c "import modal; modal.App.lookup('swe-remote', create_if_missing=True)"
                """.strip()
                raise AssertionError(
                    f"The modal app '{APP_NAME}' must exist. Create it with {creation_cmd}."
                ) from e
    assert _MODAL_APP is not None
    return _MODAL_APP


ENCRYPTED_PORT = 8888
REMOTE_PACKAGE_PATH = (Path(__file__).parent / "remote").as_posix()
REMOTE_PACKAGE_NAME = "sweremote"
SESSION_ID = "modal"
WAITING_INTERVAL = 0.5
MAX_WAITING_TIME = 120
BACKGROUND_SLEEP_INTERVAL = 0.2

MODAL_RETRY = tenacity.retry(
    stop=tenacity.stop_after_attempt(10),
    wait=tenacity.wait_exponential_jitter(initial=5, max=60),
    retry=tenacity.retry_if_exception_type(grpclib.exceptions.GRPCError),
)


MAX_DEBUGGING_LEN = 102400


def default_secret(docker_url: str) -> modal.Secret | None:
    """
    Get the default secret for the given docker URL.
    """
    if docker_url.startswith("img-repo.c9x.me"):
        return modal.Secret.from_name("img-repo-guest")
    return None


def init_modal_sandbox(
    config: ModalConfig, use_tunnel: bool = False
) -> tuple[modal.Sandbox, dict[str, ToolType]]:
    # Tools
    plugin_tools = make_python_plugins_from_dir(
        config.plugin_root,
        config.bind_target,
        plugin_names=config.plugin_names,
    )
    tools = {
        **config.tools,
        **plugin_tools,
    }

    # Modal sandbox
    secret = default_secret(config.image_url)

    # Modal sandbox
    image = (
        modal.Image.from_registry(config.image_url, secret=secret)
        .entrypoint([])
        .add_local_dir(config.plugin_root, remote_path=config.bind_target, copy=True)
        .add_local_dir(
            REMOTE_PACKAGE_PATH, remote_path=f"/{REMOTE_PACKAGE_NAME}", copy=True
        )
    )
    app = global_modal_app()

    # TIP: server logs can be retrieved after sandbox.terminate
    # and sandbox.stdout/err.read()
    @MODAL_RETRY
    def create_sandbox():
        # TIP: server logs can be retrieved after sandbox.terminate
        # and sandbox.stdout/err.read()
        return modal.Sandbox.create(
            # NOTE: this should be point to a newer version of python (e.g., 3.10)
            config.server_python_path,
            "-m",
            f"{REMOTE_PACKAGE_NAME}.server",
            "--port",
            str(ENCRYPTED_PORT),
            workdir="/",
            app=app,
            image=image,
            unencrypted_ports=[ENCRYPTED_PORT] if use_tunnel else [],
            timeout=config.sandbox_timeout,
            cpu=config.cpu,
            memory=config.memory,
            block_network=config.block_network,
        )

    try:
        sandbox = create_sandbox()
    except Exception as e:
        raise BackendInitError(
            "Exceeded maximum retries for sandbox creation due to rate limit"
        ) from e
    return sandbox, tools


_P = ParamSpec("_P")
_R = TypeVar("_R")


@runtime_checkable
class ModalBackend(ToolBackend, Protocol):
    _tools: dict[str, ToolType]
    sandbox: modal.Sandbox
    config: ModalConfig
    sandbox_start_time: float

    def run_client(
        self,
        method: Callable[Concatenate[PersistentClient, _P], _R],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R: ...

    @property
    def tools(self) -> dict[str, ToolType]:
        return self._tools

    def create_session(self) -> None:
        # Wait until it's alive
        start_time = time.time()
        while True:
            time.sleep(WAITING_INTERVAL)
            if self.sandbox.poll() is not None:
                raise BackendInitError(self.sandbox.stderr.read())
            try:
                self.run_client(
                    PersistentClient.create_session,
                    session_id=SESSION_ID,
                    command_args=["/bin/bash"],
                    session_timeout=self.config.session_timeout,
                    start_script=self.config.start_script,
                )
                break
            except Exception as e:
                if time.time() - start_time > MAX_WAITING_TIME:
                    raise BackendInitError("Modal server not started in time") from e
                continue
        # Record the start time of the sandbox
        self.sandbox_start_time = time.perf_counter()

    @property
    def duration(self) -> float:
        return time.perf_counter() - self.sandbox_start_time

    def timed_out(self, allowed_error_seconds: int = -30) -> bool:
        return self.duration > self.config.sandbox_timeout + allowed_error_seconds

    def _run_bash(self, command: str) -> SessionOutput:
        """
        Run a bash command in the sandbox and return the output.
        """
        return self.run_client(
            PersistentClient.run_session,
            session_id=SESSION_ID,
            command=command,
            sanitize=False,
        )

    def run_bash(self, command: str) -> BashResult:
        result = self._run_bash(command)
        if (
            result["status"] == "error"
            or (exit_code := self.run_client(PersistentClient.get_exitcode, SESSION_ID))
            is None
        ):
            exit_code = -1
        # Here exit_code will always be in scope because the if condition is False only when
        # both checks are False, and then exit_code is set to -1
        return BashResult(**result, exit_code=exit_code)  # type: ignore

    def __del__(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        # poll is None means the sandbox is still running
        with contextlib.suppress(Exception):
            if hasattr(self, "sandbox") and self.sandbox.poll() is None:
                self.sandbox.terminate()

    def get_debugging_info(self, concise: bool = True) -> str:
        sandbox_id = self.sandbox.object_id
        try:
            task_id = str(self.sandbox._get_task_id())
        except AttributeError:
            task_id = "N/A"
        concise_message = (
            f"duration: {self.duration}\nsandbox_id: {sandbox_id}\ntask_id: {task_id}"
        )
        if concise:
            return concise_message
        stdout = self.sandbox.stdout.read()[-MAX_DEBUGGING_LEN:]
        stderr = self.sandbox.stderr.read()[-MAX_DEBUGGING_LEN:]
        message = f"stdout: {stdout}\nstderr: {stderr}\n{concise_message}"
        return message

    def __str__(self):
        return str(self.config)

    def __repr__(self):
        return str(self)


class ModalBackend_NoTunnel(ModalBackend):
    """
    A tunnel-free modal backend where we deploy the server in the sandbox and communicate it using
    client_cli.py through `sandbox.exec`. It's slower but more stable than using a tunnel.
    Speed can be optimized by making client_cli also persistent, but this overhead is negligible
    compared to LLM inference and parallel sandbox execution.
    """

    def __init__(self, config: ModalConfig) -> None:
        self.config = config
        self.sandbox, self._tools = init_modal_sandbox(config, use_tunnel=False)
        self.create_session()

    def run_client(
        self,
        method: Callable[Concatenate[PersistentClient, _P], _R],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        proc = self.sandbox.exec(
            self.config.server_python_path,
            "-m",
            f"{REMOTE_PACKAGE_NAME}.client_cli",
            "--host",
            "localhost",
            "--port",
            str(ENCRYPTED_PORT),
        )
        payload = dict(
            method=method.__name__,
            args=args,
            kwargs=kwargs,
        )
        payload_str = json.dumps(payload)
        proc.stdin.write(payload_str)
        proc.stdin.write_eof()
        proc.stdin.drain()
        returncode = proc.wait()
        if returncode != 0:
            raise ServerError(
                f"Error running server method {method.__name__}. "
                f"Return code: {returncode}.\n"
                f"Output: {proc.stdout.read()}\nError: {proc.stderr.read()}"
            )
        else:
            output = proc.stdout.read()
            return json.loads(output)


class ModalBackend_Tunnel(ModalBackend):
    @property
    def client(self) -> PersistentClient:
        # Each time we build a new client. We can't hold a persistent connection
        # since idle TCP connections are getting killed ~every 120s.
        return PersistentClient(self.host, self.port, False)

    def run_client(
        self,
        method: Callable[Concatenate[PersistentClient, _P], _R],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        return method(self.client, *args, **kwargs)

    def __init__(self, config: ModalConfig, background_mode: bool = False) -> None:
        warnings.warn(
            "ModalBackend_Tunnel is not recommended due to the instability of the tunnel.",
            stacklevel=2,
        )

        if background_mode:
            warnings.warn(
                "Using background mode. Make sure this is intended.", stacklevel=2
            )
            # XXX: keep it for now but method assignment is generally not recommended
            self._run_bash = self.run_bash_background  # type: ignore

        self.config = config
        # Tools
        self.sandbox, self._tools = init_modal_sandbox(config, use_tunnel=True)
        try:
            self.host, self.port = self.sandbox.tunnels()[ENCRYPTED_PORT].tcp_socket
        except Exception as e:
            raise BackendInitError("Failed to create tunnel") from e

        self.create_session()

    def run_bash_background(self, command: str) -> SessionOutput:
        # Sanitization would happen in feedback function, so we don't need to sanitize here
        # We do a background loop to send a command in background & periodically check its
        # status to prevent TCP connection being killed for long running commands..

        # Just use the same client to do the loop
        client = self.client
        task_id = client.submit_run_session(
            session_id=SESSION_ID,
            command=command,
            sanitize=False,
        )
        result: SessionOutput = PersistentClient.fetch_task_result_until_success(
            get_client=lambda: client,
            task_id=task_id,
            sleep_interval=BACKGROUND_SLEEP_INTERVAL,
        )
        return result
