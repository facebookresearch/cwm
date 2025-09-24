# Copyright (c) Meta Platforms, Inc. and affiliates.

# ruff: noqa
import json
import random
import socket
import ssl
import time
import contextlib
import sys

if sys.version_info >= (3, 9):
    from collections.abc import Callable
    from typing import TypedDict  # Python 3.9+
else:
    from typing import Callable
    from typing_extensions import TypedDict  # Python 3.8

from pathlib import Path
from typing import Any, Optional, List, Tuple
from .session import SessionOutput


# If socket doesn't receive data for this long, we assume the server is blocked.
# This should not happen at all if the server is bug-free and session timeout < SOCKET_TIMEOUT
SOCKET_TIMEOUT = 900
MAX_BUFFER_BYTES = 6400 * 1024


class ServerError(Exception):
    pass


class RunCommandResponse(TypedDict):
    returncode: int
    output: str
    timeout: bool


class CheckTaskStatusResponse(TypedDict):
    is_done: bool
    task_result: Any


class PersistentClient:
    """
    A synchronized client to communicate with the TCP server.

    NOTE: the client is persistent and not thread safe. In a multi-threaded environment,
    please create a new client for each thread.
    """

    @staticmethod
    def from_registry(registry_path: str) -> "PersistentClient":
        """Get the next server address from the server registry"""
        path = Path(registry_path)
        assert path.exists()
        server_strs = (p for p in path.rglob("*") if p.is_file())
        servers: List[Tuple[str, int]] = []
        for server_str in server_strs:
            server, port_str = server_str.stem.split(":")
            port = int(port_str)
            servers.append((server, port))
        host, port = random.SystemRandom().choice(servers)
        return PersistentClient(host, port)

    def __init__(self, host: str, port: int, tls_encryption: bool = False):
        # Only used for __str__
        self.host = host
        self.port = port

        raw_sock = socket.create_connection((host, port), timeout=SOCKET_TIMEOUT)

        if tls_encryption:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            self.sock = context.wrap_socket(raw_sock, server_hostname=host)
        else:
            self.sock = raw_sock  # type: ignore

        self.sock_file = self.sock.makefile("rb")

    def communicate(self, request: dict) -> dict:
        """
        This is a global API to communicate with the server.
        """
        try:
            return self._communicate(request)
        except ssl.SSLError as e:
            raise ServerError(f"SSL error: {e}") from e

    def _communicate(self, request: dict) -> dict:
        # Convert to JSON, add a newline so the server knows where to stop reading
        request_str = json.dumps(request) + "\n"
        self.sock.sendall(request_str.encode("utf-8"))
        # Read the response
        response_data = self.sock_file.readline(MAX_BUFFER_BYTES)

        if not response_data:
            raise ServerError("Empty response from server. Please check the server.")

        if not response_data.endswith(b"\n"):
            raise ServerError(f"Response too long from server ({len(response_data)})")

        resp_str = response_data.decode("utf-8").strip()
        response = json.loads(resp_str)
        if response["status"] == "error":
            raise ServerError(response)
        return response

    def __str__(self):
        return f"{self.__class__.__name__}({self.host}:{self.port})"

    def __enter__(self):
        # Support context manager
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        # Support context manager
        self.destroy()

    def destroy(self) -> None:
        # Close the connection when done
        with contextlib.suppress(Exception):
            if hasattr(self, "sock_file") and not self.sock_file.closed:
                self.sock_file.close()
            if hasattr(self, "sock") and self.sock.fileno() != -1:
                self.sock.close()

    def __del__(self):
        self.destroy()

    # Let's define some typed APIs
    # They need to sync with the server APIs
    def run_command(
        self,
        command_args: List[str],
        timeout: Optional[float] = None,
    ) -> RunCommandResponse:
        """
        Run a command
        """
        request = dict(action="run_command", command_args=command_args, timeout=timeout)
        response = self.communicate(request)
        return response["result"]

    def get_new_id(self) -> str:
        """
        Get a new session ID from the server.
        """
        request = dict(
            action="get_new_id",
        )
        response = self.communicate(request)
        return response["result"]

    def create_session(
        self,
        session_id: str,
        command_args: List[str],
        session_timeout: float,
        start_script: Optional[str] = None,
    ) -> None:
        """
        Create a new session with the given command arguments.
        """
        request = dict(
            action="create_session",
            session_id=session_id,
            command_args=command_args,
            session_timeout=session_timeout,
            start_script=start_script,
        )
        self.communicate(request)

    def run_session(
        self,
        session_id: str,
        command: str,
        sanitize: bool = True,
    ) -> SessionOutput:
        """
        Run a command in the given session.
        """
        request = dict(
            action="run_session",
            command=command,
            session_id=session_id,
            sanitize=sanitize,
        )
        response = self.communicate(request)
        return response["result"]

    def get_exitcode(self, session_id: str) -> Optional[int]:
        """
        Get the exit code of the last command in the given session.
        """
        request = dict(
            action="get_exitcode",
            session_id=session_id,
        )
        response = self.communicate(request)
        assert isinstance(response["result"], int) or response["result"] is None
        return response["result"]

    def stop_session(self, session_id: str) -> None:
        """
        Stop the given session.
        """
        request = dict(
            action="stop_session",
            session_id=session_id,
        )
        self.communicate(request)

    def is_server_alive(self) -> bool:
        """
        Check if the server is alive.
        """
        request = dict(
            action="is_server_alive",
        )
        response = self.communicate(request)
        assert response["result"]
        return response["result"]

    # Let's define some typed APIs
    # They need to sync with the server APIs
    def check_task_status(self, task_id: str) -> CheckTaskStatusResponse:
        request = dict(action="check_task_status", task_id=task_id)
        response = self.communicate(request)
        return response["result"]  # type: ignore

    def submit_run_command(
        self,
        command_args: List[str],
        timeout: Optional[float] = None,
    ) -> str:
        """
        Run a command in the background
        """
        request = dict(
            action="run_command",
            command_args=command_args,
            timeout=timeout,
            background=True,
        )
        response = self.communicate(request)
        return response["result"]["task_id"]

    def submit_run_session(
        self,
        session_id: str,
        command: str,
        sanitize: bool = True,
    ) -> str:
        """
        Run a command in the given session in the background.
        """
        request = dict(
            action="run_session",
            command=command,
            session_id=session_id,
            sanitize=sanitize,
            background=True,
        )
        response = self.communicate(request)
        return response["result"]["task_id"]

    @staticmethod
    def fetch_task_result_until_success(
        get_client: Callable[[], "PersistentClient"],
        task_id: str,
        sleep_interval: float,
    ) -> Any:
        """
        Helper function to repeatedly check a background task until it succeeds.
        """
        while True:
            client = get_client()
            response: CheckTaskStatusResponse = client.check_task_status(task_id)
            if response["is_done"]:
                return response["task_result"]  # type: ignore
            # Wait before checking again
            time.sleep(sleep_interval)
