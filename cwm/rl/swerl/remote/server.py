# Copyright (c) Meta Platforms, Inc. and affiliates.

# ruff: noqa
# A general TCP server that can handle multiple clients and execute commands in a persistent session

import argparse
import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import suppress
from typing import Any, Optional, List, Dict

from .session import AsyncSession, SessionOutput

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
LOGGER = logging.getLogger()

# Maps env_id to the actual environment
# We store IDs to avoid broadcasting issues
SESSION_STORE: Dict[str, AsyncSession] = {}
# Each session should not be shared across multiple clients
SESSION_LOCK: Dict[str, asyncio.Lock] = {}


def get_session_lock(session_id: str) -> asyncio.Lock:
    if session_id not in SESSION_LOCK:
        SESSION_LOCK[session_id] = asyncio.Lock()
    return SESSION_LOCK[session_id]


# 6400KB buffer size should be good enough
MAX_BUFFER_BYTES = 6400 * 1024

# This will be assigned before the server starts
SEMAPHORE: Optional[asyncio.Semaphore] = None
HOST: Optional[str] = None
PORT: Optional[int] = None
ID_COUNTER = 0
HOST_NAME = socket.gethostname()


def get_current_time() -> str:
    """
    Get the current time in a human-readable format.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _get_new_id() -> str:
    """
    Generate a new unique ID across the node.
    """
    global PORT
    assert PORT is not None, "Port must be set before generating IDs"
    global ID_COUNTER
    ID_COUNTER += 1
    # NOTE: without this extra randomness, IDs can duplicate if the server
    # restarts and the resource is not cleaned up properly.
    short_uuid = str(uuid.uuid4())[:8]
    return f"{HOST_NAME}-{PORT}-{ID_COUNTER}-{short_uuid}"


def get_new_id(_: Dict, prefix: str = "") -> str:
    return prefix + _get_new_id()


def create_session(request: Dict) -> None:
    """
    Create a new persistent session and return the session id
    """
    session_id: str = request["session_id"]
    if session_id in SESSION_STORE:
        raise RuntimeError(f"Session ID {session_id} already exists.")
    command_args: List[str] = request["command_args"]
    timeout: float = request["session_timeout"]
    start_script: Optional[str] = request.get("start_script", None)
    session = AsyncSession(
        command_args=command_args,
        timeout=timeout,
        start_script=start_script,
    )
    SESSION_STORE[session_id] = session
    return None


async def run_session(request: Dict) -> SessionOutput:
    session_id: str = request["session_id"]
    command: str = request["command"]
    sanitize: bool = request.get("sanitize", True)
    session = SESSION_STORE[session_id]
    session_lock = get_session_lock(session_id)
    async with session_lock:
        output = await session.communicate(command, sanitize=sanitize)
    return output


async def get_exitcode(request: Dict) -> Optional[int]:
    session_id: str = request["session_id"]
    session = SESSION_STORE[session_id]
    session_lock = get_session_lock(session_id)
    async with session_lock:
        exitcode = await session.get_exitcode()
    return exitcode


async def run_command(request: Dict) -> Dict:
    """Run an arbitrary command and return the output."""
    command: List[str] = request["command_args"]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(), timeout=request["timeout"]
        )
        timeout = False
    except TimeoutError:
        process.kill()
        stdout = b""
        timeout = True

    if process.returncode is None:
        process.terminate()

    return dict(returncode=process.returncode, output=stdout.decode(), timeout=timeout)


def stop_session(request: Dict) -> None:
    session_id: str = request["session_id"]
    session = SESSION_STORE[session_id]
    session.stop()
    del SESSION_STORE[session_id]
    if session_id in SESSION_LOCK:
        del SESSION_LOCK[session_id]
    return None


def is_server_alive(_: Dict) -> bool:
    """As long as this function is triggered, the server is alive."""
    return True


def _test_oom(request: Dict) -> None:
    num_bytes = request["num_bytes"]
    # Allocate a large amount of memory
    bytearray(num_bytes)


# Background tasks
KEY_BACKGROUND = "background"
PENDING = object()
TASKS: Dict[str, Any] = {}


def check_task_status(request: Dict) -> Dict:
    task_id = request["task_id"]
    if TASKS[task_id] is PENDING:
        return dict(is_done=False, task_result=None)
    result = TASKS.pop(task_id)
    if isinstance(result, Exception):
        raise result
    # We only allow to get the task result once
    return dict(is_done=True, task_result=result)


# fn_name, is_async
COMMAND_FN_DICT = {
    fn.__name__: (fn, is_async)
    for fn, is_async in [
        (create_session, False),
        (stop_session, False),
        (get_new_id, False),
        (run_session, True),
        (get_exitcode, True),
        (run_command, True),
        (check_task_status, False),
        (is_server_alive, False),
        (_test_oom, False),
    ]
}


# active clients
NUM_ACTIVE_CLIENTS = 0


async def handle_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    global NUM_ACTIVE_CLIENTS
    try:
        NUM_ACTIVE_CLIENTS += 1
        LOGGER.info(f"#Active clients: {NUM_ACTIVE_CLIENTS}")
        await handle_client_loop(reader, writer)
    except Exception:
        LOGGER.exception("Failed to send response to client")
    finally:
        NUM_ACTIVE_CLIENTS -= 1
        writer.close()
        with suppress(ConnectionResetError):
            await writer.wait_closed()
        LOGGER.info(f"Client disconnected: {NUM_ACTIVE_CLIENTS}")


MAX_LOG_LENGTH = 1000


async def handle_client_loop(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """
    Read and send until the connection is closed.
    """
    assert SEMAPHORE is not None, "Semaphore must be initialized before use"
    while True:
        try:
            # We'll read until we get a newline or up to 64KB
            data = await reader.readline()
            if not data:
                return

            request_str = data.decode("utf-8").strip()
            LOGGER.info(f"Received raw request: {request_str}")
            request: Dict = json.loads(request_str)

            action_fn, is_async = COMMAND_FN_DICT[request["action"]]

            # Check if it's a background task
            is_background = request.get(KEY_BACKGROUND) is True

            if is_background:
                # Only async tasks can do background
                assert is_async
                task_id = get_new_id({}, prefix="bg-")
                TASKS[task_id] = PENDING
                LOGGER.info(f"Starting background task: {task_id}")

                async def run_bg_task():
                    nonlocal action_fn
                    nonlocal task_id
                    try:
                        # We limit the concurrency for the action function
                        async with SEMAPHORE:
                            result = await action_fn(request)  # type: ignore # noqa: B023
                        TASKS[task_id] = result  # noqa: B023
                    except Exception as e:
                        LOGGER.exception(f"Exception in background task {task_id}")  # noqa: B023
                        TASKS[task_id] = e  # noqa: B023

                task = asyncio.create_task(run_bg_task())
                task.add_done_callback(
                    lambda _: LOGGER.info(f"Background task {task_id} done")  # noqa: B023
                )
                response = dict(status="ok", result=dict(task_id=task_id, done=False))
            else:
                # We limit the concurrency for the action function
                if is_async:
                    async with SEMAPHORE:
                        result = await action_fn(request)  # type: ignore
                else:
                    # blocking call
                    result = action_fn(request)  # type: ignore
                response = dict(status="ok", result=result)
        except Exception as e:
            # Catch-all for unexpected errors
            trace = traceback.format_exc(limit=6)
            error_msg = f"{type(e)}: {str(e)}\n\n{trace}"
            response = dict(status="error", result=error_msg)
            LOGGER.exception("Error when processing request")

        response_str = json.dumps(response)
        response_bytes = response_str.encode("utf-8") + b"\n"

        # If the response is too long, send an error message instead
        if len(response_bytes) > MAX_BUFFER_BYTES:
            message = f"Response too long: {len(response_bytes)}"
            response = dict(status="error", result=message)
            response_str = json.dumps(response)
            response_bytes = response_str.encode("utf-8") + b"\n"
            LOGGER.error(message)

        response_logstr = response_str[:MAX_LOG_LENGTH]
        LOGGER.info(f"Sending {len(response_bytes)} bytes: {response_logstr}..")
        writer.write(response_bytes)
        await writer.drain()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server to.",
    )
    parser.add_argument(
        "--max_connections",
        type=int,
        default=16,
        help="Port number to bind the server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port number to bind the server to.",
    )
    parser.add_argument(
        "--server_registry",
        type=str,
        required=False,
        default="",
        help="A directory where we write the server address to an empty file.",
    )
    args = parser.parse_args()

    # # Debugging
    # print(os.environ)

    global HOST, PORT, SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(args.max_connections)
    HOST = args.host
    PORT = args.port
    assert HOST is not None and PORT is not None
    assert HOST == "0.0.0.0", "Only support 0.0.0.0 for now"

    server = await asyncio.start_server(
        handle_client,
        host=HOST,
        port=PORT,
        limit=MAX_BUFFER_BYTES,  # 64KB buffer size
    )
    addr = server.sockets[0].getsockname()
    PORT = addr[1]
    server_address = f"{HOST_NAME}:{PORT}"
    LOGGER.info(f"Server listening on {server_address}")

    if args.server_registry:
        server_registry = os.path.abspath(args.server_registry)
        server_path = os.path.join(server_registry, server_address)
        with open(server_path, "w"):
            pass

    try:
        async with server:
            await server.serve_forever()
    finally:
        if args.server_registry:
            print(f"Removing server registry file: {server_path}")
            with suppress(FileNotFoundError):
                os.remove(server_path)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Shutting down server.")
