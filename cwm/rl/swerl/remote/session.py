# Copyright (c) Meta Platforms, Inc. and affiliates.

# ruff: noqa
import asyncio
import subprocess
from typing import Optional, List
import sys

if sys.version_info >= (3, 9):
    from typing import TypedDict, Literal  # Python 3.9+
else:
    from typing_extensions import TypedDict, Literal


class SessionOutput(TypedDict):
    status: Literal["success", "error"]
    output: str
    error_type: Literal["none", "timeout", "too_long", "exit", "broken_pipe", "other"]


class _BufferTooLongError(Exception):
    """For internal use only."""


class _ProgramExitError(Exception):
    """For internal use only."""


class AsyncSession:
    """
    An async, persistent bash-like session. The `communicate` API allows receiving outputs from the session
    given a command. This is also the only supposed way to interact with the session.

    Example:
    ```python
    session = Session(["bash"], timeout=5)
    output = session.communicate("pwd && echo 'hello'")
    print(output)
    ```
    """

    def __init__(
        self,
        command_args: List[str],
        timeout: float,
        # Will always be invoked after the session starts
        start_script: Optional[str] = None,
        # Some defaults that generally don't need to be changed
        sentinel: str = "<<exit>1",
        # Sentinel command's output should match the sentinel
        # making them different can avoid some false matches
        sentinel_command: str = 'echo "<<exit>$((1))"',
        # The sentinel command will make echo $? no effect
        # as the exit code is always 0
        # So we always save the exit code in a variable
        exit_code_var: str = "__EXIT_CODE__",
        chunk_size: int = 4096,
        empty_read_delay: float = 0.2,
        # A feature of the session that truncates too long outputs
        # limit to 480KB (~96k tokens)
        max_buffer_size: int = 480 * 1024,
    ):
        self.command_args = command_args
        self.start_script = start_script
        self.sentinel = sentinel
        self.sentinel_command = sentinel_command
        self.exit_code_var = exit_code_var
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.empty_read_delay = empty_read_delay
        self.proc: Optional[asyncio.subprocess.Process] = None
        # self.selector: selectors.Optional[DefaultSelector] = None
        self._tmp_buffer = b""
        self.max_buffer_size = max_buffer_size

    async def ensure_started(self) -> None:
        if self.proc is None:
            await self.start()
        assert (
            self.proc is not None
        ), "[ensure_started] process None, should not happen."

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            *self.command_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # text=True,
            bufsize=0,
        )
        if self.start_script is not None:
            # NOTE(yuxiang): it's actually better to raise some error if start script fails;
            # now it just silently fails
            await self._communicate(self.start_script, sanitize=False)

    async def restart(self) -> None:
        self.stop()
        await self.start()

    async def _send(self, command: str):
        """
        Sends the user's multiline command to the given bash process,
        then appends a sentinel echo so we know when the command is done.

        NOTE: `_send` twice will lead to undefined behavior. The only interaction API is `communicate`.
        """
        assert self.proc is not None, "[_send] process None, should not happen."
        command = command.rstrip() + "\n"
        exitcode_command = f"{self.exit_code_var}=$?\n"
        sentinel_command = self.sentinel_command.rstrip() + "\n"
        command = command + exitcode_command + sentinel_command
        assert self.proc.stdin
        self.proc.stdin.write(command.encode())
        await self.proc.stdin.drain()

    async def _read(self) -> str:
        """
        Reads output from the process until the sentinel is found.
        """
        assert self.proc is not None, "[_read] process None, should not happen."
        assert self.proc.stdout
        # We want to store some partial data here
        self._tmp_buffer = b""
        while True:
            data = await self.proc.stdout.read(self.chunk_size)
            if data:
                self._tmp_buffer += data
            else:
                # Early return if the process has already finisehd.
                if self.proc.returncode is not None:
                    raise _ProgramExitError
                # We must have this to avoid hanging when "exit" is called,
                # which makes self.proc.stdout.read synchronized
                await asyncio.sleep(self.empty_read_delay)
            if len(self._tmp_buffer) > self.max_buffer_size:
                raise _BufferTooLongError
            sentinel_index = self._tmp_buffer.rfind(self.sentinel.encode())
            if sentinel_index != -1:
                output = self._tmp_buffer[:sentinel_index].decode(errors="ignore")
                return output

    async def get_exitcode(self) -> Optional[int]:
        """
        Get the exit code of the last command. If failed, return None
        """
        output = await self.communicate(f"echo ${self.exit_code_var}")
        if output["status"] == "error":
            return None
        try:
            return int(output["output"])
        except (ValueError, TypeError, OverflowError):
            return None

    async def communicate(self, command: str, sanitize: bool = True) -> SessionOutput:
        """
        Sends a command to the process and reads the output until the command is done.
        """
        await self.ensure_started()
        assert self.proc is not None, "[communicate] process None, should not happen."
        output = await self._communicate(command, sanitize)
        if output["error_type"] != "none":
            # We force stop the session to avoid undefined behavior
            self.stop()
        return output

    async def _communicate(self, command: str, sanitize: bool) -> SessionOutput:
        """Internal implementation. Just runs the command and returns the output."""

        def sanitize_fn(output: str) -> str:
            if output.endswith("\n"):
                output = output[:-1]
            return output

        try:
            await self._send(command)
            read_output = await asyncio.wait_for(self._read(), timeout=self.timeout)
            if sanitize:
                read_output = sanitize_fn(read_output)
            output = SessionOutput(
                status="success", output=read_output, error_type="none"
            )
        except TimeoutError:
            read_output = self._tmp_buffer.decode(errors="ignore")
            assert self.proc is not None
            if self.proc.returncode is not None:
                error_type = "exit"
            else:
                error_type = "timeout"
            output = SessionOutput(
                status="error",
                output=read_output,
                error_type=error_type,  # type: ignore
            )
        except _ProgramExitError:
            read_output = self._tmp_buffer.decode(errors="ignore")
            assert self.proc is not None
            assert self.proc.returncode is not None
            output = SessionOutput(
                status="error",
                output=read_output,
                error_type="exit",
            )
        except _BufferTooLongError:
            read_output = self._tmp_buffer.decode(errors="ignore")
            output = SessionOutput(
                status="error",
                output=read_output,
                error_type="too_long",
            )
        except BrokenPipeError as e:
            output = SessionOutput(
                status="error", output=str(e), error_type="broken_pipe"
            )
        except Exception as e:
            output = SessionOutput(
                status="error", output=f"{type(e)}: {str(e)}", error_type="other"
            )
        return output

    def __str__(self):
        return (
            f"Session(command_args={self.command_args}, active={self.proc is not None})"
        )

    def __repr__(self):
        return str(self)

    def stop(self) -> None:
        if self.proc is not None:
            if self.proc.returncode is None:
                self.proc.terminate()
            # not waiting to avoid hanging
            # self.proc.wait()
            self.proc = None
            self.selector = None

    def __del__(self):
        self.stop()
