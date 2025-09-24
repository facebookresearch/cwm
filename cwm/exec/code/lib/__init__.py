# Copyright (c) Meta Platforms, Inc. and affiliates.

import marshal
import sys
import typing as tp
from multiprocessing.connection import Connection
from pathlib import Path

from .server import ForkServer, JSONConnection

T = tp.TypeVar("T")


class _Process:
    def __init__(
        self,
        fork_server: ForkServer,
        vpid: int,
        input_w: Connection,
        output_r: JSONConnection,
    ) -> None:
        self.fork_server = fork_server
        self.vpid = vpid
        self.input_w = input_w
        self.output_r = output_r
        self.closed = False

    def send(self, obj: tp.Any) -> None:
        assert not self.closed
        self.input_w.send(obj)

    def poll(self, timeout: float) -> bool:
        assert not self.closed
        return self.output_r.poll(timeout)

    def recv(self) -> tp.Any:
        assert not self.closed
        return self.output_r.recv()

    def shutdown(self) -> None:
        if self.closed:
            return

        self.closed = True
        self.input_w.close()
        self.output_r.close()
        self.fork_server.kill(self.vpid)


class Runner:
    """
    A sandboxed runner convenient to execute arbitrary
    Python code in a sandboxed environment. The code to
    execute is passed directly as a callable and executed
    in a fresh sandboxed environment. Its arguments
    are pickled to be fed in the sandboxed environment,
    and the return value is serialized using the json
    module. Note that the use of json puts a rather severe
    constraint on the output type of the function to be run.

    The code to run cannot modify or access globals of
    the launching process. Nonetheless, it is possible
    to import modules from lib's parent package.
    """

    class Error(Exception):
        """
        An error happened during execution of the
        sandboxed code.
        """

    class RaisedError(Error):
        "The sandboxed code raised an exception."

    class TimeoutError(Error):
        "The sandboxed code exceeded its time quota."

    def __init__(
        self,
        fork_server: ForkServer | None = None,
        timeout: float | None = None,
        reuse_sandbox: bool = False,
        fork_in_worker: bool = True,
        **spawn_args,
    ):
        """
        Initialize a new runner object.

        Arguments:
            fork_server (ForkServer, optional): the fork server
                used to execute code when ``run()`` is called.
            timeout (float, optional): timeout for one run call
                in seconds; if unspecified, no timeout is used.
            reuse_sandbox (bool, optional): whether the same
                sandbox must be reused for multiple consecutive
                ``run`` calls.
            fork_in_worker (bool, optional): when ``reuse_sandbox``
                is set, this flag controls whether the sandboxed
                process executes unsafe code in a new process;
                doing so provides better isolation across runs.
                This flag is set by default.
            **spawn_args: more arguments, such as resource limits,
                that will be forwarded to ``ForkServer.spawn``.
        """
        if fork_server is not None:
            self.fork_server = fork_server
        else:
            self.fork_server = ForkServer.global_instance()
        self.timeout = timeout
        self.spawn_args = spawn_args
        self.fork_in_worker = fork_in_worker if reuse_sandbox else False

        self.process: _Process | None = None
        if reuse_sandbox:
            self._spawn()

    def _spawn(self) -> None:
        runner_py = str(Path(__file__).parent / "runner.py")
        root_dir = str(Path(__file__).parents[2])
        cmd = [sys.executable, runner_py, "--root_dir", root_dir]
        if self.fork_in_worker:
            cmd.append("--fork_before_work")
        vpid, input_w, output_r = self.fork_server.spawn(
            cmd=cmd,
            **self.spawn_args,
        )
        self.process = _Process(self.fork_server, vpid, input_w, output_r)

    def run(self, fn: tp.Callable[..., T], /, *args, **kwargs) -> T:
        """
        Call ``fn(*args, **kwargs)`` in a sandbox and return
        the computed value.

        Note:
            Side-effects of ``fn`` such as arguments mutations
            or changes to global variables won't be reflected
            in the caller.
        """
        if fn.__closure__ is not None:
            raise ValueError(f"cannot marshal closure argument {fn!r}")

        spawned = False
        cleanup = True
        if self.process is None:
            spawned = True
            self._spawn()

        assert self.process is not None
        try:
            code = marshal.dumps(fn.__code__)
            self.process.send(
                {
                    "code": code,
                    "args": args,
                    "kwargs": kwargs,
                }
            )

            if self.timeout is not None:
                if not self.process.poll(timeout=self.timeout):
                    raise Runner.TimeoutError()

            res = self.process.recv()
            if "return" in res:
                cleanup = spawned
                return res["return"]
            if "except" in res:
                cleanup = spawned
                raise Runner.RaisedError(res)

            raise RuntimeError(res)

        finally:
            if cleanup:
                self.process.shutdown()
                self.process = None
                if not spawned:
                    self._spawn()


_runner: Runner | None = None


def run(fn: tp.Callable[..., T], *args, **kwargs) -> T:
    """
    Call ``fn(*args, **kwargs)`` in a sandbox and return
    the computed value.

    See ``Runner.run``.
    """
    global _runner
    if _runner is None:
        _runner = Runner()
    return _runner.run(fn, *args, **kwargs)
