# Copyright (c) Meta Platforms, Inc. and affiliates.

import atexit
import json
import logging
import multiprocessing
import os
import random
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from getpass import getuser
from multiprocessing.connection import Connection
from os import path as osp
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

from filelock import FileLock

logger = logging.getLogger()

_fork_server: Optional["ForkServer"] = None
_fork_server_lock = threading.Lock()


class Executor(Enum):
    DEFAULT = -1
    BUBBLEWRAP = 0
    FORK = 1
    PERSISTENT_BUBBLEWRAP = 2


@dataclass
class ResourceLimits:
    # Safe defaults
    memory: int | None = int(2e9)
    tmpfs_size: int | None = int(1e9)
    cpu_time: int | None = None


class JSONConnection:
    """
    Simple Connection wrapper, but send() and recv() use JSON serialization to
    prevent side-effects when unpickling.
    """

    def __init__(self, c: Connection):
        self._c = c

    def send(self, obj: Any) -> None:
        self._c.send_bytes(json.dumps(obj).encode("utf8"))

    def recv(self) -> Any:
        # Putting some arbitrary limit here to prevent DoS.
        return json.loads(self._c.recv_bytes(100 * 1024 * 1024))

    def poll(self, timeout: float = 0.0) -> bool:
        return self._c.poll(timeout)

    def close(self) -> None:
        self._c.close()

    def fileno(self) -> int:
        return self._c.fileno()

    @property
    def closed(self) -> bool:
        return self._c.closed

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


class ForkServer:
    """
    Fork server for sandboxed program execution.

    Use spawn() to execute a program, specified by a command-line. It will
    return a handle and two file descriptors for sending input and obtaining
    output from the command, respectively. Commands need to be tailored to the
    input/output format expected by the call site.
    The command-line will be extended by two position arguments, corresponding
    to a numeric file descriptor to read input from and a numeric file
    descriptor to write output to.
    """

    def __init__(self, default_executor: Executor = Executor.BUBBLEWRAP):
        ctx = multiprocessing.get_context("spawn")
        read_pipe, write_pipe = ctx.Pipe(duplex=False)

        self.process = ctx.Process(
            target=_run,
            args=(
                os.getpid(),
                read_pipe,
            ),
        )
        self.process.start()
        read_pipe.close()
        self.default_executor = default_executor
        self.lock = threading.Lock()
        self.next_vpid = 1  # process handles
        self.pipe = write_pipe
        self.rng = random.Random(int.from_bytes(os.urandom(4), sys.byteorder))

    @staticmethod
    def global_instance() -> "ForkServer":
        global _fork_server
        if _fork_server is None:  # check without lock first
            with _fork_server_lock:
                if _fork_server is None:
                    fs = ForkServer()
                    atexit.register(lambda: fs.stop())
                    _fork_server = fs
        return _fork_server

    def spawn(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        executor: Executor = Executor.DEFAULT,
        rlimits: ResourceLimits | None = None,
        timeout: int = 30,
        retries: int = 10,
    ) -> tuple[int, Connection, JSONConnection]:
        """
        Launches a sandboxed process.

        If the process fails to start or does not send a canary message within
        the specified timeout, retries will be attempted as specified.

        Returns a vPID that can be passed to ForkServer.kill() as well as
        multiprocessing.Connection objects to communicate with the sandboxed
        process (input writing and output reading ends).
        """
        if executor == Executor.DEFAULT:
            executor = self.default_executor

        if rlimits is None:
            rlimits = ResourceLimits()
            if executor == Executor.FORK:
                # FORK executor does not support file size limits; disable them
                # to suppress warnings during execution.
                rlimits.tmpfs_size = None

        for retry in range(max(retries + 1, 1)):
            if retry > 0:
                # exponential backoff + jitter
                t = 0.1 * (1.5 ** (retry + self.rng.random()))
                time.sleep(t)

            input_r, input_w = multiprocessing.Pipe(duplex=False)
            output_r, output_w = multiprocessing.Pipe(duplex=False)

            if env is None:
                env = dict(os.environ)

            with self.lock:
                vpid = self.next_vpid
                self.next_vpid += 1
                self.pipe.send(
                    (vpid, "spawn", (executor, cmd, env, input_r, output_w, rlimits))
                )
            # Read output from JSONConnection to avoid unpickling from an untrusted
            # process.
            output_rj = JSONConnection(output_r)

            input_r.close()
            output_w.close()

            # Wait for canary
            failure_msg = (
                f"retrying {retry + 1}/{retries}" if retry < retries else "giving up"
            )
            if not output_r.poll(timeout=timeout):
                logger.warning(f"Timeout waiting for canary message, {failure_msg}")
                self.kill(vpid)
                continue

            res = output_rj.recv()
            if "canary" not in res or res["canary"] != "chirp":
                logger.warning(f"Unexpected canary message: '{res}', {failure_msg}")
                self.kill(vpid)
                continue

            return vpid, input_w, output_rj

        raise RuntimeError("Failed to start sandboxed process; check your logs")

    def kill(self, vpid: int) -> None:
        with self.lock:
            self.pipe.send((vpid, "kill", None))

    def stop(self) -> None:
        with self.lock:
            self.pipe.send((0, "exit", None))


def killpid(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        traceback.print_exc()
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        traceback.print_exc()
    try:
        os.waitpid(pid, 0)
    except OSError:
        traceback.print_exc()


@dataclass
class BwrapForkServer:
    pid: int
    sock: socket.socket
    rlimits: ResourceLimits
    busy_r: Connection | None = None
    num_done: int = 0


def _sanitize_env(env: dict[str, str]):
    # LD_PRELOAD messes with the code_contests sandbox
    if "LD_PRELOAD" in env:
        del env["LD_PRELOAD"]


def _run(parent_pid: int, cmd_pipe: Connection):
    # logging levels specific to the execution server
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logger.setLevel(logging.WARNING)

    logger.info("Execution server: process started")
    rng = random.Random(int.from_bytes(os.urandom(4), sys.byteorder))
    try:
        pids: dict[int, int | tuple[int, BwrapForkServer]] = {}  # vpid -> pid
        bwrapforkservers: list[BwrapForkServer] = []
        while True:
            if not cmd_pipe.poll(timeout=0.5):
                try:
                    # Parent still alive?
                    os.kill(parent_pid, 0)
                except OSError:
                    break
                continue
            vpid, command, extra = cmd_pipe.recv()
            if command == "spawn":
                executor, cmd, env, input_r, output_w, rlimits = extra
                _sanitize_env(env)

                busy_r = None
                busy_w = None
                bwrapforkserver: BwrapForkServer | None = None

                if executor == Executor.PERSISTENT_BUBBLEWRAP:
                    for v in bwrapforkservers:
                        if v.busy_r is None and v.rlimits == rlimits:
                            bwrapforkserver = v
                    if bwrapforkserver is None:
                        bwrapforkserver_socket, remote_socket = socket.socketpair()
                        bwrapforkserver_pid = os.fork()
                        if bwrapforkserver_pid == 0:
                            try:
                                for v in bwrapforkservers:
                                    if v.busy_r is not None:
                                        v.busy_r.close()
                                    v.sock.close()
                                input_r.close()
                                output_w.close()
                                cmd_pipe.close()
                                bwrapforkserver_socket.close()
                                execute_bwrap(
                                    [
                                        sys.executable,
                                        osp.join(
                                            Path(__file__).parent, "bwrapforkserver.py"
                                        ),
                                    ],
                                    env,
                                    remote_socket,
                                    remote_socket,
                                    rlimits,
                                    seed=rng.randint(0, 10000),
                                    send_done=False,
                                )
                            except Exception:
                                traceback.print_exc()
                            finally:
                                os._exit(0)
                        else:
                            remote_socket.close()
                        os.setpgid(bwrapforkserver_pid, bwrapforkserver_pid)
                        bwrapforkserver = BwrapForkServer(
                            pid=bwrapforkserver_pid,
                            sock=bwrapforkserver_socket,
                            rlimits=rlimits,
                        )
                        bwrapforkservers.append(bwrapforkserver)
                    busy_r, busy_w = multiprocessing.Pipe(duplex=False)

                    bwrapforkserver.busy_r = busy_r

                child_pid = os.fork()
                if child_pid == 0:
                    try:
                        mypid = os.getpid()
                        os.setpgid(mypid, mypid)

                        for v in bwrapforkservers:
                            if v is bwrapforkserver:
                                continue
                            if v.busy_r is not None:
                                v.busy_r.close()
                            v.sock.close()
                        del bwrapforkservers

                        if bwrapforkserver is not None:
                            assert executor == Executor.PERSISTENT_BUBBLEWRAP
                            assert busy_r is not None
                            busy_r.close()
                            sock = bwrapforkserver.sock
                            del bwrapforkserver

                        pids = {}
                        match executor:
                            case Executor.BUBBLEWRAP:
                                execute_bwrap(
                                    cmd,
                                    env,
                                    input_r,
                                    output_w,
                                    rlimits,
                                    seed=rng.randint(0, 10000),
                                )
                            case Executor.FORK:
                                execute_fork(
                                    cmd,
                                    env,
                                    input_r,
                                    output_w,
                                    rlimits,
                                )
                            case Executor.PERSISTENT_BUBBLEWRAP:
                                assert busy_w is not None
                                execute_bwrapforkserver(
                                    sock, cmd, input_r, output_w, busy_w
                                )
                            case _:
                                logger.error(f"Unknown executor: {executor}")
                                output_w.send_bytes(
                                    json.dumps({"error": "unknown executor"}).encode(
                                        "utf8"
                                    )
                                )

                        if busy_w is not None:
                            busy_w.close()
                    except Exception:
                        traceback.print_exc()
                    finally:
                        os._exit(0)

                if busy_w is not None:
                    busy_w.close()
                input_r.close()
                output_w.close()

                assert vpid not in pids
                os.setpgid(child_pid, child_pid)
                if bwrapforkserver is not None:
                    pids[vpid] = (child_pid, bwrapforkserver)
                else:
                    pids[vpid] = child_pid
            elif command == "kill":
                if vpid in pids:
                    bwrapforkserver = None
                    pid = pids[vpid]
                    if isinstance(pid, tuple):
                        pid, bwrapforkserver = pid
                        assert bwrapforkserver is not None
                        assert bwrapforkserver.busy_r is not None
                        if bwrapforkserver.busy_r.poll():
                            try:
                                if bwrapforkserver.busy_r.recv_bytes(1) == b"1":
                                    bwrapforkserver.num_done += 1
                                    if bwrapforkserver.num_done < 20:
                                        bwrapforkserver.busy_r = None
                                        bwrapforkserver = None
                            except EOFError:
                                pass
                    del pids[vpid]
                    killpid(pid)
                    if bwrapforkserver is not None:
                        killpid(bwrapforkserver.pid)
                        bwrapforkservers = list(
                            v for v in bwrapforkservers if v is not bwrapforkserver
                        )
            elif command == "exit":
                break
            else:
                raise RuntimeError(f"Execution server: unknown command {command}")
    except EOFError:
        logger.error("Execution server: main process died!")
        traceback.print_exc()
    finally:
        for pid in pids.values():
            if isinstance(pid, tuple):
                pid, bwrapforkserver = pid
                if bwrapforkserver in bwrapforkservers:
                    killpid(bwrapforkserver.pid)
                    bwrapforkservers = list(
                        v for v in bwrapforkservers if v is not bwrapforkserver
                    )
            killpid(pid)
        for v in bwrapforkservers:
            killpid(v.pid)


def execute_bwrap(
    cmd: list[str],
    env: dict[str, str],
    input_r: Connection | socket.socket,
    output_w: Connection | socket.socket,
    rlimits: ResourceLimits,
    seed: int,
    send_done: bool = True,
) -> None:
    def set_rlimit():
        if rlimits.memory is not None:
            resource.setrlimit(resource.RLIMIT_AS, (rlimits.memory, rlimits.memory))
            resource.setrlimit(resource.RLIMIT_DATA, (rlimits.memory, rlimits.memory))
        if rlimits.cpu_time is not None:
            resource.setrlimit(
                resource.RLIMIT_CPU, (rlimits.cpu_time, rlimits.cpu_time)
            )

    status_r, status_w = multiprocessing.Pipe(duplex=False)

    args = [
        "bwrap",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--new-session",
        "--unshare-all",
        "--cap-drop",
        "ALL",
    ]
    if rlimits.tmpfs_size is not None:
        args += ["--size", str(rlimits.tmpfs_size)]
    args += [
        "--tmpfs",
        "/tmp",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--dir",
        "/tmp/sandbox",
        "--chdir",
        "/tmp/sandbox",
        "--info-fd",
        str(status_w.fileno()),
    ]

    args += [
        *cmd,
        str(input_r.fileno()),
        str(output_w.fileno()),
    ]

    # Under heavy system load (kernel load, not CPU load), bubblewrap can take a
    # long time to start. When launching many sandboxes in parallel, bubblewrap
    # itself is a main cause of high load, unfortunately. Here, we attempt to
    # linearlize sandboxes starting up in order to keep overall startup times
    # low. We do this at the machine level with lock files and will warn the
    # user if startup times are very high.
    # We can isolate sandbox creation time with `--info-fd`: bubblewrap will
    # write to it after launching the sandboxed process.
    lockfile = osp.join(os.environ.get("TMPDIR", "/dev/shm"), f"bwrap.lock.{getuser()}")
    # Allow a handful concurrent sandbox launches
    n_locks = 4
    random.seed(seed)

    def get_lock():
        locks = [FileLock(lockfile + f".{i}") for i in range(n_locks)]
        while True:
            lock = random.choice(locks)
            try:
                return lock.acquire(timeout=0.01)
            except BaseException:
                pass

    start = time.perf_counter()
    with get_lock():
        elapsed = time.perf_counter() - start
        if elapsed > 10:
            logger.warning(f"Waited {elapsed:.02f}s for bubblewrap lock")
        process = subprocess.Popen(
            args,
            close_fds=True,
            pass_fds=(input_r.fileno(), output_w.fileno(), status_w.fileno()),
            stdin=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            env=env,
            preexec_fn=set_rlimit,
        )

        # Hold the lock for up to one second while bubblerwap is starting up,
        # hopefully lessening the load on the kernel.
        start = time.perf_counter()
        r = status_r.poll(1)

    if not r:
        # Keep waiting for bubblewrap to start and report if it's taking a very
        # long time.
        r = status_r.poll()
        elapsed = time.perf_counter() - start
        if elapsed > 10:
            logger.warning(f"bubblewrap took {elapsed:.02f}s to start")

    try:
        stdout, stderr = process.communicate()
    except BaseException:
        process.kill()
        process.wait()
        raise

    returncode = process.poll()
    try:
        if send_done:
            assert isinstance(output_w, Connection)
            output_w.send_bytes(
                json.dumps(
                    {
                        "done": True,
                        "returncode": returncode,
                        "stderr": stderr.decode(errors="replace"),
                    }
                ).encode("utf8")
            )
    except BrokenPipeError:
        # Most likely due to premature aborts (the caller stopped listening);
        # ignore this
        pass
    except Exception:
        logger.exception("Error sending 'done' message")


def execute_fork(
    cmd: list[str],
    env: dict[str, str],
    input_r: Connection,
    output_w: Connection,
    rlimits: ResourceLimits,
) -> None:
    def set_rlimit():
        if rlimits.memory is not None:
            resource.setrlimit(resource.RLIMIT_AS, (rlimits.memory, rlimits.memory))
            resource.setrlimit(resource.RLIMIT_DATA, (rlimits.memory, rlimits.memory))
        if rlimits.cpu_time is not None:
            resource.setrlimit(
                resource.RLIMIT_CPU, (rlimits.cpu_time, rlimits.cpu_time)
            )

    status_r, status_w = multiprocessing.Pipe(duplex=False)

    if rlimits.tmpfs_size is not None:
        logger.warning("tempfs size cannot be enforced")

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = osp.join(tmpdir, "run")
        os.mkdir(cwd)
        tmpdir = osp.join(tmpdir, "tmp")
        os.mkdir(tmpdir)
        env["TMPDIR"] = tmpdir
        env["TEMP"] = tmpdir
        env["TMP"] = tmpdir

        process = subprocess.Popen(
            [*cmd, str(input_r.fileno()), str(output_w.fileno())],
            close_fds=True,
            pass_fds=(input_r.fileno(), output_w.fileno(), status_w.fileno()),
            stdin=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            env=env,
            preexec_fn=set_rlimit,
            cwd=cwd,
        )

        try:
            stdout, stderr = process.communicate()
        except BaseException:
            process.kill()
            process.wait()
            raise

        returncode = process.poll()
        try:
            output_w.send_bytes(
                json.dumps(
                    {
                        "done": True,
                        "returncode": returncode,
                        "stderr": stderr.decode(errors="replace"),
                    }
                ).encode("utf8")
            )
        except BrokenPipeError:
            # Most likely due to premature aborts (the caller stopped listening);
            # ignore this
            pass
        except Exception:
            logger.exception("Error sending 'done' message")


def execute_bwrapforkserver(
    sock: socket.socket,
    cmd: list[str],
    input_r: Connection,
    output_w: Connection,
    busy_r: Connection,
) -> None:
    s = json.dumps({"cmd": cmd}).encode()
    sock.send(len(s).to_bytes(4, "little", signed=False))
    sock.send(s)
    socket.send_fds(
        sock,
        [b"the cow goes moo"],
        [input_r.fileno(), output_w.fileno()],
    )
    input_r.close()

    def recvn(sock: socket.socket, n: int) -> bytearray:
        buf = bytearray()
        while len(buf) != n:
            d = sock.recv(n - len(buf))
            if len(d) == 0:
                raise RuntimeError("socket closed")
            buf.extend(d)
        return buf

    def recvmsg(sock: socket.socket):
        len = int.from_bytes(recvn(sock, 4), "little", signed=False)
        assert len < 65536
        return recvn(sock, len)

    response = json.loads(recvmsg(sock).decode())
    assert response["the sheep goes"] == "baa"

    # print(response)

    try:
        busy_r.send_bytes(b"1")
        output_w.send_bytes(
            json.dumps(
                {
                    "done": True,
                    "returncode": response["returncode"],
                    "stderr": response["stderr"],
                }
            ).encode("utf8")
        )
    except BrokenPipeError:
        # Most likely due to premature aborts (the caller stopped listening);
        # ignore this
        pass
    except Exception:
        logger.exception("Error sending 'done' message")
