# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import socket
import subprocess
import sys


def recvn(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray()
    while len(buf) != n:
        d = sock.recv(n - len(buf))
        if len(d) == 0:
            raise RuntimeError("socket closed")
        buf.extend(d)
    return buf


def recvmsg(sock: socket.socket) -> bytearray:
    len = int.from_bytes(recvn(sock, 4), "little", signed=False)
    assert len < 65536
    return recvn(sock, len)


if __name__ == "__main__":
    sock = socket.socket(socket.AF_UNIX, fileno=int(sys.argv[1]))

    while True:
        msg = json.loads(recvmsg(sock))
        message, fds, _, _ = socket.recv_fds(sock, 0x1000, 2)
        assert message.decode() == "the cow goes moo"

        cmd = msg["cmd"]
        input_r, output_w = fds

        process = subprocess.Popen(
            [*cmd, str(input_r), str(output_w)],
            close_fds=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(input_r, output_w),
        )

        try:
            stdout, stderr = process.communicate()
        except BaseException:
            process.kill()
            process.wait()
            raise

        returncode = process.poll()

        s = json.dumps(
            {
                "the sheep goes": "baa",
                "returncode": returncode,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
            }
        ).encode()
        sock.send(len(s).to_bytes(4, "little", signed=False))
        sock.send(s)

        os.close(input_r)
        os.close(output_w)
