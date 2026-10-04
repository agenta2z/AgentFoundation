"""stdio <-> unix-socket MCP relay (stdlib only), spawned by the Devmate ``dm`` CLI.

dm-core's ``--mcp-servers`` has two transports: ``stdio`` (dm spawns a command and
speaks MCP over its stdin/stdout) and ``socket`` (dm connects to a unix socket).
The socket transport does not complete real-model turns in dm-core 2026.10.02
(the agent loop stalls after MCP warmup — a dm-core bug; see
scripts/native_spikes/s14), but the stdio transport works. AF tools run in the
inferencer process, so a dm-spawned subprocess cannot host them directly — it
must relay back to the in-process ``LocalMcpSocketServer``.

Both dm's stdio MCP framing and that server's framing are the same newline-
delimited JSON-RPC, so this relay is a pure bidirectional byte forwarder: dm's
stdin -> the unix socket, and the unix socket -> dm's stdout. No parsing, no
third-party deps (so dm can spawn it with any ``python3``).

Usage (dm spawns it):  python3 mcp_stdio_relay.py <unix_socket_path>
"""

from __future__ import annotations

import os
import socket
import sys
import threading


def _pump_stdin_to_socket(sock: "socket.socket") -> None:
    try:
        while True:
            data = os.read(0, 65536)
            if not data:
                break
            sock.sendall(data)
    except OSError:
        pass
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _pump_socket_to_stdout(sock: "socket.socket") -> None:
    try:
        while True:
            data = sock.recv(65536)
            if not data:
                break
            os.write(1, data)
    except OSError:
        pass


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: mcp_stdio_relay.py <unix_socket_path>\n")
        return 2
    path = sys.argv[1]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except OSError as exc:
        sys.stderr.write(f"relay: cannot connect to {path}: {exc}\n")
        return 1
    up = threading.Thread(target=_pump_stdin_to_socket, args=(sock,), daemon=True)
    up.start()
    try:
        _pump_socket_to_stdout(sock)
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
