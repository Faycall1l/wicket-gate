"""Stateful session-counter with Redis staging (Orchra export/import pattern).

Endpoints: GET /inc (+1), GET /value, POST /export (counter -> Redis key),
GET /healthz. On boot with IMPORT_ON_START=1, imports Redis key if present.
Redis client is stdlib-only (minimal RESP over socket) — no dependencies.
"""

from __future__ import annotations

import os
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer

REDIS_HOST = os.getenv("REDIS_HOST", "redis-anchor")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
STATE_KEY = os.getenv("STATE_KEY", "stateful:counter")
IMPORT_ON_START = os.getenv("IMPORT_ON_START", "0") == "1"

counter = 0


def _resp(*args: str) -> bytes:
    out = f"*{len(args)}\r\n".encode()
    for a in args:
        b = a.encode()
        out += f"${len(b)}\r\n".encode() + b + b"\r\n"
    return out


def _redis(*args: str) -> bytes:
    s = socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=5)
    try:
        s.sendall(_resp(*args))
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            if buf.endswith(b"\r\n"):
                break
        return buf
    finally:
        s.close()


def _parse_int(reply: bytes) -> int | None:
    # Handles $-1 (nil), $len\r\n<digits>, :<int>
    if reply.startswith(b"$-1"):
        return None
    if reply[:1] == b"$":
        _, _, rest = reply.partition(b"\r\n")
        num, _, _ = rest.partition(b"\r\n")
        try:
            return int(num)
        except ValueError:
            return None
    if reply[:1] == b":":
        try:
            return int(reply[1:].split(b"\r\n")[0])
        except ValueError:
            return None
    return None


def export_state() -> bool:
    return _redis("SET", STATE_KEY, str(counter)).startswith(b"+OK")


def import_state() -> bool:
    global counter
    v = _parse_int(_redis("GET", STATE_KEY))
    if v is not None:
        counter = v
        return True
    return False


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: str) -> None:
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        global counter
        me = os.getenv("HOSTNAME", "?")
        if self.path == "/inc":
            counter += 1
            self._send(200, f"{counter}@{me}")
        elif self.path == "/value":
            self._send(200, f"{counter}@{me}")
        elif self.path == "/healthz":
            self._send(200, "ok")
        else:
            self._send(404, "nf")

    def do_POST(self):
        if self.path == "/export":
            self._send(200 if export_state() else 500, str(counter))
        elif self.path == "/import":
            self._send(200 if import_state() else 500, str(counter))
        else:
            self._send(404, "nf")


if __name__ == "__main__":
    if IMPORT_ON_START:
        try:
            print(f"import: {import_state()} counter={counter}", flush=True)
        except Exception as e:
            print(f"import failed (fresh start): {e}", flush=True)
    HTTPServer(("0.0.0.0", 8080), H).serve_forever()
