"""A publication served over HTTP on 127.0.0.1, as a static web server serves one, with the faults
a real one shows now and then: a missing file, a server error, a redirect to a login page, an
answer cut short, an answer that never comes."""

from __future__ import annotations

import http.server
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Server:
    """Where the publication is served, and the fault to answer each path with, by path relative
    to it: ``missing``, ``error``, ``redirect``, ``cut`` (a ``Content-Length`` the body does not
    reach), ``short`` (fewer bytes, no ``Content-Length``) or ``stall`` (no answer for 2 s)."""

    url: str
    faults: dict[str, str] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)


@contextmanager
def served(root: Path) -> Iterator[Server]:
    server: Server

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            relative = self.path.lstrip("/")
            server.requests.append(relative)
            fault = server.faults.get(relative)
            if fault == "stall":
                time.sleep(2)
            if fault == "error":
                self.send_error(503)
                return
            if fault == "redirect":
                self.send_response(302)
                self.send_header("Location", "https://auth.example.org/login")
                self.end_headers()
                return
            path = root / relative
            if fault == "missing" or not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if fault != "short":
                self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[: len(data) // 2] if fault in ("cut", "short") else data)

        def log_message(self, *args: object) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    server = Server(f"http://127.0.0.1:{httpd.server_address[1]}")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        httpd.shutdown()
        httpd.server_close()
