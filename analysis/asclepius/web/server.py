"""Serve the web app on this computer only: `uv run asclepius-app`, then use the page it opens.

The page (`index.html`) talks to a small JSON API over `asclepius.web.workspace.Workspace`:

| Request | Does |
|---|---|
| `POST /api/clear` | forget the files picked |
| `PUT /api/files/<role>?name=<file name>` | add one picked file, its bytes as the body; role `reports`, `metadata` or `export` |
| `POST /api/load` | load the files; the review, or `{"error": ...}` with status 400 |
| `POST /api/run` | start the run, `{"group_a": ..., "group_b": ...}` for lab reports |
| `GET /api/status` | the run's state and each step's |

The server listens on 127.0.0.1 and refuses requests naming another host or coming from another
site's page, so no other web page open in the browser can read the data or start a run. Picked
files are kept in a temporary folder that is deleted when the app stops.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from asclepius.run import Settings
from asclepius.web.workspace import ROLES, LoadError, RunError, Workspace

PAGE = Path(__file__).parent / "index.html"
DEFAULT_PORT = 8765
LOCAL_HOSTS = ("127.0.0.1", "localhost")


class App(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, workspace: Workspace, settings: Settings | None) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.workspace = workspace
        self.settings = settings
        self.lock = threading.Lock()  # one load or upload at a time


class Handler(BaseHTTPRequestHandler):
    server: App

    def log_message(self, format: str, *args: object) -> None:  # keep the terminal quiet
        pass

    # ------------------------------------------------------------------ requests

    def do_GET(self) -> None:
        if not self._allowed():
            return
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(HTTPStatus.OK, self.server.workspace.status())
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_PUT(self) -> None:
        if not self._allowed():
            return
        url = urlsplit(self.path)
        parts = url.path.strip("/").split("/")
        if len(parts) != 3 or parts[:2] != ["api", "files"] or parts[2] not in ROLES:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        name = parse_qs(url.query).get("name", [""])[0]
        content = self._body()
        try:
            with self.server.lock:
                self.server.workspace.add_file(parts[2], name, content)
        except LoadError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        self._send(HTTPStatus.NO_CONTENT, b"", "text/plain")

    def do_POST(self) -> None:
        if not self._allowed():
            return
        path = urlsplit(self.path).path
        workspace = self.server.workspace
        body = self._body()
        try:
            if path == "/api/clear":
                with self.server.lock:
                    if workspace.running():
                        raise LoadError("A run is in progress; wait for it to finish.")
                    workspace.clear()
                self._json(HTTPStatus.OK, {})
            elif path == "/api/load":
                with self.server.lock:
                    review = workspace.load()
                self._json(HTTPStatus.OK, review)
            elif path == "/api/run":
                choice = json.loads(body or b"{}")
                if not isinstance(choice, dict):
                    raise RunError("The run request must name the groups as a JSON object.")
                workspace.start_run(
                    choice.get("group_a"), choice.get("group_b"), settings=self.server.settings
                )
                self._json(HTTPStatus.ACCEPTED, workspace.status())
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        except (LoadError, RunError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except json.JSONDecodeError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "The request was not valid JSON."})

    # ------------------------------------------------------------------ helpers

    def _allowed(self) -> bool:
        """Only this computer's own page: the Host must be local (no DNS rebinding) and a request
        sent by a page must come from this app's page (no cross-site requests)."""
        port = self.server.server_address[1]
        own = {f"{host}:{port}" for host in LOCAL_HOSTS}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in own or (origin is not None and urlsplit(origin).netloc not in own):
            self._json(HTTPStatus.FORBIDDEN, {"error": "Requests from other sites are refused."})
            return False
        return True

    def _body(self) -> bytes:
        """The request body. A missing or malformed length reads as no body; the loader then
        says the file is empty."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return b""
        return self.rfile.read(length) if length > 0 else b""

    def _json(self, status: HTTPStatus, payload: object) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)


def make_server(folder: str | Path, port: int = DEFAULT_PORT, settings: Settings | None = None) -> App:
    """The app, serving a workspace kept in `folder`, on 127.0.0.1:`port` (0 picks a free port).
    `settings` replaces the default analysis settings (tests use faster ones)."""
    return App(port, Workspace(folder), settings)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="asclepius-app", description="Start the Asclepius web app.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page in a browser")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="asclepius-") as folder:
        try:
            server = make_server(folder, args.port)
        except OSError as error:
            parser.exit(1, f"Cannot listen on port {args.port} ({error}); try --port with another number.\n")
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(
            f"Asclepius is running at {url}\nStop it with Ctrl+C; the files picked are then deleted.",
            flush=True,
        )
        if not args.no_browser:
            threading.Timer(0.5, partial(webbrowser.open, url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
