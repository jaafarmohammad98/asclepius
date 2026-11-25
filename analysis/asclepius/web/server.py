"""Serve the web app on this computer only: `uv run asclepius-app`, then use the page it opens.

The page (`index.html`) talks to a small JSON API over `asclepius.web.workspace.Workspace`:

| Request | Does |
|---|---|
| `POST /api/clear` | forget the files picked |
| `PUT /api/files/<role>?name=<file name>` | add one picked file, its bytes as the body; role `reports`, `metadata` or `export` |
| `POST /api/load` | load the files; the review, or `{"error": ...}` with status 400 |
| `POST /api/example/<kind>` | load the built-in example (`asclepius.examples`), `lab-report` or `self-report`, in place of the files; the review |
| `POST /api/run` | start the run, `{"group_a": ..., "group_b": ...}` for lab reports |
| `GET /api/status` | the run's state and each step's |
| `GET /api/findings` | each step's findings, sizes, labels and plots (`asclepius.web.findings`) |
| `GET /api/export/findings.csv` | the finished run's findings as a CSV file (`asclepius.web.export`) |
| `GET /api/export/report.html` | the finished run's report, with its plots and run record, as an HTML file |
| `POST /api/hypotheses` | start writing hypotheses from the finished run's findings (`asclepius.web.hypotheses`) |
| `GET /api/hypotheses` | their state and, once finished, each hypothesis with its finding, citations and sources |
| `GET /api/export/hypotheses.html` | the hypotheses as an HTML report with the run records |
| `GET /api/settings` | the LLM settings (`asclepius.web.llm_settings`); the key only as whether one is stored |
| `POST /api/settings` | save them, `{"backend": ..., "anthropic": {...}, "local": {...}}`; 400 when the engine would refuse them |
| `POST /api/settings/check` | ask the LLM one short question; which model answered |
| `GET /api/graph` | the bottleneck map (`asclepius.web.graph`): scored terms, co-mentions and syndrome links (with the latest run's finding links, and `run`, its dataset), or the graph's state while it builds |
| `GET /api/graph/edge?a=<term>&b=<term>` | the link a → b and the passages it was counted from; 404 when the graph has none |
| `GET /api/leaderboard` | the drug repurposing leaderboard (`asclepius.web.leaderboard`): each cached drug's TOPSIS criteria, risks and targets, and the known causes apart |
| `GET /topsis.js` | the leaderboard's TOPSIS, which the page ranks with as the weight sliders move |
| `GET /vendor/cytoscape.min.js` | the network-drawing library, shipped with the app so the page needs no network |
| `GET /extractor/...` | the extractor's browser page (`extractor/`, built by `npm run build`), for reading report PDFs |

The server listens on 127.0.0.1 and refuses requests naming another host or coming from another
site's page, so no other web page open in the browser can read the data or start a run. The
extractor page is the one `npm run build` writes: the installed app carries a copy
(`asclepius/web/extractor/`, put there by `package/build.py`), and a source checkout serves
`extractor/dist/`. It reads PDFs in the browser; nothing it reads reaches this server. Picked
files are kept in a temporary folder that is deleted when the app stops. As it opens, the app builds
the knowledge graph from the paper library in the background (`asclepius.web.graph`), and adds each
finished run's findings to it.
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
from urllib.parse import parse_qs, unquote, urlsplit

from asclepius.examples import EXAMPLES
from asclepius.run import Settings
from asclepius.web import llm_settings
from asclepius.web.graph import KnowledgeGraph
from asclepius.web.hypotheses import Engine
from asclepius.web.leaderboard import leaderboard_view
from asclepius.web.llm_settings import SettingsError
from asclepius.web.workspace import ROLES, LoadError, RunError, Workspace

PAGE = Path(__file__).parent / "index.html"
SCRIPTS = {  # the page's scripts
    "/vendor/cytoscape.min.js": Path(__file__).parent / "vendor" / "cytoscape.min.js",
    "/topsis.js": Path(__file__).parent / "topsis.js",
}
PACKAGED_EXTRACTOR = Path(__file__).parent / "extractor"  # in the installed app
CHECKOUT_EXTRACTOR = Path(__file__).resolve().parents[3] / "extractor" / "dist"  # in a source checkout
EXTRACTOR_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".webmanifest": "application/manifest+json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".wasm": "application/wasm",
}
NOT_BUILT = (
    "<!doctype html><title>Extractor not built</title><p>The extractor page is not built. In the "
    "<code>extractor</code> folder run <code>npm install</code> and <code>npm run build</code>, then "
    "start the app again.</p>"
)
DEFAULT_PORT = 8765
LOCAL_HOSTS = ("127.0.0.1", "localhost")
EXPORTS = {  # path -> the file's suffix, its type and the workspace method that makes it
    "/api/export/findings.csv": ("csv", "text/csv; charset=utf-8", Workspace.export_csv),
    "/api/export/report.html": ("html", "text/html; charset=utf-8", Workspace.export_report),
    "/api/export/hypotheses.html": ("html", "text/html; charset=utf-8", Workspace.export_hypotheses),
}


class App(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, port: int, workspace: Workspace, settings: Settings | None, engine: Engine, extractor: Path
    ) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.extractor = extractor
        self.workspace = workspace
        self.settings = settings
        self.engine = engine
        self.knowledge = workspace.knowledge or KnowledgeGraph(engine)  # built by `main` as the app opens
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
        elif path == "/api/findings":
            self._json(HTTPStatus.OK, self.server.workspace.findings())
        elif path == "/api/hypotheses":
            self._json(HTTPStatus.OK, self.server.workspace.hypotheses_view())
        elif path == "/api/settings":
            engine = self.server.engine
            self._json(HTTPStatus.OK, llm_settings.read(engine.settings, engine.env))
        elif path == "/api/graph":
            self._json(HTTPStatus.OK, self.server.knowledge.map_view())
        elif path == "/api/leaderboard":
            self._json(HTTPStatus.OK, leaderboard_view(self.server.engine.drugs, self.server.knowledge))
        elif path == "/api/graph/edge":
            self._edge(parse_qs(urlsplit(self.path).query))
        elif path in SCRIPTS:
            self._send(HTTPStatus.OK, SCRIPTS[path].read_bytes(), "text/javascript; charset=utf-8")
        elif path in EXPORTS:
            self._export(path)
        elif path == "/extractor":  # the page's file paths are relative to the folder
            self._send(HTTPStatus.MOVED_PERMANENTLY, b"", "text/plain", {"Location": "/extractor/"})
        elif path.startswith("/extractor/"):
            self._extractor(unquote(path.removeprefix("/extractor/")))
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
            elif (kind := path.removeprefix("/api/example/")) in EXAMPLES and path != kind:
                with self.server.lock:
                    review = workspace.load_example(kind)
                self._json(HTTPStatus.OK, review)
            elif path == "/api/run":
                choice = json.loads(body or b"{}")
                if not isinstance(choice, dict):
                    raise RunError("The run request must name the groups as a JSON object.")
                workspace.start_run(
                    choice.get("group_a"), choice.get("group_b"), settings=self.server.settings
                )
                self._json(HTTPStatus.ACCEPTED, workspace.status())
            elif path == "/api/hypotheses":
                with self.server.lock:  # no load or clear while they start
                    workspace.start_hypotheses()
                self._json(HTTPStatus.ACCEPTED, workspace.hypotheses_status())
            elif path == "/api/settings":
                form = json.loads(body or b"{}")
                if not isinstance(form, dict):
                    raise SettingsError("The settings must be a JSON object.")
                engine = self.server.engine
                with self.server.lock:  # one save at a time
                    saved = llm_settings.save(engine.settings, form, engine.env)
                self._json(HTTPStatus.OK, saved)
            elif path == "/api/settings/check":
                engine = self.server.engine
                self._json(HTTPStatus.OK, llm_settings.check(engine.settings, engine.env, engine.connect_llm))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        except (LoadError, RunError, SettingsError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except json.JSONDecodeError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "The request was not valid JSON."})

    def _export(self, path: str) -> None:
        """A finished run's export, as a file to save, named by the dataset's fingerprint."""
        workspace = self.server.workspace
        suffix, content_type, make = EXPORTS[path]
        try:
            with self.server.lock:  # no load or clear between the export and its name
                content = make(workspace)
                fingerprint = str(getattr(workspace.data, "fingerprint", ""))[:12]
        except RunError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        name = f"asclepius-{Path(path).stem}-{fingerprint}.{suffix}"
        self._send(
            HTTPStatus.OK,
            content.encode("utf-8"),
            content_type,
            {"Content-Disposition": f'attachment; filename="{name}"'},
        )

    def _edge(self, query: dict[str, list[str]]) -> None:
        a, b = (query.get(k, [""])[0] for k in ("a", "b"))
        try:
            self._json(HTTPStatus.OK, self.server.knowledge.edge(a, b))
        except KeyError:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"The knowledge graph has no link from {a} to {b}."})

    def _extractor(self, name: str) -> None:
        """One file of the extractor's built page; nothing outside its folder."""
        root = self.server.extractor.resolve()
        if not (root / "index.html").is_file():
            self._send(HTTPStatus.NOT_FOUND, NOT_BUILT.encode(), "text/html; charset=utf-8")
            return
        try:
            file = (root / (name or "index.html")).resolve()
        except (OSError, ValueError):  # a NUL byte, or a name the system refuses
            file = root
        if not file.is_relative_to(root) or not file.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        kind = EXTRACTOR_TYPES.get(file.suffix, "application/octet-stream")
        self._send(HTTPStatus.OK, file.read_bytes(), kind)

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
        # allow_nan=False: NaN is not JSON, and the page's parser would refuse the whole answer
        self._send(status, json.dumps(payload, allow_nan=False).encode(), "application/json")

    def _send(
        self, status: HTTPStatus, body: bytes, content_type: str, headers: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)


def make_server(
    folder: str | Path,
    port: int = DEFAULT_PORT,
    settings: Settings | None = None,
    engine: Engine | None = None,
    extractor: Path | None = None,
) -> App:
    """The app, serving a workspace kept in `folder`, on 127.0.0.1:`port` (0 picks a free port).
    `settings` replaces the default analysis settings (tests use faster ones); `engine`, where the
    hypothesis engine's library, index and LLM settings are (tests use their own); `extractor`, the
    folder of the extractor's built page (by default the installed copy, else the checkout's)."""
    engine = engine or Engine()
    if extractor is None:
        extractor = PACKAGED_EXTRACTOR if PACKAGED_EXTRACTOR.is_dir() else CHECKOUT_EXTRACTOR
    return App(port, Workspace(folder, engine, KnowledgeGraph(engine)), settings, engine, extractor)


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
        server.knowledge.start()
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
