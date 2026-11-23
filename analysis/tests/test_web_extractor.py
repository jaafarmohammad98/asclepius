"""The extractor's browser page, served by the app at /extractor/ so one install gives both."""

import threading
import urllib.request

import pytest
from test_web_server import call

from asclepius.web.server import make_server


def serve(tmp_path, extractor):
    server = make_server(tmp_path / "work", port=0, extractor=extractor)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def built(tmp_path):
    """A built page as `npm run build` lays it out, and the app serving it."""
    page = tmp_path / "dist"
    (page / "assets").mkdir(parents=True)
    (page / "index.html").write_text(
        '<!doctype html><title>DUTCH extractor</title><script src="./assets/index-1.js">'
    )
    (page / "assets" / "index-1.js").write_text("console.log(1)")
    (page / "assets" / "index-1.css").write_text("body{}")
    (page / "sw.js").write_text("self.addEventListener('fetch', () => {})")
    (tmp_path / "secret.txt").write_text("not part of the page")
    server, url = serve(tmp_path, page)
    yield url
    server.shutdown()
    server.server_close()


def content_type(url):
    with urllib.request.urlopen(url) as response:
        return response.headers["Content-Type"]


def test_the_page_and_its_files_are_served_with_their_types(built):
    status, page = call(built + "/extractor/")
    assert status == 200 and b"<title>DUTCH extractor</title>" in page
    assert content_type(built + "/extractor/").startswith("text/html")
    assert content_type(built + "/extractor/assets/index-1.js").startswith("text/javascript")
    assert content_type(built + "/extractor/assets/index-1.css").startswith("text/css")
    assert content_type(built + "/extractor/sw.js").startswith("text/javascript")


def test_without_the_slash_it_redirects_so_relative_paths_resolve(built):
    status, page = call(built + "/extractor")  # urllib follows the redirect
    assert status == 200 and b"<title>DUTCH extractor</title>" in page


@pytest.mark.parametrize(
    "path",
    ["/extractor/../secret.txt", "/extractor/%2e%2e/secret.txt", "/extractor/assets/", "/extractor/%00"],
)
def test_nothing_outside_the_page_files_is_served(built, path):
    status, content = call(built + path)
    assert status == 404 and b"not part of the page" not in content


def test_a_missing_file_is_not_found(built):
    assert call(built + "/extractor/assets/missing.js")[0] == 404


def test_the_app_page_links_to_the_extractor(built):
    assert b'href="/extractor/"' in call(built + "/")[1]


def test_when_the_page_is_not_built_it_says_how(tmp_path):
    server, url = serve(tmp_path, tmp_path / "nowhere")
    try:
        status, content = call(url + "/extractor/")
        assert status == 404 and b"npm run build" in content
    finally:
        server.shutdown()
        server.server_close()
