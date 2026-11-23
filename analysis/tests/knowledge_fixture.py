"""A small paper library and vocabulary for the knowledge graph and its map (M6-02, M6-04).

BDNF is in a PSSD paper and a PFS paper, so two syndromes link to it; it is co-mentioned with
serotonin in two papers (31000001, 31000003) and with COMT in one (31000002).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from hypothesis_engine.library import write_build
from hypothesis_engine.pubmed import Paper

from asclepius.web.graph import READY
from asclepius.web.hypotheses import Engine
from asclepius.web.server import App, make_server

VOCABULARY = """
[[term]]
name = "post-SSRI sexual dysfunction"
kind = "syndrome"
aliases = []
abbreviations = ["PSSD"]
pmid = "31000001"

[[term]]
name = "post-finasteride syndrome"
kind = "syndrome"
aliases = []
abbreviations = ["PFS"]
pmid = "31000002"

[[term]]
name = "BDNF"
kind = "gene"
aliases = []
pmid = "31000001"

[[term]]
name = "COMT"
kind = "gene"
aliases = []
pmid = "31000002"

[[term]]
name = "serotonin"
kind = "hormone"
aliases = []
pmid = "31000001"
"""

PAPERS = [
    Paper(
        "31000001",
        "Post-SSRI sexual dysfunction",
        "BDNF and serotonin fell in PSSD.",
        "10.1/pssd",
        None,
        "J Sex Med",
        2022,
    ),
    Paper(
        "31000002",
        "Post-finasteride syndrome",
        "BDNF and COMT changed in PFS.",
        None,
        None,
        "Andrology",
        2021,
    ),
    Paper(
        "31000003",
        "Serotonin and growth factors",
        "Serotonin and BDNF were measured.",
        None,
        None,
        None,
        2020,
    ),
]
TOPICS = {"31000001": [("PSSD", 1)], "31000002": [("PFS", 1)], "31000003": [("serotonin", 1)]}


def knowledge_engine(folder: Path) -> Engine:
    """An engine whose library and vocabulary are the ones above, written into `folder`."""
    write_build(folder / "library.sqlite", PAPERS, TOPICS, "sha", "2026-11-24", [])
    (folder / "vocabulary.toml").write_text(VOCABULARY, encoding="utf-8")
    return Engine(library=folder / "library.sqlite", vocabulary=folder / "vocabulary.toml")


@contextmanager
def serving(folder: Path) -> Iterator[tuple[App, str]]:
    """The app over that library, its graph built, and its address; stopped afterwards."""
    server = make_server(folder / "work", port=0, engine=knowledge_engine(folder))
    server.knowledge.start()
    assert server.knowledge.wait(timeout=30) == READY
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
