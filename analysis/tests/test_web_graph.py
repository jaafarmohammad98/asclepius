"""The knowledge graph in the web app (M6-02): built from the library in the background when the app
opens, never stored."""

from hypothesis_engine.library import write_build
from hypothesis_engine.pubmed import Paper

from asclepius.web.graph import FAILED, READY, KnowledgeGraph
from asclepius.web.hypotheses import BUILD_LIBRARY, Engine

VOCABULARY = """
[[term]]
name = "post-SSRI sexual dysfunction"
kind = "syndrome"
aliases = []
abbreviations = ["PSSD"]
pmid = "31000001"

[[term]]
name = "BDNF"
kind = "gene"
aliases = []
pmid = "31000001"
"""


def test_the_graph_is_built_from_the_library_when_started(tmp_path):
    paper = Paper("31000001", "Post-SSRI sexual dysfunction", "BDNF fell in PSSD.", None, None, None, 2022)
    write_build(tmp_path / "library.sqlite", [paper], {"31000001": [("PSSD", 1)]}, "sha", "2026-11-23", [])
    (tmp_path / "vocabulary.toml").write_text(VOCABULARY, encoding="utf-8")
    built = KnowledgeGraph(
        Engine(library=tmp_path / "library.sqlite", vocabulary=tmp_path / "vocabulary.toml")
    )
    built.start()
    assert built.wait(timeout=30) == READY
    assert built.graph is not None
    assert built.graph.edges["post-SSRI sexual dysfunction", "BDNF"]["kind"] == "literature"


def test_without_a_library_the_graph_says_how_to_build_one(tmp_path):
    built = KnowledgeGraph(Engine(library=tmp_path / "missing.sqlite"))
    built.start()
    assert built.wait(timeout=30) == FAILED
    assert built.graph is None
    assert built.message.endswith(BUILD_LIBRARY)
