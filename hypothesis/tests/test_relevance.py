"""Hand-made relevance test: known questions, worded unlike the papers, find the known paper.

Runs the real local model (bge-small-en-v1.5) with the network cut off, on twelve real papers
(relevance_fixture.py). The model must have been downloaded once (`asclepius-library index`);
without it these tests are skipped.
"""

from __future__ import annotations

import socket

import pytest
from relevance_fixture import make_relevance_library

from hypothesis_engine.embedding import LocalEmbedder, model_downloaded
from hypothesis_engine.search import Index, build_index

pytestmark = pytest.mark.skipif(
    not model_downloaded(), reason="embedding model not downloaded; run `asclepius-library index` once"
)

# (question, the paper that answers it, a word the passage found in it must hold), read by hand
KNOWN = [
    ("is post-finasteride syndrome a real clinical entity", "39953145", "PFS"),
    ("similarities between finasteride and SSRI sexual side effects", "29675596", "SSRI"),
    ("why is the prevalence of PSSD hard to measure", "39289881", "prevalence"),
    (
        "UK medicines regulator warning about lasting sexual dysfunction from antidepressants",
        "39289881",
        "MHRA",
    ),
    ("dutasteride and risk of aggressive prostate cancer", "37697052", "prostate cancer"),
    ("history of how fluoxetine and the first SSRIs were developed", "41640943", "fluoxetine"),
    ("depression and suicidal thoughts in men who took a hair loss drug", "32033719", "suicidal"),
    ("Spanish literature review of the syndrome", "35983808", "review"),
]


def _disconnected(*_args, **_kwargs):
    raise OSError("network disconnected (test)")


def _cut_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every way out to the network fails, as with the cable pulled."""
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, _disconnected)
    monkeypatch.setattr(socket, "create_connection", _disconnected)
    monkeypatch.setattr(socket, "getaddrinfo", _disconnected)


@pytest.fixture
def offline(monkeypatch):
    _cut_off(monkeypatch)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    path = tmp_path_factory.mktemp("relevance") / "library.sqlite"
    make_relevance_library(path)
    return path


@pytest.fixture(scope="module")
def index(library, tmp_path_factory):
    """Built and opened with the network cut off; searched with it cut off too (`offline`)."""
    path = tmp_path_factory.mktemp("index") / "index.sqlite"
    with pytest.MonkeyPatch.context() as monkeypatch:
        _cut_off(monkeypatch)
        embedder = LocalEmbedder()  # loads the model from disk
        build_index(library, path, embedder)
        opened = Index.open(path, library, embedder)
    with opened:
        yield opened


def test_the_network_is_really_cut_off(offline):
    with pytest.raises(OSError, match="disconnected"):
        socket.create_connection(("pubmed.ncbi.nlm.nih.gov", 443))


@pytest.mark.parametrize(("question", "pmid", "word"), KNOWN, ids=[q for q, _p, _w in KNOWN])
def test_a_known_question_finds_its_paper_in_the_top_three(index, offline, question, pmid, word):
    hits = index.search(question, k=3)
    assert pmid in [h.paper_id for h in hits], [(h.passage_id, h.title) for h in hits]
    assert any(word.casefold() in h.text.casefold() for h in hits if h.paper_id == pmid)


def test_most_known_questions_find_their_paper_first(index, offline):
    first = sum(index.search(question, k=1)[0].paper_id == pmid for question, pmid, _word in KNOWN)
    assert first >= len(KNOWN) - 2


def test_the_search_command_prints_the_passages_offline(index, library, offline, tmp_path, capsys):
    from hypothesis_engine.build import main

    assert main(["index", "--library", str(library), "--index", str(tmp_path / "index.sqlite")]) == 0
    question = "dutasteride and risk of aggressive prostate cancer"
    assert (
        main(
            [
                "search",
                question,
                "-k",
                "2",
                "--library",
                str(library),
                "--index",
                str(tmp_path / "index.sqlite"),
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("1. 37697052:")
    assert len([line for line in lines if line[:1].isdigit()]) == 2
