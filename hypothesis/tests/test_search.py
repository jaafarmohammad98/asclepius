"""The local search index, with a stand-in for the embedding model (see test_relevance.py for the real one)."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from hypothesis_engine.library import write_build
from hypothesis_engine.private import PrivatePaper
from hypothesis_engine.pubmed import Paper
from hypothesis_engine.search import Index, SearchIndexError, build_index

# The stand-in "understands" these words to mean the same; keyword search does not.
SAME_MEANING = {"neurosteroid": "allopregnanolone", "neurosteroids": "allopregnanolone", "fluid": "csf"}


class WordEmbedder:
    """Each word is a direction; words in SAME_MEANING share one. Counts what it embedded."""

    name = "test-words"
    dimensions = 512

    def __init__(self) -> None:
        self.embedded = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in re.findall(r"\w+", text.casefold()):
            vector[zlib.crc32(SAME_MEANING.get(word, word).encode()) % self.dimensions] += 1
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def passages(self, texts: Sequence[str]) -> np.ndarray:
        self.embedded += len(texts)
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dimensions))

    def query(self, text: str) -> np.ndarray:
        return self._vector(text)


FILLER = " ".join(f"filler{i}" for i in range(30))
PAPERS = [
    Paper(
        "1",
        "Allopregnanolone after finasteride",
        f"Allopregnanolone in CSF was lower. {FILLER}",
        None,
        None,
        None,
        2020,
    ),
    Paper(
        "2",
        "Hair loss treatment",
        f"Finasteride slowed hair loss in most men. {FILLER}",
        None,
        None,
        None,
        2021,
    ),
    Paper(
        "3",
        "Genital numbness after SSRIs",
        f"Genital anaesthesia persisted after sertraline. {FILLER}",
        None,
        None,
        None,
        2022,
    ),
    Paper("4", "A letter", None, None, None, None, 2019),
]
PRIVATE = PrivatePaper(
    "private-0123456789abcdef", "cohort.pdf", "Our cohort", f"Twelve men with lasting anhedonia. {FILLER}"
)


def _library(
    path: Path, papers: Sequence[Paper] = PAPERS, private: Sequence[PrivatePaper] = (PRIVATE,)
) -> Path:
    found_by = {p.pmid: [("PSSD", rank)] for rank, p in enumerate(papers, start=1)}
    write_build(path, papers, found_by, "sha", "2026-11-22T00:00:00+00:00", list(private))
    return path


@pytest.fixture
def index(tmp_path):
    library = _library(tmp_path / "library.sqlite")
    build_index(library, tmp_path / "index.sqlite", WordEmbedder())
    with Index.open(tmp_path / "index.sqlite", library, WordEmbedder()) as opened:
        yield opened


def test_a_search_returns_passages_with_their_paper_and_text(index):
    top = index.search("sertraline genital anaesthesia", k=3)[0]
    assert (top.passage_id, top.paper_id, top.private) == ("3:1", "3", False)
    assert top.title == "Genital numbness after SSRIs"
    assert top.text.startswith("Genital anaesthesia persisted after sertraline.")


def test_a_passage_is_found_by_meaning_without_sharing_a_word_with_the_query(index):
    (top, *_rest) = index.search("neurosteroid fluid", k=3)
    assert top.paper_id == "1"
    assert top.keyword_rank is None and top.meaning_rank == 1


def test_a_passage_is_found_by_an_exact_word_and_by_its_stem(index):
    top = index.search("persisting", k=1)[0]
    assert top.paper_id == "3"
    assert top.keyword_rank == 1


def test_a_passage_both_searches_find_ranks_above_one_only_meaning_finds(index):
    # "finasteride" is a keyword of papers 1 and 2; "allopregnanolone" by meaning only matches paper 1
    hits = index.search("finasteride neurosteroid", k=4)
    assert hits[0].paper_id == "1"
    assert hits[0].keyword_rank is not None and hits[0].meaning_rank is not None
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_k_limits_the_results_and_a_query_with_no_words_finds_nothing(index):
    assert len(index.search("finasteride", k=2)) == 2
    assert index.search("  ?! ", k=5) == []


def test_a_paper_without_an_abstract_is_found_by_its_title(index):
    assert index.search("letter", k=1)[0].passage_id == "4:1"


def test_private_papers_are_searched_and_marked_private(index):
    top = index.search("lasting anhedonia cohort", k=1)[0]
    assert (top.paper_id, top.private) == (PRIVATE.id, True)


def test_a_passage_reads_back_by_its_id(index):
    passage = index.passage("1:1")
    assert passage is not None and passage.paper_id == "1"
    assert passage.text.startswith("Allopregnanolone in CSF was lower.")
    assert index.passage("1:99") is None


def test_the_index_records_the_library_version_and_model(index):
    assert (index.library_version, index.model) == (1, "test-words")


def test_an_index_of_an_older_library_version_is_refused(tmp_path):
    library = _library(tmp_path / "library.sqlite")
    build_index(library, tmp_path / "index.sqlite", WordEmbedder())
    _library(library)  # version 2
    with pytest.raises(SearchIndexError, match="asclepius-library index"):
        Index.open(tmp_path / "index.sqlite", library, WordEmbedder())


def test_an_index_made_with_another_model_is_refused(tmp_path):
    library = _library(tmp_path / "library.sqlite")
    build_index(library, tmp_path / "index.sqlite", WordEmbedder())
    other = WordEmbedder()
    other.name = "another-model"
    with pytest.raises(SearchIndexError, match="test-words"):
        Index.open(tmp_path / "index.sqlite", library, other)


def test_opening_a_missing_index_names_the_command(tmp_path):
    library = _library(tmp_path / "library.sqlite")
    with pytest.raises(SearchIndexError, match="asclepius-library index"):
        Index.open(tmp_path / "index.sqlite", library, WordEmbedder())


def test_rebuilding_embeds_only_passages_it_has_not_embedded_before(tmp_path):
    library = _library(tmp_path / "library.sqlite")
    first = build_index(library, tmp_path / "index.sqlite", WordEmbedder())
    assert (first.passages, first.embedded) == (5, 5)
    new = Paper("5", "A new paper", f"Something new about tinnitus. {FILLER}", None, None, None, 2023)
    _library(library, [*PAPERS, new])
    embedder = WordEmbedder()
    second = build_index(library, tmp_path / "index.sqlite", embedder)
    assert (second.version, second.passages, second.embedded, embedder.embedded) == (2, 6, 1, 1)
    with Index.open(tmp_path / "index.sqlite", library, WordEmbedder()) as index:
        assert index.search("tinnitus", k=1)[0].paper_id == "5"


def test_a_paper_a_new_version_drops_leaves_the_index(tmp_path):
    library = _library(tmp_path / "library.sqlite")
    build_index(library, tmp_path / "index.sqlite", WordEmbedder())
    _library(library, PAPERS[1:], private=())
    build_index(library, tmp_path / "index.sqlite", WordEmbedder())
    with Index.open(tmp_path / "index.sqlite", library, WordEmbedder()) as index:
        found = {h.paper_id for h in index.search("allopregnanolone anhedonia finasteride", k=10)}
        assert found == {"2", "3", "4"}


def test_without_the_model_searching_says_how_to_download_it_and_does_not_try(tmp_path, monkeypatch):
    import socket

    from hypothesis_engine.embedding import LocalEmbedder, ModelMissing

    class TriedTheNetwork(BaseException):  # not an Exception: nothing on the way may catch it
        pass

    def disconnected(*_args, **_kwargs):
        raise TriedTheNetwork

    monkeypatch.setattr(socket, "getaddrinfo", disconnected)
    monkeypatch.setattr(socket.socket, "connect", disconnected)
    with pytest.raises(ModelMissing, match="asclepius-library index"):
        LocalEmbedder(tmp_path / "models")
