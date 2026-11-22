"""Building the paper library from recorded PubMed responses (fixtures/build, recorded 2026-11-22)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from replay import ReplayTransport
from test_rate_limit import FakeClock, _most_in_any_second

from hypothesis_engine.build import build_library
from hypothesis_engine.library import Library, LibraryError
from hypothesis_engine.pubmed import PubMed, PubMedError, RateLimiter, TransientError
from hypothesis_engine.topics import DEFAULT_TOPICS, load_topics

FOLDER = Path(__file__).parent / "fixtures" / "build"
TOPICS = load_topics(FOLDER / "topics.toml")
WHEN = datetime(2026, 11, 22, 9, 30, tzinfo=UTC)

# From the recorded esearch answers, read by hand: the topics' PMIDs, in relevance order.
PFS = ["31399423", "32033719", "39953145", "35983808"]
ENZYMES = ["32317131", "35983808", "37697052", "29675596"]
LETTERS = ["19439893", "21690548"]
SHARED = "35983808"  # found by both PFS and enzymes
# no abstract in PubMed: the two letters, and a BMJ news item found by the PFS search
LACKING = {"19439893", "21690548", "31399423"}


class Recorded:
    """The replayed PubMed, plus the times its requests were sent."""

    def __init__(self) -> None:
        self.clock = FakeClock()
        self.replay = ReplayTransport(FOLDER)
        self.sent_at: list[float] = []
        self.pubmed = PubMed(
            self._send, RateLimiter(3, self.clock.time, self.clock.sleep), self.clock.sleep, 4
        )

    def _send(self, endpoint: str, params: dict[str, str]) -> bytes:
        self.sent_at.append(self.clock.now)
        return self.replay(endpoint, params)


def _build(path: Path, pubmed: PubMed | None = None):
    return build_library(path, TOPICS, pubmed or Recorded().pubmed, now=lambda: WHEN)


def test_every_paper_found_by_a_topic_search_is_stored_once(tmp_path):
    report = _build(tmp_path / "library.sqlite")
    with Library.open(tmp_path / "library.sqlite") as library:
        expected = set(PFS) | set(ENZYMES) | set(LETTERS)
        assert {p.pmid for p in library.papers()} == expected
        assert library.count() == len(expected) == report.papers


def test_every_paper_has_an_identifier_and_an_abstract_or_is_marked_as_lacking_one(tmp_path):
    report = _build(tmp_path / "library.sqlite")
    with Library.open(tmp_path / "library.sqlite") as library:
        for paper in library.papers():
            assert paper.pmid.isdigit()
            assert paper.title
            assert paper.abstract is None or paper.abstract.strip()
        lacking = {p.pmid for p in library.papers() if p.abstract is None}
        assert lacking == LACKING
        assert library.lacking_abstract() == report.lacking_abstract == len(LACKING)


def test_a_paper_is_listed_under_every_topic_that_found_it(tmp_path):
    _build(tmp_path / "library.sqlite")
    with Library.open(tmp_path / "library.sqlite") as library:
        assert library.topics(SHARED) == ["PFS", "steroid metabolism enzymes"]
        assert library.topics(LETTERS[0]) == ["letters"]


def test_a_stored_paper_reads_back_with_its_identifiers(tmp_path):
    _build(tmp_path / "library.sqlite")
    with Library.open(tmp_path / "library.sqlite") as library:
        paper = library.paper("32317131")
        assert paper is not None
        assert paper.title == "Post-finasteride syndrome."
        assert paper.doi == "10.1016/j.abd.2020.02.001"
        assert paper.pmcid == "PMC7253896"
        assert paper.year == 2020
        assert library.paper("99999999") is None


def test_the_first_build_is_version_1_and_each_rebuild_adds_one(tmp_path):
    path = tmp_path / "library.sqlite"
    assert _build(path).version == 1
    assert _build(path).version == 2
    with Library.open(path) as library:
        assert library.version == 2
        assert library.built_at == "2026-11-22T09:30:00+00:00"


def test_requests_stay_under_three_a_second(tmp_path):
    recorded = Recorded()
    _build(tmp_path / "library.sqlite", recorded.pubmed)
    assert len(recorded.sent_at) >= 5  # three searches, then efetch in batches of 4
    assert _most_in_any_second(recorded.sent_at) <= 3


def test_a_request_that_fails_for_a_moment_is_sent_again(tmp_path):
    recorded = Recorded()
    replay = recorded.replay
    failed: list[str] = []

    def flaky(endpoint: str, params: dict[str, str]) -> bytes:
        if not failed:
            failed.append(endpoint)
            raise TransientError("HTTP 503")
        return replay(endpoint, params)

    recorded.replay = flaky  # type: ignore[assignment]
    assert _build(tmp_path / "library.sqlite", recorded.pubmed).papers > 0
    assert failed == ["esearch"]


def test_a_build_that_fails_leaves_the_previous_library_as_it_was(tmp_path):
    path = tmp_path / "library.sqlite"
    first = _build(path)
    recorded = Recorded()
    replay = recorded.replay

    def broken(endpoint: str, params: dict[str, str]) -> bytes:
        if endpoint == "efetch":
            raise PubMedError("efetch: HTTP 400 Bad Request")
        return replay(endpoint, params)

    recorded.replay = broken  # type: ignore[assignment]
    with pytest.raises(PubMedError):
        _build(path, recorded.pubmed)
    with Library.open(path) as library:
        assert library.version == 1
        assert library.count() == first.papers


def test_opening_a_missing_library_names_the_build_command(tmp_path):
    with pytest.raises(LibraryError, match="asclepius-library build"):
        Library.open(tmp_path / "nothing.sqlite")


def test_opening_a_file_that_is_not_a_library_fails(tmp_path):
    path = tmp_path / "other.sqlite"
    sqlite3.connect(path).execute("CREATE TABLE notes (x)").connection.commit()
    with pytest.raises(LibraryError, match="not a paper library"):
        Library.open(path)


def test_the_shipped_topic_list_covers_every_topic_in_the_spec():
    names = [t.name for t in load_topics(DEFAULT_TOPICS)]
    assert names == [
        "PSSD",
        "PFS",
        "PAS",
        "neurosteroids",
        "steroid metabolism enzymes",
        "serotonin and sexual function",
        "small-fibre neuropathy",
        "epigenetic effects of drugs",
    ]
    assert sum(t.max_papers for t in load_topics(DEFAULT_TOPICS)) <= 5000  # "a few thousand"
