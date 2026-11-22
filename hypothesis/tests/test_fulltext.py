"""Open-access full text from recorded answers (fixtures/fulltext, recorded 2026-11-22):
Europe PMC's open-access lookups, then PubMed Central's article XML."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fulltext_fixture import (
    DOWNLOAD_BATCH,
    FOLDER,
    FREE_TO_READ_ONLY,
    LOOKUP_BATCH,
    NO_COPY,
    OPEN_ACCESS,
    make_library,
)
from replay import ReplayTransport
from test_rate_limit import FakeClock, _most_in_any_second

from hypothesis_engine.europepmc import EuropePMC
from hypothesis_engine.fulltext import describe, fetch_full_texts
from hypothesis_engine.library import Coverage, Library, LibraryError
from hypothesis_engine.pubmed import PubMed, RateLimiter, TransientError

WHEN = datetime(2026, 11, 22, 10, 0, tzinfo=UTC)
# download batches of two, in PMID order
FIRST_BATCH = "7253896,11450419"
LAST_BATCH = "12864535"


class Recorded:
    """Replayed Europe PMC and PubMed Central, the requests they got and when they were sent.

    `fail` maps a request (the efetch ids, or "searchPOST#n" for the nth lookup) to the
    exception to raise instead; `answer` maps one to the body to send instead of the recorded one.
    """

    def __init__(
        self, fail: dict[str, BaseException] | None = None, answer: dict[str, bytes] | None = None
    ) -> None:
        self.clock = FakeClock()
        self.replay = ReplayTransport(FOLDER)
        self.fail = fail or {}
        self.answer = answer or {}
        self.lookups = 0
        self.downloads: list[str] = []
        self.sent_at: dict[str, list[float]] = {"europepmc": [], "ncbi": []}
        self.europepmc = EuropePMC(
            self._lookup, RateLimiter(3, self.clock.time, self.clock.sleep), self.clock.sleep, LOOKUP_BATCH
        )
        self.pubmed = PubMed(
            self._download, RateLimiter(3, self.clock.time, self.clock.sleep), self.clock.sleep
        )

    def _lookup(self, endpoint: str, fields: dict[str, str]) -> bytes:
        self.lookups += 1
        self.sent_at["europepmc"].append(self.clock.now)
        return self._reply(f"searchPOST#{self.lookups}", endpoint, fields)

    def _download(self, endpoint: str, params: dict[str, str]) -> bytes:
        assert (endpoint, params["db"]) == ("efetch", "pmc")
        self.downloads.append(params["id"])
        self.sent_at["ncbi"].append(self.clock.now)
        return self._reply(params["id"], endpoint, params)

    def _reply(self, key: str, endpoint: str, params: dict[str, str]) -> bytes:
        if key in self.fail:
            raise self.fail[key]
        if key in self.answer:
            return self.answer[key]
        return self.replay(endpoint, params)


def _library(tmp_path: Path) -> Path:
    path = tmp_path / "library.sqlite"
    make_library(path)
    return path


def _fetch(path: Path, recorded: Recorded | None = None, **options):
    recorded = recorded or Recorded()
    return fetch_full_texts(
        path, recorded.europepmc, recorded.pubmed, now=lambda: WHEN, download_batch=DOWNLOAD_BATCH, **options
    )


def test_open_access_papers_get_their_text_and_licence(tmp_path):
    path = _library(tmp_path)
    _fetch(path)
    with Library.open(path) as library:
        for pmid, (pmcid, licence) in OPEN_ACCESS.items():
            full = library.full_text(pmid)
            assert full is not None
            assert (full.pmcid, full.licence) == (pmcid, licence)
        paper = library.full_text("32317131")
        assert paper is not None
        assert "With a short half-life ranging from 4.7 to 7.1 h,1 it is able" in paper.text
        assert library.full_text(FREE_TO_READ_ONLY) is None  # free to read, but not open access
        assert library.full_text(NO_COPY) is None


def test_coverage_counts_papers_with_full_text(tmp_path):
    report = _fetch(_library(tmp_path))
    assert report.coverage == Coverage(
        papers=5, checked=5, open_access=3, with_text=3, unreadable=0, to_download=0
    )
    assert report.failed == 0
    assert describe(report) == "full text for 3 of 5 papers (60%); 3 have an open-access copy"


def test_a_copy_pmc_sends_no_text_for_is_counted_and_not_asked_for_again(tmp_path):
    path = _library(tmp_path)
    report = _fetch(path, Recorded(answer={LAST_BATCH: b"<pmc-articleset/>"}))
    assert report.coverage.with_text == 2
    assert report.coverage.unreadable == 1
    assert describe(report).endswith("3 have an open-access copy, 1 of them without readable text")

    again = Recorded()
    _fetch(path, again)
    assert again.downloads == []


def test_a_finished_run_sends_nothing_when_run_again(tmp_path):
    path = _library(tmp_path)
    _fetch(path)
    again = Recorded()
    assert _fetch(path, again).coverage.with_text == 3
    assert (again.lookups, again.downloads) == (0, [])


def test_an_interrupted_download_resumes_without_starting_over(tmp_path):
    path = _library(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        _fetch(path, Recorded(fail={LAST_BATCH: KeyboardInterrupt()}))
    with Library.open(path) as library:
        assert library.full_text("32317131") is not None  # stored before the interruption

    resumed = Recorded()
    assert _fetch(path, resumed).coverage.with_text == 3
    assert resumed.lookups == 0  # every paper was already looked up
    assert resumed.downloads == [LAST_BATCH]  # not the first batch again


def test_an_interrupted_lookup_resumes_with_the_papers_not_yet_looked_up(tmp_path):
    path = _library(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        _fetch(path, Recorded(fail={"searchPOST#2": KeyboardInterrupt()}))

    resumed = Recorded()
    assert _fetch(path, resumed).coverage.checked == 5
    assert resumed.lookups == 2  # of three: the first was kept


def test_requests_stay_under_three_a_second_to_each_source(tmp_path):
    recorded = Recorded()
    _fetch(_library(tmp_path), recorded)
    assert len(recorded.sent_at["europepmc"]) == 3  # lookups of two
    assert len(recorded.sent_at["ncbi"]) == 2  # downloads of two
    for times in recorded.sent_at.values():
        assert _most_in_any_second(times) <= 3


def test_a_download_that_keeps_failing_is_left_for_the_next_run(tmp_path):
    path = _library(tmp_path)
    report = _fetch(path, Recorded(fail={LAST_BATCH: TransientError("HTTP 503")}))
    assert report.failed == 1
    assert report.coverage.to_download == 1
    assert describe(report).endswith("; 1 still to fetch, run `asclepius-library fulltext` to continue")

    again = Recorded()
    assert _fetch(path, again).coverage.with_text == 3
    assert again.downloads == [LAST_BATCH]


def test_a_run_stops_when_pmc_stops_answering(tmp_path):
    path = _library(tmp_path)
    outage = {ids: TransientError("timed out") for ids in (FIRST_BATCH, LAST_BATCH)}
    down = Recorded(fail=outage)
    with pytest.raises(TransientError, match="2 downloads failed in a row"):
        _fetch(path, down, failures_in_a_row=2)
    assert down.downloads == [FIRST_BATCH] * 4 + [LAST_BATCH] * 4  # each sent, then retried three times

    assert _fetch(path).coverage.with_text == 3  # the lookups were kept; the next run downloads


def test_a_library_from_before_full_text_is_upgraded_in_place(tmp_path):
    path = _library(tmp_path)
    db = sqlite3.connect(path)
    db.executescript("DROP TABLE open_access; PRAGMA user_version = 1;")  # what M4-01a wrote
    db.close()
    with pytest.raises(LibraryError, match="asclepius-library fulltext"):
        Library.open(path)

    _fetch(path)
    with Library.open(path) as library:
        assert library.count() == 5
        assert library.coverage().with_text == 3


def test_full_text_needs_a_built_library(tmp_path):
    missing = tmp_path / "nothing.sqlite"
    with pytest.raises(LibraryError, match="asclepius-library build"):
        _fetch(missing)
    assert not missing.exists()
