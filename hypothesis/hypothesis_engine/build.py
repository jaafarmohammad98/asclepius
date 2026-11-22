"""Build the paper library: `asclepius-library build`, then `asclepius-library fulltext`.

`build` runs each topic search, fetches every paper found (once, however many topics found
it) and stores them as the library's next version. Nothing is written until every request
has succeeded, so a failed build leaves the previous library as it was. It then downloads
the open-access full text (see `fulltext.py`) and prints how many papers have it.

`fulltext` does only that second step, for the current version. It picks up where an
interrupted run stopped.

`refresh` is the monthly update: it runs the same searches, fetches only the papers not
stored yet, reads the private folder (`private.py`), and stores what is new as one version
(none when nothing is). Then it downloads the full text still missing, looking again for
copies of papers that had none a month ago. Running it twice in a row adds nothing the
second time.

Set NCBI_API_KEY to an NCBI API key (free, from an NCBI account) to send 10 requests a
second instead of 3. It is only ever sent to NCBI.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hypothesis_engine import europepmc
from hypothesis_engine.fulltext import describe, fetch_full_texts
from hypothesis_engine.library import LibraryError, stored_pmids, write_build, write_refresh
from hypothesis_engine.private import read_private_folder
from hypothesis_engine.pubmed import (
    PubMed,
    PubMedError,
    RateLimiter,
    TransientError,
    http_transport,
    rate_for,
)
from hypothesis_engine.topics import DEFAULT_TOPICS, Topic, TopicsError, load_topics

# inside hypothesis/, whatever folder the command runs from (git ignores hypothesis/data/)
DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "data" / "library.sqlite"
DEFAULT_PRIVATE = DEFAULT_LIBRARY.parent / "private"


@dataclass(frozen=True)
class BuildReport:
    version: int
    papers: int
    lacking_abstract: int
    found_per_topic: dict[str, int]  # topic -> PMIDs its search returned
    missing: int  # PMIDs a search returned that efetch had no usable record for


@dataclass(frozen=True)
class RefreshReport:
    version: int | None  # the new version, None when nothing was new
    new_papers: int
    private_added: int
    private_removed: int
    private_papers: int  # in the folder now
    skipped: list[tuple[str, str]]  # PDFs not added: (file name, reason)


def _fingerprint(topics: Sequence[Topic]) -> str:
    """Identifies the exact searches a build ran, so two versions can be told apart."""
    text = "\n".join(f"{t.name}\t{t.query}\t{t.max_papers}" for t in topics)
    return hashlib.sha256(text.encode()).hexdigest()


def _search(
    topics: Sequence[Topic], pubmed: PubMed, progress: Callable[[str], None]
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, int]]:
    """PMID -> (topic, rank) pairs in first-found order, and how many each topic's search found."""
    found_by: dict[str, list[tuple[str, int]]] = {}
    found_per_topic: dict[str, int] = {}
    for topic in topics:
        pmids = pubmed.search(topic.query, topic.max_papers)
        found_per_topic[topic.name] = len(pmids)
        progress(f"{topic.name}: {len(pmids)} papers")
        for rank, pmid in enumerate(pmids, start=1):
            found_by.setdefault(pmid, []).append((topic.name, rank))
    return found_by, found_per_topic


def build_library(
    path: Path,
    topics: Sequence[Topic],
    pubmed: PubMed,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    progress: Callable[[str], None] = lambda _message: None,
    private_folder: Path | None = None,
) -> BuildReport:
    """Search, fetch and store; return what the new library version holds.

    With `private_folder` its PDFs are the new version's private papers; without, the
    previous version's are kept.
    """
    found_by, found_per_topic = _search(topics, pubmed, progress)
    progress(f"fetching {len(found_by)} paper records")
    fetched = {paper.pmid: paper for paper in pubmed.fetch(list(found_by))}
    papers = [fetched[pmid] for pmid in found_by if pmid in fetched]
    private = read_private_folder(private_folder).papers if private_folder else None

    stamp = now().isoformat(timespec="seconds")
    version = write_build(path, papers, found_by, _fingerprint(topics), stamp, private)
    return BuildReport(
        version=version,
        papers=len(papers),
        lacking_abstract=sum(1 for p in papers if p.abstract is None),
        found_per_topic=found_per_topic,
        missing=len(found_by) - len(papers),
    )


def refresh_library(
    path: Path,
    topics: Sequence[Topic],
    pubmed: PubMed,
    private_folder: Path,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    progress: Callable[[str], None] = lambda _message: None,
) -> RefreshReport:
    """Add the papers the searches find that the library lacks, and the private folder's PDFs.

    Stored records are not downloaded again. Nothing new: no new version.
    """
    stored = stored_pmids(path)  # fails before any request when there is no library
    found_by, _found_per_topic = _search(topics, pubmed, progress)
    new = [pmid for pmid in found_by if pmid not in stored]
    if new:
        progress(f"fetching {len(new)} new paper records")
    papers = pubmed.fetch(new) if new else []
    folder = read_private_folder(private_folder)
    for name, reason in folder.skipped:
        progress(f"private folder: {name} skipped, {reason}")
    stamp = now().isoformat(timespec="seconds")
    change = write_refresh(path, papers, found_by, _fingerprint(topics), stamp, folder.papers)
    return RefreshReport(
        change.version,
        change.new_papers,
        change.private_added,
        change.private_removed,
        len(folder.papers),
        folder.skipped,
    )


def describe_refresh(report: RefreshReport, path: Path) -> str:
    private = f"{report.private_papers} private"
    if report.private_added or report.private_removed:
        private += f" ({report.private_added} added, {report.private_removed} removed)"
    if report.version is None:
        return f"nothing new for {path}: no new papers, {private} unchanged"
    return f"library version {report.version} at {path}: {report.new_papers} new papers, {private}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asclepius-library", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="download paper records from PubMed, then their full text")
    fulltext = commands.add_parser("fulltext", help="download the open-access full text still missing")
    refresh = commands.add_parser(
        "refresh", help="monthly update: add new papers and the private folder's PDFs, then their full text"
    )
    for command in (build, fulltext, refresh):
        command.add_argument(
            "--library", type=Path, default=DEFAULT_LIBRARY, help=f"SQLite file (default {DEFAULT_LIBRARY})"
        )
    for command in (build, refresh):
        command.add_argument(
            "--topics", type=Path, default=DEFAULT_TOPICS, help="topic searches (default: the shipped list)"
        )
        command.add_argument(
            "--private",
            type=Path,
            default=DEFAULT_PRIVATE,
            help=f"folder of your own PDFs, never sent anywhere (default {DEFAULT_PRIVATE})",
        )
    args = parser.parse_args(argv)

    say = lambda message: print(message, file=sys.stderr)  # noqa: E731
    downloading = args.command == "fulltext"
    try:
        api_key = os.environ.get("NCBI_API_KEY") or None
        pubmed = PubMed(http_transport(api_key), RateLimiter(rate_for(api_key)))
        if args.command == "build":
            topics = load_topics(args.topics)
            report = build_library(args.library, topics, pubmed, progress=say, private_folder=args.private)
            say(
                f"library version {report.version} at {args.library}: {report.papers} papers, "
                f"{report.lacking_abstract} without an abstract"
                + (f", {report.missing} found by a search but not fetchable" if report.missing else "")
            )
        elif args.command == "refresh":
            topics = load_topics(args.topics)
            say(
                describe_refresh(
                    refresh_library(args.library, topics, pubmed, args.private, progress=say), args.library
                )
            )
        downloading = True
        lookup = europepmc.EuropePMC(europepmc.http_transport(), RateLimiter(europepmc.RATE))
        text = fetch_full_texts(args.library, lookup, pubmed, progress=say)
        say(describe(text) + (f"; library version {text.version}" if text.version else ""))
    except KeyboardInterrupt:
        if downloading:
            say("asclepius-library: stopped; `asclepius-library fulltext` continues the download")
        else:
            say("asclepius-library: stopped before the build was stored; the library is as it was")
        return 130
    except (TopicsError, LibraryError, PubMedError, TransientError, europepmc.EuropePMCError) as error:
        say(f"asclepius-library: {error}")
        if downloading and not isinstance(error, LibraryError):
            say("what was downloaded is kept; `asclepius-library fulltext` continues")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
