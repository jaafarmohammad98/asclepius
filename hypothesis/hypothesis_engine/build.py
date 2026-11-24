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

After the full text, `build` fetches the approved drugs acting on the knowledge graph's top
bottleneck targets from ChEMBL (MyChem.info when ChEMBL fails) and caches them in the data folder
(`drugs.py`); `drugs` does only that, for the current version. The app reads the cache and never
fetches. Only the targets' gene symbols and ChEMBL ids are sent, through the privacy guard.

`index` builds the local search index of the current version (`search.py`), downloading the
embedding model the first time; `search` prints the passages most relevant to a question.
Both run without the network once the model is downloaded.

`evidence` reads an analysis findings export and writes, for each strong or suggestive
finding and each ranked intervention, the library passages that may support or contradict it
(`evidence.py`).

`hypotheses` (`pipeline.py`, shared with the web app) does what `evidence` does, then has the configured LLM turn each bundle into a
hypothesis in the fixed five sections (`hypotheses.py`), every quote checked against the paper
it cites (`citations.py`), every intervention named a candidate for study with its known risks,
and writes them with the findings it gave no usable hypothesis for, under the researchers-only
notice when any intervention is in the file. Every request it sends goes through the library's
`PrivateTextGuard` (`privacy.py`) first: one holding more private-paper text than short quotes
stops the run before it leaves, and nothing is written.

`llm-check` sends one short question to the configured LLM (`llm.py`) and prints which model
answered, so a researcher can check their key or local model before a long run.

Set NCBI_API_KEY to an NCBI API key (free, from an NCBI account) to send 10 requests a
second instead of 3. It is only ever sent to NCBI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hypothesis_engine import drugs, europepmc
from hypothesis_engine.embedding import DEFAULT_MODELS, LocalEmbedder, ModelMissing
from hypothesis_engine.evidence import FindingsError, gather_evidence, read_findings
from hypothesis_engine.fulltext import describe, fetch_full_texts
from hypothesis_engine.hypotheses import Failure, Hypothesis, export
from hypothesis_engine.library import (
    DEFAULT_LIBRARY,
    LibraryError,
    stored_pmids,
    write_build,
    write_refresh,
)
from hypothesis_engine.llm import DEFAULT_SETTINGS, LLMError, LLMNotConfigured, connect, load_settings
from hypothesis_engine.pipeline import run_hypotheses
from hypothesis_engine.privacy import PrivateTextLeak
from hypothesis_engine.private import read_private_folder
from hypothesis_engine.pubmed import (
    PubMed,
    PubMedError,
    RateLimiter,
    TransientError,
    http_transport,
    rate_for,
)
from hypothesis_engine.search import DEFAULT_INDEX, Index, SearchIndexError, build_index
from hypothesis_engine.topics import DEFAULT_TOPICS, Topic, TopicsError, load_topics
from hypothesis_engine.vocabulary import DEFAULT_VOCABULARY, VocabularyError, load_vocabulary

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
    """What a refresh changed in the library, and which private PDFs it could not read."""

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
        version=change.version,
        new_papers=change.new_papers,
        private_added=change.private_added,
        private_removed=change.private_removed,
        private_papers=len(folder.papers),
        skipped=folder.skipped,
    )


def describe_refresh(report: RefreshReport, path: Path) -> str:
    """One line for the researcher: what the refresh added, or that nothing was new."""
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
    drugs_command = commands.add_parser(
        "drugs", help="fetch the approved drugs acting on the top bottleneck targets, from ChEMBL"
    )
    index = commands.add_parser(
        "index", help="build the local search index of the current version (downloads the model once)"
    )
    search = commands.add_parser("search", help="print the library passages most relevant to a question")
    search.add_argument("query", help="the question or finding to search for")
    search.add_argument("-k", type=int, default=10, help="how many passages (default 10)")
    evidence = commands.add_parser(
        "evidence", help="gather library passages for each strong or suggestive finding of an analysis export"
    )
    evidence.add_argument("findings", type=Path, help="the analysis web app's findings.csv")
    evidence.add_argument(
        "--out", type=Path, required=True, help="JSON file to write the evidence bundles to"
    )
    evidence.add_argument("-k", type=int, default=5, help="passages each way per finding (default 5)")
    hypotheses = commands.add_parser(
        "hypotheses", help="have the configured LLM write a hypothesis for each strong or suggestive finding"
    )
    hypotheses.add_argument("findings", type=Path, help="the analysis web app's findings.csv")
    hypotheses.add_argument("--out", type=Path, required=True, help="JSON file to write the hypotheses to")
    hypotheses.add_argument("-k", type=int, default=5, help="passages each way per finding (default 5)")
    llm_check = commands.add_parser("llm-check", help="ask the configured LLM one short question")
    for command in (hypotheses, llm_check):
        command.add_argument(
            "--settings",
            type=Path,
            default=DEFAULT_SETTINGS,
            help=f"LLM settings file (default {DEFAULT_SETTINGS})",
        )
    for command in (index, search, evidence, hypotheses):
        command.add_argument(
            "--index", type=Path, default=DEFAULT_INDEX, help=f"search index file (default {DEFAULT_INDEX})"
        )
        command.add_argument(
            "--models",
            type=Path,
            default=DEFAULT_MODELS,
            help=f"embedding model folder (default {DEFAULT_MODELS})",
        )
    for command in (build, drugs_command):
        command.add_argument(
            "--drugs",
            type=Path,
            default=drugs.DEFAULT_DRUGS,
            help=f"drug cache (default {drugs.DEFAULT_DRUGS})",
        )
        command.add_argument(
            "--vocabulary",
            type=Path,
            default=DEFAULT_VOCABULARY,
            help="the knowledge graph's terms (default: the shipped list)",
        )
    for command in (build, fulltext, refresh, index, search, evidence, hypotheses, drugs_command):
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
    if args.command == "llm-check":
        return _llm_check(args.settings, say)
    if args.command in ("index", "search"):
        return _index_or_search(args, say)
    if args.command == "evidence":
        return _evidence(args, say)
    if args.command == "hypotheses":
        return _hypotheses(args, say)
    if args.command == "drugs":
        return _drugs(args, say)
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
        full_text = fetch_full_texts(args.library, lookup, pubmed, progress=say)
        say(describe(full_text) + (f"; library version {full_text.version}" if full_text.version else ""))
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
    if args.command == "build":
        return _drugs(args, say)
    return 0


def _drugs(args: argparse.Namespace, say: Callable[[str], None]) -> int:
    """Fetch and cache the drugs; a failure leaves the previous cache as it was."""
    try:
        cache = drugs.update_drugs(
            args.library,
            load_vocabulary(args.vocabulary),
            args.drugs,
            drugs.http_transport(drugs.CHEMBL),
            drugs.http_transport(drugs.MYCHEM),
            progress=say,
        )
    except KeyboardInterrupt:
        say(
            "asclepius-library: stopped; the drug cache is as it was, `asclepius-library drugs` fetches it again"
        )
        return 130
    except PrivateTextLeak as error:
        say(f"asclepius-library: stopped before sending {error}; the drug cache is as it was")
        return 1
    except (LibraryError, VocabularyError, drugs.DrugsError, TransientError) as error:
        say(f"asclepius-library: {error}")
        say("the drug cache is as it was; `asclepius-library drugs` fetches it again")
        return 1
    say(
        f"drug cache at {args.drugs}: {len(cache.ranked)} approved drugs on {len(cache.targets)} targets, "
        f"{len(cache.excluded)} known causes excluded"
    )
    return 0


def _index_or_search(args: argparse.Namespace, say: Callable[[str], None]) -> int:
    try:
        if args.command == "index":
            report = build_index(args.library, args.index, LocalEmbedder(args.models, download=True), say)
            say(
                f"search index of library version {report.version} at {args.index}: "
                f"{report.passages} passages, {report.embedded} newly embedded"
            )
            return 0
        with Index.open(args.index, args.library, LocalEmbedder(args.models)) as index:
            hits = index.search(args.query, args.k)
        for rank, hit in enumerate(hits, start=1):
            private = " (private)" if hit.private else ""
            print(f"{rank}. {hit.passage_id}{private}  {hit.title}")
            print(f"   {' '.join(hit.text.split())[:300]}")
        if not hits:
            say("nothing found")
    except KeyboardInterrupt:
        say("asclepius-library: stopped; the search index is as it was")
        return 130
    except (LibraryError, SearchIndexError, ModelMissing) as error:
        say(f"asclepius-library: {error}")
        return 1
    return 0


def _evidence(args: argparse.Namespace, say: Callable[[str], None]) -> int:
    try:
        findings = read_findings(args.findings)
        with Index.open(args.index, args.library, LocalEmbedder(args.models)) as index:
            bundles = gather_evidence(findings, index, args.k)
    except KeyboardInterrupt:
        say("asclepius-library: stopped; nothing was written")
        return 130
    except (FindingsError, LibraryError, SearchIndexError, ModelMissing) as error:
        say(f"asclepius-library: {error}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps([b.as_dict() for b in bundles], indent=2, ensure_ascii=False, allow_nan=False)
    args.out.write_text(text + "\n", encoding="utf-8")
    say(
        f"{len(bundles)} of {len(findings)} findings worth testing (strong or suggestive, or ranked interventions), "
        f"evidence written to {args.out}"
    )
    return 0


def _hypotheses(args: argparse.Namespace, say: Callable[[str], None]) -> int:
    try:
        settings = load_settings(os.environ, args.settings)
        findings = read_findings(args.findings)
        made = run_hypotheses(
            findings,
            settings,
            args.library,
            args.index,
            LocalEmbedder(args.models),
            k=args.k,
            connect_llm=connect,
        )
    except KeyboardInterrupt:
        say("asclepius-library: stopped; nothing was written")
        return 130
    except PrivateTextLeak as error:
        say(f"asclepius-library: stopped before sending {error}; nothing was written")
        return 1
    except (LLMNotConfigured, FindingsError, LibraryError, SearchIndexError, ModelMissing) as error:
        say(f"asclepius-library: {error}")
        return 1
    hypotheses = [r for r in made.results if isinstance(r, Hypothesis)]
    failed = [r for r in made.results if isinstance(r, Failure)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(export(made.results), indent=2, ensure_ascii=False, allow_nan=False)
    args.out.write_text(text + "\n", encoding="utf-8")
    say(
        f"{len(made.bundles)} of {len(findings)} findings worth testing (strong or suggestive, or ranked interventions): "
        f"{len(hypotheses)} hypotheses, {len(failed)} failed, written to {args.out}"
    )
    for failure in failed:
        say(f"  failed: {failure.finding.statement}: {failure.reasons[-1]}")
    for quote in made.quoted:
        say(f"  sent from {', '.join(sorted(quote.paper_ids))} ({quote.words} words): {quote.text}")
    return 0


def _llm_check(settings_path: Path, say: Callable[[str], None]) -> int:
    try:
        llm = connect(load_settings(os.environ, settings_path))
        completion = llm.complete([{"role": "user", "content": "Reply with the one word: ready"}])
    except (LLMNotConfigured, LLMError) as error:
        say(f"asclepius-library: {error}")
        return 1
    say(f"{completion.backend} model {completion.model} answered: {completion.text.strip()[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
