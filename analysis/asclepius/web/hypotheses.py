"""Hypotheses in the web app (milestone 4): the hypothesis engine run on the finished analysis run's
findings, in the background, finding by finding.

The engine (`hypothesis_engine.pipeline`) gathers library passages for each strong or suggestive
finding and has the LLM set in the settings screen (`asclepius.web.llm_settings`) write a hypothesis
in the five sections, every quote checked against its paper (spec H1) and every intervention named a
candidate for study with its known risks (H3). Every request passes the private-text guard first.

The view gives each hypothesis the finding it came from (`finding_ref`, and `key` for the page to
link the two), each citation the place of its quote in its passage (`span`), and every cited passage
once in `sources`, with links to its paper. The report is one HTML file with the run records: the
analysis run's, what wrote the hypotheses, and each hypothesis's own (spec H6).
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from hypothesis_engine.citations import quote_span
from hypothesis_engine.drugs import DEFAULT_DRUGS
from hypothesis_engine.embedding import DEFAULT_MODELS, LocalEmbedder, ModelMissing
from hypothesis_engine.evidence import EvidenceBundle, FindingsError, read_findings
from hypothesis_engine.evidence_log import LOG_PATH
from hypothesis_engine.hypotheses import Failure, Hypothesis, export
from hypothesis_engine.library import DEFAULT_LIBRARY, LibraryError
from hypothesis_engine.llm import DEFAULT_SETTINGS, LLMNotConfigured, connect, load_settings
from hypothesis_engine.pipeline import Connect, Generated, Source, run_hypotheses
from hypothesis_engine.privacy import PrivateTextLeak
from hypothesis_engine.search import DEFAULT_INDEX, Embedder, SearchIndexError
from hypothesis_engine.vocabulary import DEFAULT_VOCABULARY

from asclepius.run import PROVISIONAL
from asclepius.web.style import head as page_head

IDLE, RUNNING, FINISHED, FAILED = "idle", "running", "finished", "failed"
BUILD_LIBRARY = (
    "Build the paper library first: run `asclepius-library build`, then `asclepius-library index` "
    "(in a source checkout, `uv run` them in the hypothesis folder). The first build takes an hour or more."
)
REF = ("section", "comparison", "marker_id", "sex", "group_a", "group_b")
BUSY = "Hypotheses are being written; wait for them to finish."


class Busy(ValueError):
    """Hypotheses are being written: they cannot be started again, or forgotten, until they finish."""


@dataclass(frozen=True)
class Engine:
    """Where the hypothesis engine's files are, and how it reaches its models (tests replace these)."""

    library: Path = DEFAULT_LIBRARY
    index: Path = DEFAULT_INDEX
    models: Path = DEFAULT_MODELS
    settings: Path = DEFAULT_SETTINGS
    vocabulary: Path = DEFAULT_VOCABULARY  # the knowledge graph's terms (`asclepius.web.graph`)
    drugs: Path = DEFAULT_DRUGS  # the drug cache the leaderboard ranks (`asclepius.web.leaderboard`)
    evidence: Path = LOG_PATH  # the evidence log that weighs the terms (`asclepius.web.evidence_log`)
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    embedder: Callable[[Path], Embedder] = LocalEmbedder
    connect_llm: Connect = connect
    k: int = 5  # passages each way per finding


class HypothesisRun:
    """The latest hypotheses: written in the background from one analysis run's findings."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.reset()

    def reset(self) -> None:
        """Forget the hypotheses (a new dataset or run makes them another run's)."""
        with self._lock:
            if self.running():
                raise Busy(BUSY)
            self.state, self.message, self.done, self.total = IDLE, "", 0, 0
            self.generated: Generated | None = None
            self.analysis: dict[str, object] | None = None
            self.date: str | None = None

    def running(self) -> bool:
        """Whether hypotheses are being written now."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, findings_csv: str, analysis: dict[str, object], folder: Path) -> None:
        """Write hypotheses for the findings export `findings_csv` of the run `analysis` records."""
        with self._lock:
            if self.running():
                raise Busy(BUSY)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "findings.csv"
            path.write_text(findings_csv, encoding="utf-8")
            self.state, self.message, self.done, self.total = (
                RUNNING,
                "Gathering evidence from the library...",
                0,
                0,
            )
            self.generated, self.analysis = None, analysis
            self.date = datetime.now(UTC).isoformat(timespec="seconds")
            self._thread = threading.Thread(target=self._run, args=(path,), daemon=True)
            self._thread.start()

    def _progress(self, done: int, total: int) -> None:
        with self._lock:
            self.done, self.total = done, total
            self.message = f"Writing hypothesis {min(done + 1, total)} of {total}..." if done < total else ""

    def _run(self, path: Path) -> None:
        engine = self.engine
        try:
            settings = load_settings(engine.env, engine.settings)
            generated = run_hypotheses(
                read_findings(path),
                settings,
                engine.library,
                engine.index,
                engine.embedder(engine.models),
                k=engine.k,
                connect_llm=engine.connect_llm,
                progress=self._progress,
            )
        except LLMNotConfigured as error:
            state, message, generated = FAILED, f"{error} Set it up in Settings, above.", None
        except (LibraryError, SearchIndexError, ModelMissing) as error:
            state, message, generated = FAILED, f"{error}. {BUILD_LIBRARY}", None
        except PrivateTextLeak as error:
            state, message, generated = FAILED, f"Stopped before sending {error}; nothing was kept.", None
        except FindingsError as error:
            state, message, generated = FAILED, str(error), None
        except Exception as error:  # shown on the page rather than lost in the terminal
            state, message, generated = FAILED, f"Unexpected error: {error}", None
        else:
            made = sum(isinstance(r, Hypothesis) for r in generated.results)
            state = FINISHED
            message = (
                f"{len(generated.bundles)} of {generated.findings} findings worth testing (strong or "
                f"suggestive, or ranked interventions): {made} hypotheses, {len(generated.results) - made} failed."
            )
        with self._lock:
            self.state, self.message, self.generated = state, message, generated

    def status(self) -> dict[str, object]:
        """Their state (idle, running, finished, failed), a one-line message, and findings done of total."""
        with self._lock:
            return {"state": self.state, "message": self.message, "done": self.done, "total": self.total}

    def view(self) -> dict[str, object]:
        """The hypotheses as the page shows them (see the module docstring); none until finished."""
        with self._lock:
            generated, analysis, date = self.generated, self.analysis, self.date
            out: dict[str, object] = {
                "state": self.state,
                "message": self.message,
                "done": self.done,
                "total": self.total,
            }
        return out | hypotheses_view(generated, analysis, date)

    def report(self) -> str:
        """The finished hypotheses as one HTML file, with the run records."""
        view = self.view()
        if view["state"] != FINISHED:
            raise ValueError("Write the hypotheses, and let them finish, before exporting them.")
        with self._lock:
            generated, analysis = self.generated, self.analysis
        assert generated is not None and analysis is not None
        return report_html(view, analysis, export(generated.results))


def finding_ref(bundle: EvidenceBundle) -> dict[str, str | None]:
    """Which finding of the findings view a result is for: its section and its names."""
    return {k: getattr(bundle.finding, k) for k in REF}


def ref_key(ref: Mapping[str, str | None]) -> str:
    """One text for a finding's ref; the page builds the same from a finding row (`findingKey`)."""
    return "|".join(ref[k] or "" for k in REF)


def links(source: Source) -> list[dict[str, str]]:
    """Where to read the paper: PubMed, its open-access full text, its DOI. None for a private paper."""
    if source.private:
        return []
    found = [
        {"label": f"PubMed {source.paper_id}", "url": f"https://pubmed.ncbi.nlm.nih.gov/{source.paper_id}/"}
    ]
    if source.pmcid:
        found.append(
            {
                "label": f"Full text ({source.pmcid})",
                "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{source.pmcid}/",
            }
        )
    if source.doi:
        found.append({"label": f"DOI {source.doi}", "url": f"https://doi.org/{source.doi}"})
    return found


def _spanned(entries: list[dict[str, Any]], sources: Mapping[str, Source]) -> list[dict[str, Any]]:
    """Each cited entry with `span`: where its quote is in its passage, None if it is not there."""
    out = []
    for entry in entries:
        source = sources.get(entry["passage_id"])
        out.append(entry | {"span": quote_span(entry["quote"], source.text) if source else None})
    return out


def result_view(
    bundle: EvidenceBundle, result: Hypothesis | Failure, sources: Mapping[str, Source]
) -> dict[str, Any]:
    """A result as the engine writes it, with the finding it is for and its quotes' places."""
    ref = finding_ref(bundle)
    shown: dict[str, Any] = {"finding_ref": ref, "key": ref_key(ref), **result.as_dict()}
    if isinstance(result, Hypothesis):
        for section in ("mechanism", "supporting", "contradicting"):
            shown[section] = _spanned(shown[section], sources)
        shown["candidates"] = [c | {"risks": _spanned(c["risks"], sources)} for c in shown["candidates"]]
    return shown


def hypotheses_view(
    generated: Generated | None, analysis: Mapping[str, Any] | None, date: str | None
) -> dict[str, object]:
    """The page's view of a finished run (see the module docstring); empty before one."""
    if generated is None:
        return {
            "notice": None,
            "hypotheses": [],
            "failed": [],
            "sources": {},
            "record": None,
            "private_sent": [],
        }
    pairs = list(zip(generated.bundles, generated.results, strict=True))
    made = [result_view(b, r, generated.sources) for b, r in pairs if isinstance(r, Hypothesis)]
    failed = [result_view(b, r, generated.sources) for b, r in pairs if isinstance(r, Failure)]
    sources = {pid: {**asdict(s), "links": links(s)} for pid, s in generated.sources.items()}
    return {
        "notice": export(generated.results).get("notice"),
        "hypotheses": made,
        "failed": failed,
        "sources": sources,
        "record": generation_record(generated, analysis, date),
        # what of the researcher's private papers was sent: shown on this page only, never exported
        "private_sent": [
            {"text": q.text, "words": q.words, "papers": sorted(q.paper_ids)} for q in generated.quoted
        ],
    }


def generation_record(
    generated: Generated, analysis: Mapping[str, Any] | None, date: str | None
) -> dict[str, object]:
    """What wrote these hypotheses (spec H6), beside each hypothesis's own record."""
    settings = next((r.record.get("settings") for r in generated.results), None)
    return {
        "date": date,
        "library_version": generated.library_version,
        "library_built_at": generated.library_built_at,
        "search_model": generated.search_model,
        "settings": settings,
        "findings": generated.findings,
        "findings_worth_testing": len(generated.bundles),
        "analysis_tool": (analysis or {}).get("tool"),
        "dataset_fingerprint": ((analysis or {}).get("dataset") or {}).get("fingerprint"),
        "private_passages_sent": len(generated.quoted),
    }


# ---------------------------------------------------------------------------------- HTML report


def _e(value: object) -> str:
    return escape("" if value is None else str(value))


def _source_html(entry: Mapping[str, Any], sources: Mapping[str, Any]) -> str:
    """A citation: the quote, the paper with its links, and the passage with the quote marked."""
    source = sources.get(entry["passage_id"]) or {}
    where = ", ".join(_e(x) for x in (source.get("journal"), source.get("year")) if x)
    paper = f"{_e(entry.get('title') or source.get('title'))}" + (f" ({where})" if where else "")
    if source.get("private"):
        paper += " (a private paper on the researcher's computer)"
    anchors = " · ".join(
        f'<a href="{_e(link["url"])}">{_e(link["label"])}</a>' for link in source.get("links", [])
    )
    passage = ""
    if source.get("text") and not source.get("private"):  # a private paper's text never leaves this computer
        text, span = source["text"], entry.get("span")
        marked = (
            f"{_e(text[: span[0]])}<mark>{_e(text[span[0] : span[1]])}</mark>{_e(text[span[1] :])}"
            if span
            else _e(text)
        )
        passage = f"<details><summary>Passage {_e(entry['passage_id'])}</summary><p>{marked}</p></details>"
    return (
        f"<blockquote>“{_e(entry['quote'])}”</blockquote>"
        f'<div class="cite muted">{paper}{" · " + anchors if anchors else ""}</div>{passage}'
    )


def _hypothesis_html(h: Mapping[str, Any], sources: Mapping[str, Any]) -> str:
    """One hypothesis in its five sections, then its candidates and the citations removed."""
    finding = h["finding"]
    chips = "".join(f'<span class="label">{_e(label)}</span>' for label in finding["labels"])
    parts = [
        f"<article><h2>{_e(finding['statement'])}</h2>",
        f'<p class="notice">{_e(h["notice"])}</p>' if h.get("notice") else "",
        "<h3>1. The finding in the data</h3>",
        f"<p>{_e(finding['statement'])}</p><p>Strength: {_e(finding['label'] or 'not labelled')} {chips}</p>",
        f'<p class="muted">From {_e(finding["analysis_tool"])}, dataset {_e(finding["dataset_fingerprint"][:12])}…, '
        f"run {_e(finding['run_date'])}.</p>",
        "<h3>2. Proposed mechanism</h3><ol>",
        *(f"<li>{_e(c['text'])}{_source_html(c, sources)}</li>" for c in h["mechanism"]),
        "</ol>",
    ]
    for number, (key, title) in enumerate(
        (("supporting", "Supporting papers"), ("contradicting", "Contradicting papers")), 3
    ):
        parts.append(f"<h3>{number}. {title}</h3>")
        entries = h[key]
        parts.append(
            "<ul>" + "".join(f"<li>{_e(c['why'])}{_source_html(c, sources)}</li>" for c in entries) + "</ul>"
            if entries
            else '<p class="muted">None among the library passages found.</p>'
        )
    parts.append("<h3>5. How to test it</h3><ul>")
    parts += [
        f"<li>{_e(t['design'])} <span class='muted'>({_e(t['measure'])})</span></li>" for t in h["test"]
    ]
    parts.append("</ul>")
    if h["candidates"]:
        parts.append("<h3>Treatment candidates</h3><ul>")
        for c in h["candidates"]:
            risks = "".join(f"<li>{_e(r['text'])}{_source_html(r, sources)}</li>" for r in c["risks"])
            parts.append(
                f'<li><strong>{_e(c["intervention"])}</strong> <span class="label">{_e(c["label"])}</span>'
                + (f"<ul>{risks}</ul>" if risks else f'<p class="muted">{_e(c["risks_note"])}</p>')
                + "</li>"
            )
        parts.append("</ul>")
    removed = h["record"].get("citations_removed") or []
    if removed:
        parts.append(f'<p class="muted">Citations removed as unverified: {_e("; ".join(removed))}.</p>')
    return "".join(parts) + "</article>"


def report_html(view: Mapping[str, Any], analysis: Mapping[str, Any], exported: Mapping[str, Any]) -> str:
    """One self-contained HTML file: the notice when it applies, each hypothesis in its five sections
    with its sources, the findings that got none and why, and the run records, shown and embedded as
    JSON (`id="run-record"`). The private passages sent are counted, never shown, and a private
    paper's passage is cited by its quote only."""
    records = {"analysis": analysis, "hypotheses": view["record"], "export": exported}
    records_json = json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False)
    embedded = records_json.replace("<", "\\u003c")  # no "</script>" can end the block early
    record = view["record"]
    head = (
        "<dl>"
        f"<dt>Analysis</dt><dd>{_e(analysis['tool'])}, dataset {_e(analysis['dataset']['fingerprint'])}, run {_e(analysis.get('date'))}</dd>"
        f"<dt>Library</dt><dd>version {_e(record['library_version'])}, built {_e(record['library_built_at'])}; search model {_e(record['search_model'])}</dd>"
        f"<dt>LLM</dt><dd>{_e(json.dumps(record['settings']))}</dd>"
        f"<dt>Written</dt><dd>{_e(record['date'])}</dd>"
        "</dl>"
    )
    failed = ""
    if view["failed"]:
        failed = (
            "<h2>Findings with no hypothesis</h2><ul>"
            + "".join(
                f"<li>{_e(f['finding']['statement'])}<br><span class='muted'>{_e(f['reasons'][-1] if f['reasons'] else '')}</span></li>"
                for f in view["failed"]
            )
            + "</ul>"
        )
    notice = f'<p class="notice">{_e(view["notice"])}</p>' if view.get("notice") else ""
    return (
        page_head("Asclepius hypotheses")
        + f"{notice}<h1>Asclepius hypotheses</h1><span class='label'>{_e(PROVISIONAL)}</span>{head}"
        '<p class="muted">Each hypothesis is built from group-level findings only. Every quote was found '
        "word for word in the paper it cites; unverified citations were removed. These are ideas to "
        "test, not conclusions.</p>"
        + "".join(_hypothesis_html(h, view["sources"]) for h in view["hypotheses"])
        + failed
        + '<h2>Run records</h2><p class="muted">What these hypotheses were made from: the analysis run, '
        "the library version, the search model, the LLM and its settings (never the key), and each "
        "hypothesis's prompt version and attempts.</p>"
        f"<pre>{_e(records_json)}</pre>"
        f'<script type="application/json" id="run-record">{embedded}</script>'
        "</body></html>\n"
    )
