"""Privacy guard (M4-08, spec H5): no individual patient record is ever sent to an LLM.

The hypotheses command runs on the findings the analysis app exported from two synthetic
datasets (`tests/fixtures/privacy/`, written by `privacy_fixture.py`), with a library that holds
a private paper. The request hook sees every request just before it would leave; the model is a
fake that answers every kind of request (hypothesis, interventions check, risks) usably, so each
kind is sent. Every request is then searched for the datasets' record ids and individual values,
and for the private paper's text.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_evidence import PAPERS
from test_search import WordEmbedder

from hypothesis_engine import build, hypotheses
from hypothesis_engine.evidence import EvidenceBundle, gather_evidence
from hypothesis_engine.library import write_build
from hypothesis_engine.llm import LLM, OutgoingRequest, Settings, connect
from hypothesis_engine.privacy import PrivateTextGuard
from hypothesis_engine.private import PrivatePaper
from hypothesis_engine.search import build_index

PRIVACY = Path(__file__).parent / "fixtures" / "privacy"
LOCAL = Settings(backend="local", model="llama3.1:8b")
FINDINGS = {"lab": PRIVACY / "lab_findings.csv", "self_report": PRIVACY / "self_report_findings.csv"}

PRIVATE = PrivatePaper(
    "private-00c0ffee00c0ffee",
    "unpublished.pdf",
    "Unpublished cohort of androsterone in PSSD",
    "Androsterone was lower in our unpublished cohort of patients with PSSD and with PFS. "
    "The cohort was followed by the clinic for eleven months after the drug was stopped. "
    "Etiocholanolone, bupropion and paroxetine were noted in the charts of most of them. "
    "Two of the men asked that their results be kept out of any publication until the second "
    "sample was taken, and the clinic agreed to hold the whole series back until then.",
)  # 74 words: longer than one quote may be


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        yield from csv.DictReader(f)


def individual_values() -> dict[str, str]:
    """Every record id and individual value of the two datasets -> where it comes from."""
    found: dict[str, str] = {}

    def add(value: str, where: str, number: bool = False) -> None:
        if value:
            found[value] = where
            if number:
                found[repr(float(value))] = where  # as json.dumps would write the number

    for row in _rows(PRIVACY / "lab" / "metadata.csv"):
        add(row["record_id"], "lab record id")
    for row in _rows(PRIVACY / "lab" / "reports.csv"):
        add(row["value"], f"lab value of {row['record_id']}", number=True)
    for row in _rows(PRIVACY / "self_report" / "cases.csv"):
        add(row["case_id"], "self-report case id")
        add(row["author_key"], "self-report author key")
    for row in _rows(PRIVACY / "self_report" / "labs.csv"):
        add(row["value"], f"self-report lab value of {row['case_id']}", number=True)
    return found


INDIVIDUAL = individual_values()


def leaks(requests: list[OutgoingRequest]) -> list[str]:
    """The record ids and individual values that appear in any of `requests`."""
    return sorted(
        f"{value} ({where})" for r in requests for value, where in INDIVIDUAL.items() if value in r.text()
    )


def fake_model(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """A local model server that answers every kind of request with a usable reply."""
    system = body["messages"][0]["content"]
    content = body["messages"][1]["content"]
    # the JSON the prompt starts with (a test may add text after it)
    data = json.JSONDecoder().raw_decode(content)[0] if system != hypotheses.MENTION_SYSTEM else {}
    passages = data.get("passages", [])
    first = passages[0] if passages else {"passage_id": "", "text": ""}
    quote = first["text"].split(".")[0]
    if system == hypotheses.MENTION_SYSTEM:
        reply: dict[str, Any] = {"interventions": []}
    elif system == hypotheses.RISK_SYSTEM:
        reply = {"risks": [{"risk": "A risk.", "passage_id": first["passage_id"], "quote": quote}]}
    else:
        reply = {
            "mechanism": [{"claim": "A claim.", "passage_id": first["passage_id"], "quote": quote}],
            "supporting": [{"passage_id": first["passage_id"], "quote": quote, "why": "Agrees."}],
            "contradicting": [],
            "test": [{"design": "Measure it again in a new cohort.", "measure": "lab test"}],
            "candidates": [],
        }
    return {"model": "m", "choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}]}


@pytest.fixture
def library_files(tmp_path: Path) -> list[str]:
    """The test library (`test_evidence.PAPERS`) with PRIVATE in its private folder."""
    library = tmp_path / "library.sqlite"
    found_by = {p.pmid: [("PSSD", rank)] for rank, p in enumerate(PAPERS, start=1)}
    write_build(library, PAPERS, found_by, "sha", "2026-11-22T00:00:00+00:00", [PRIVATE])
    build_index(library, tmp_path / "index.sqlite", WordEmbedder())
    return ["--library", str(library), "--index", str(tmp_path / "index.sqlite")]


@dataclass
class Run:
    """What one run of the hypotheses command did."""

    code: int  # its exit code
    seen: list[OutgoingRequest]  # the requests the hook saw, refused ones too
    received: list[dict[str, Any]]  # the request bodies the model got
    bundles: list[EvidenceBundle]  # the evidence gathered, before any prompt
    printed: str


@pytest.fixture
def run_hypotheses(
    library_files: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> Any:
    """`run_hypotheses(kind)`: the hypotheses command on that dataset's findings, as a `Run`."""

    def go(kind: str) -> Run:
        done = Run(0, [], [], [], "")

        def connecting(settings: Settings, on_request: Any = None, **_kwargs: Any) -> LLM:
            def hook(request: OutgoingRequest) -> None:
                done.seen.append(request)
                if on_request is not None:
                    on_request(request)  # the command's own checks, which may refuse it

            def model(url: str, body: dict[str, Any]) -> dict[str, Any]:
                done.received.append(body)
                return fake_model(url, body)

            return connect(settings, on_request=hook, local_transport=model)

        def gathering(*args: Any, **kwargs: Any) -> list[EvidenceBundle]:
            done.bundles.extend(gather_evidence(*args, **kwargs))
            return done.bundles

        monkeypatch.setattr(build, "LocalEmbedder", lambda *_args, **_kwargs: WordEmbedder())
        monkeypatch.setattr(build, "load_settings", lambda *_args: LOCAL)
        monkeypatch.setattr(build, "connect", connecting)
        monkeypatch.setattr(build, "gather_evidence", gathering)
        out = tmp_path / f"{kind}.json"
        done.code = build.main(["hypotheses", str(FINDINGS[kind]), *library_files, "--out", str(out)])
        done.printed = capsys.readouterr().err
        return done

    return go


def leak_into_prompt(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Every hypothesis prompt from now on ends with `text`."""
    real = hypotheses.prompt

    def leaking(bundle: Any, passages: Any) -> Any:
        messages = real(bundle, passages)
        messages[-1]["content"] += "\n" + text
        return messages

    monkeypatch.setattr(hypotheses, "prompt", leaking)


def test_the_fixtures_individual_values_are_distinctive():
    """No group statistic in the findings export equals a record's value by chance, so a value
    found in a request was sent as itself."""
    exported = "".join(path.read_text(encoding="utf-8") for path in FINDINGS.values())
    assert len(INDIVIDUAL) > 2000
    assert [v for v in INDIVIDUAL if v in exported] == []


SYSTEMS = {
    hypotheses.MENTION_SYSTEM: "interventions check",
    hypotheses.RISK_SYSTEM: "risks",
}


@pytest.mark.parametrize(
    ("kind", "requests"),
    [
        ("lab", {"hypothesis", "interventions check"}),  # no intervention: no risks asked for
        ("self_report", {"hypothesis", "interventions check", "risks"}),
    ],
)
def test_no_record_id_or_individual_value_is_in_any_request(run_hypotheses, kind, requests):
    done = run_hypotheses(kind)
    assert done.code == 0
    system = [r.body["messages"][0]["content"] for r in done.seen]
    assert {SYSTEMS.get(s, "hypothesis") for s in system} == requests, "every kind of request was sent"
    assert leaks(done.seen) == []


def test_a_record_value_added_to_a_prompt_is_caught(run_hypotheses, monkeypatch):
    """The check above fails if a prompt carries one record-level value."""
    record_id, value = "pt-001-x", next(_rows(PRIVACY / "lab" / "reports.csv"))["value"]
    leak_into_prompt(monkeypatch, f"One patient: {record_id}, {value}")
    found = leaks(run_hypotheses("lab").seen)
    assert f"{record_id} (lab record id)" in found
    assert f"{value} (lab value of pt-001-x)" in found


@pytest.mark.parametrize("kind", ["lab", "self_report"])
def test_no_private_paper_text_is_in_any_request(run_hypotheses, kind):
    """The private paper is found as evidence, but private passages are not offered to the LLM
    (README), so no run of its words is sent."""
    done = run_hypotheses(kind)
    found = [p for b in done.bundles for p in (*b.supporting, *b.contradicting) if p.paper_id == PRIVATE.id]
    assert found, "the private paper was among the evidence"
    guard = PrivateTextGuard([PRIVATE], max_quote_words=0)
    assert [q for r in done.seen for q in guard.check(r.text())] == []


def test_a_request_with_private_text_beyond_short_quotes_is_refused_before_it_leaves(
    run_hypotheses, monkeypatch, tmp_path
):
    leak_into_prompt(monkeypatch, PRIVATE.text)
    done = run_hypotheses("lab")
    assert done.code == 1
    assert len(done.seen) == 1 and done.received == [], "refused by the hook: the model never got it"
    assert "private paper" in done.printed
    assert not (tmp_path / "lab.json").exists()
