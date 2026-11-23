"""Citation verification (M4-06, spec H1): every quote is checked against the paper it cites.

The library is a real one built from the papers of `test_evidence`; the hypotheses are made
by hand, as an LLM might have written them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_evidence import built, where  # noqa: F401  (built is a fixture)
from test_hypotheses import FIXTURES, LOCAL, bundle
from test_search import WordEmbedder

from hypothesis_engine import build
from hypothesis_engine.citations import verify_hypotheses
from hypothesis_engine.hypotheses import Citation, Claim, Failure, Hypothesis, ProposedTest, finding_section
from hypothesis_engine.library import Library
from hypothesis_engine.llm import connect

REAL = "Androsterone was lower in men with PFS."  # paper 10's abstract opens with it


def claim(paper_id: str = "10", quote: str = REAL) -> Claim:
    return Claim("Lower 5-alpha reductase activity lowers androsterone.", f"{paper_id}:1", paper_id, quote)


def citation(paper_id: str = "10", quote: str = REAL) -> Citation:
    return Citation(f"{paper_id}:1", paper_id, "A title", quote, "Same direction.")


def hypothesis(
    mechanism: tuple[Claim, ...] | None = None,
    supporting: tuple[Citation, ...] | None = None,
    contradicting: tuple[Citation, ...] = (),
) -> Hypothesis:
    return Hypothesis(
        finding_section(bundle()),
        (claim(),) if mechanism is None else mechanism,
        (citation(),) if supporting is None else supporting,
        contradicting,
        (ProposedTest("Measure urinary androsterone by GC-MS.", "lab test"),),
        {"library_version": 1, "model": "m"},
    )


@pytest.fixture
def library(built):  # noqa: F811
    with Library.open(built[0]) as opened:
        yield opened


def test_a_hypothesis_whose_quotes_are_all_in_their_papers_is_kept_as_it_is(library):
    [kept] = verify_hypotheses([hypothesis()], library)
    assert isinstance(kept, Hypothesis)
    assert kept.mechanism == (claim(),)
    assert kept.supporting == (citation(),)


def test_a_claim_citing_a_paper_not_in_the_library_is_removed(library):
    [kept] = verify_hypotheses([hypothesis(mechanism=(claim(), claim("99999")))], library)
    assert kept.mechanism == (claim(),)


def test_a_claim_quoting_words_its_paper_does_not_hold_is_removed(library):
    invented = claim(quote="Androsterone was lower in every woman with PFS.")
    [kept] = verify_hypotheses([hypothesis(mechanism=(claim(), invented))], library)
    assert kept.mechanism == (claim(),)


@pytest.mark.parametrize(
    "quote",
    [
        "Androsterone  was\nlower in men",  # white space differs
        "Andro-\nsterone was lower in men",  # broken across a line
        "Neurosteroids in post finasteride syndrome",  # paper 10's title has "post-finasteride"
        "neurosteroids in postfinasteride syndrome",
    ],
)
def test_white_space_and_hyphenation_differences_do_not_count(library, quote):
    [kept] = verify_hypotheses([hypothesis(mechanism=(claim(quote=quote),))], library)
    assert isinstance(kept, Hypothesis)
    assert len(kept.mechanism) == 1


@pytest.mark.parametrize("quote", ["drosterone was lower in men", "Androsterone was lower in me"])
def test_a_quote_must_start_and_end_on_whole_words(library, quote):
    [result] = verify_hypotheses([hypothesis(mechanism=(claim(), claim(quote=quote)))], library)
    assert result.mechanism == (claim(),)


def test_supporting_and_contradicting_entries_are_checked_the_same_way(library):
    contradicting = citation("11", "Androsterone was higher in patients than in controls.")
    [kept] = verify_hypotheses(
        [
            hypothesis(
                supporting=(citation(), citation("99999"), citation(quote="Androsterone doubled.")),
                contradicting=(contradicting, citation("11", "Androsterone was higher in every patient.")),
            )
        ],
        library,
    )
    assert kept.supporting == (citation(),)
    assert kept.contradicting == (contradicting,)


def test_a_hypothesis_left_with_no_mechanism_claim_is_discarded_as_failed(library):
    [failure] = verify_hypotheses([hypothesis(mechanism=(claim("99999"), claim(quote="Made up.")))], library)
    assert isinstance(failure, Failure)
    assert failure.finding.marker_id == "androsterone"
    assert failure.record["model"] == "m"
    assert failure.reasons == (
        "a mechanism claim cites paper 99999, which the library does not hold",
        "a mechanism claim's quote is not found word for word in paper 10",
        "no mechanism claim is left once unverified citations are removed (spec H1)",
    )


def test_what_was_removed_is_recorded_without_the_unverified_quote(library):
    [kept] = verify_hypotheses([hypothesis(supporting=(citation(), citation(quote="Made up.")))], library)
    assert kept.record["model"] == "m"
    assert kept.record["citations_removed"] == [
        "a supporting entry's quote is not found word for word in paper 10"
    ]
    assert "Made up" not in str(kept.as_dict())


def test_a_hypothesis_with_nothing_removed_records_so(library):
    [kept] = verify_hypotheses([hypothesis()], library)
    assert kept.record["citations_removed"] == []


def test_a_failure_passes_through(library):
    failed = Failure(finding_section(bundle()), ("no local model server answered",), {})
    assert verify_hypotheses([failed], library) == [failed]


def inventing_quotes(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """A model citing real passages of the bundle with quotes it made up, and one invented paper."""
    passage = json.loads(body["messages"][1]["content"])["passages"][0]["passage_id"]
    reply = {
        "mechanism": [{"claim": "A claim.", "passage_id": passage, "quote": "Invented words entirely."}],
        "supporting": [{"passage_id": passage, "quote": "More invented words.", "why": "Agrees."}],
        "contradicting": [],
        "test": [{"design": "Measure it again.", "measure": "lab test"}],
    }
    return {"model": "m", "choices": [{"message": {"content": json.dumps(reply)}}]}


def test_the_hypotheses_command_writes_no_unverified_citation(built, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setattr(build, "LocalEmbedder", lambda *_args, **_kwargs: WordEmbedder())
    monkeypatch.setattr(build, "load_settings", lambda *_args: LOCAL)
    monkeypatch.setattr(
        build, "connect", lambda settings, **_kwargs: connect(settings, local_transport=inventing_quotes)
    )
    out = tmp_path / "hypotheses.json"
    findings = str(FIXTURES / "findings_lab.csv")
    assert build.main(["hypotheses", findings, *where(built), "--out", str(out)]) == 0
    written = out.read_text(encoding="utf-8")
    assert json.loads(written)["hypotheses"] == []
    assert len(json.loads(written)["failed"]) == 2
    assert "Invented" not in written
    assert "0 hypotheses, 2 failed" in capsys.readouterr().err


def test_a_claim_without_a_paper_identifier_or_a_quote_is_removed(library):
    [kept] = verify_hypotheses([hypothesis(mechanism=(claim(), claim(""), claim(quote=" ")))], library)
    assert kept.mechanism == (claim(),)
    assert kept.record["citations_removed"] == [
        "a mechanism claim carries no paper identifier",
        "a mechanism claim carries no quoted passage",
    ]
