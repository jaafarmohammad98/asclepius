"""Treatment candidates and their risks (M4-07, spec H3).

Every intervention a hypothesis names is a candidate for study with its known risks, each citation
verified, or "risks not established in the library"; every output holding one carries the notice.
The LLM is the fake server of `test_hypotheses`: the hypothesis reply first, then one reply per
candidate whose risks it is asked for.
"""

from __future__ import annotations

import json
from typing import Any

from test_evidence import SELF_REPORT, where
from test_hypotheses import FIXTURES, LOCAL, LOWER, REPLY, FakeServer, checking_mentions
from test_search import WordEmbedder

from hypothesis_engine import build
from hypothesis_engine.evidence import REPORTED, EvidenceBundle, Passage, is_intervention, read_findings
from hypothesis_engine.hypotheses import (
    CANDIDATE,
    MENTION_SYSTEM,
    NO_RISKS,
    NOTICE,
    Failure,
    Hypothesis,
    export,
    vocabulary,
)
from hypothesis_engine.llm import LLMError, connect

SEIZURES = "Seizures were reported with bupropion at high doses."  # paper 15 of the test library
RISK = {"risks": [{"risk": "Seizures at high doses.", "passage_id": "15:1", "quote": SEIZURES}]}
NAMING_BUPROPION = {
    **REPLY,
    "test": [{"design": "A trial of bupropion measuring androsterone.", "measure": "lab test"}],
    "candidates": [{"intervention": "bupropion"}],
}


def every_candidate(written: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for h in written["hypotheses"] for c in h["candidates"]]


def test_an_intervention_the_hypothesis_names_is_a_candidate_with_its_verified_risks(generate):
    server = FakeServer(NAMING_BUPROPION, RISK)
    [hypothesis] = generate(server)
    assert isinstance(hypothesis, Hypothesis)
    [candidate] = hypothesis.candidates
    assert (candidate.intervention, candidate.label) == ("bupropion", CANDIDATE)
    [risk] = candidate.risks
    assert (risk.paper_id, risk.quote, risk.title) == ("15", SEIZURES, "Bupropion safety")
    assert candidate.risks_note is None
    asked = json.loads(server.received[1]["messages"][1]["content"])
    assert asked["intervention"] == "bupropion"
    assert [p["passage_id"] for p in asked["passages"]] == ["15:1"]  # only passages naming it


def test_a_candidate_with_no_library_passage_has_risks_not_established(generate):
    named = {**REPLY, "candidates": [{"intervention": "unobtainium"}]}
    server = FakeServer(named)
    [hypothesis] = generate(server)
    [candidate] = hypothesis.candidates
    assert candidate.risks == ()
    assert candidate.risks_note == NO_RISKS
    assert len(server.received) == 1  # nothing to ask the LLM about


def test_a_risk_whose_quote_is_not_in_the_paper_is_removed(generate):
    invented = {
        "risks": [{"risk": "Liver damage.", "passage_id": "15:1", "quote": "Bupropion damaged the liver."}]
    }
    server = FakeServer(NAMING_BUPROPION, invented, invented, invented)
    [hypothesis] = generate(server)
    [candidate] = hypothesis.candidates
    assert candidate.risks == ()
    assert candidate.risks_note == NO_RISKS
    assert candidate.record["attempts"] == 3
    assert "not found word for word in paper 15" in candidate.record["risks_removed"][0]
    assert "damaged the liver" not in json.dumps(hypothesis.as_dict())
    assert "damaged the liver" in server.received[2]["messages"][-1]["content"]  # sent back to fix


def test_a_risk_quote_from_the_paper_but_not_the_passage_offered_is_removed(generate):
    # paper 15's title is in the paper, not in the passage naming bupropion: not this risk's evidence
    title = {"risks": [{"risk": "Unsafe.", "passage_id": "15:1", "quote": "Bupropion safety"}]}
    [hypothesis] = generate(FakeServer(NAMING_BUPROPION, title, title, title))
    [candidate] = hypothesis.candidates
    assert candidate.risks == ()
    assert "not found word for word in passage 15:1" in candidate.record["risks_removed"][0]


def test_a_risk_citing_a_passage_not_given_is_sent_back(generate):
    elsewhere = {"risks": [{"risk": "Something.", "passage_id": "10:1", "quote": "Androsterone was lower"}]}
    server = FakeServer(NAMING_BUPROPION, elsewhere, RISK)
    [hypothesis] = generate(server)
    [candidate] = hypothesis.candidates
    assert [r.paper_id for r in candidate.risks] == ["15"]
    assert "which was not given" in server.received[2]["messages"][-1]["content"]


def test_an_llm_error_on_the_risks_says_they_were_not_checked(generate):
    def answers_once(url: str, body: dict[str, Any]) -> dict[str, Any]:
        if "known risks" in body["messages"][0]["content"]:
            raise LLMError("no local model server answered")
        return {"model": "m", "choices": [{"message": {"content": json.dumps(NAMING_BUPROPION)}}]}

    [hypothesis] = generate(checking_mentions(answers_once))
    [candidate] = hypothesis.candidates
    assert (
        candidate.risks_note == "risks not checked: the LLM did not answer (no local model server answered)"
    )


def test_an_export_intervention_mentioned_but_not_listed_is_still_a_candidate(generate):
    mentioned = {
        **REPLY,
        "test": [{"design": "Compare androsterone before and after Bupropion.", "measure": "lab test"}],
    }
    [hypothesis] = generate(FakeServer(mentioned, RISK), interventions=("bupropion", "sildenafil"))
    assert [c.intervention for c in hypothesis.candidates] == ["bupropion"]  # sildenafil is not named


def test_a_causal_drug_of_the_finding_is_not_taken_for_a_candidate(generate):
    self_report = next(f for f in read_findings(SELF_REPORT) if f.label == "suggestive")  # after paroxetine
    mentioned = {
        **REPLY,
        "contradicting": [],
        "test": [{"design": "Measure androsterone after paroxetine.", "measure": "lab test"}],
    }
    bundle = EvidenceBundle(self_report, (), (LOWER,), (), 1, "bge-small")
    [hypothesis] = generate(FakeServer(mentioned), bundle, interventions=("paroxetine",))
    assert hypothesis.candidates == ()


def intervention_bundle(name: str = "bupropion") -> EvidenceBundle:
    finding = next(f for f in read_findings(SELF_REPORT) if f.marker_name == name)
    passage = Passage("15:1", "15", False, "Bupropion safety", SEIZURES, "q", 0.03)
    return EvidenceBundle(finding, (), (passage,), (), 1, "bge-small")


INTERVENTION_REPLY = {
    "mechanism": [{"claim": "It could act on seizure threshold.", "passage_id": "15:1", "quote": SEIZURES}],
    "supporting": [],
    "contradicting": [],
    "test": [{"design": "Measure prolactin in a trial.", "measure": "lab test"}],
    "candidates": [],
}


def test_an_intervention_finding_s_own_intervention_is_always_a_candidate(generate):
    server = FakeServer(INTERVENTION_REPLY, RISK)
    [hypothesis] = generate(server, intervention_bundle())
    assert isinstance(hypothesis, Hypothesis)
    assert [c.intervention for c in hypothesis.candidates] == ["bupropion"]
    finding = hypothesis.finding
    assert finding.statement.startswith(
        "bupropion: improvement reported by 29% (95% CI 18% to 44%) of 41 authors"
    )
    assert finding.statement.endswith(REPORTED)
    assert finding.labels[:2] == ("based on self-reports, no controls", REPORTED)
    assert "reported outcomes" in server.received[0]["messages"][0]["content"]


def test_an_unranked_intervention_gets_no_hypothesis(generate):
    server = FakeServer()
    [failure] = generate(server, intervention_bundle("tadalafil"))
    assert isinstance(failure, Failure)
    assert "unranked intervention" in failure.reasons[0]
    assert failure.as_dict()["notice"] == NOTICE  # it still names an intervention
    assert server.received == []


def test_every_export_with_a_candidate_carries_the_notice(generate):
    with_candidate, without = generate(FakeServer(NAMING_BUPROPION, RISK, REPLY), *[_lab(), _lab()])
    written = export([with_candidate, without])
    assert next(iter(written)) == "notice" and written["notice"] == NOTICE
    first, second = written["hypotheses"]
    assert first["notice"] == NOTICE and "notice" not in second
    assert export([without]) == {"hypotheses": [without.as_dict()], "failed": []}


def test_a_hypothesis_with_a_candidate_cannot_be_written_without_the_notice(generate):
    [hypothesis] = generate(FakeServer(NAMING_BUPROPION, RISK))
    written = hypothesis.as_dict()
    assert written["notice"] == NOTICE
    assert all(c["label"] == CANDIDATE for c in written["candidates"])  # type: ignore[union-attr]


def _lab() -> EvidenceBundle:
    from test_hypotheses import bundle

    return bundle()


def answering_both_prompts(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """A model that cites the first passage it is given, for hypotheses and for risks alike."""
    system, data = body["messages"][0]["content"], json.loads(body["messages"][1]["content"])
    passage = data["passages"][0]
    quote = passage["text"].split(".")[0]
    if "known risks" in system:
        reply: dict[str, Any] = {
            "risks": [{"risk": "A risk.", "passage_id": passage["passage_id"], "quote": quote}]
        }
    else:
        reply = {
            "mechanism": [{"claim": "A claim.", "passage_id": passage["passage_id"], "quote": quote}],
            "supporting": [],
            "contradicting": [],
            "test": [{"design": "Measure it in a trial.", "measure": "lab test"}],
            "candidates": [],
        }
    return {"model": "m", "choices": [{"message": {"content": json.dumps(reply)}}]}


def test_the_hypotheses_command_marks_candidates_and_writes_the_notice(built, tmp_path, monkeypatch):
    monkeypatch.setattr(build, "LocalEmbedder", lambda *_args, **_kwargs: WordEmbedder())
    monkeypatch.setattr(build, "load_settings", lambda *_args: LOCAL)
    monkeypatch.setattr(
        build,
        "connect",
        lambda settings, **_kwargs: connect(
            settings, local_transport=checking_mentions(answering_both_prompts)
        ),
    )
    out = tmp_path / "hypotheses.json"
    assert build.main(["hypotheses", str(SELF_REPORT), *where(built), "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["notice"] == NOTICE
    names = {c["intervention"] for c in every_candidate(written)}
    assert {"sildenafil", "bupropion"} <= names
    for candidate in every_candidate(written):
        assert candidate["label"] == CANDIDATE
        assert candidate["notice"] == NOTICE
        assert candidate["risks"] or candidate["risks_note"] == NO_RISKS
    bupropion = next(c for c in every_candidate(written) if c["intervention"] == "bupropion")
    assert [r["paper_id"] for r in bupropion["risks"]] == ["15"]


def test_a_lab_export_without_interventions_has_no_notice(built, tmp_path, monkeypatch):
    monkeypatch.setattr(build, "LocalEmbedder", lambda *_args, **_kwargs: WordEmbedder())
    monkeypatch.setattr(build, "load_settings", lambda *_args: LOCAL)
    monkeypatch.setattr(
        build,
        "connect",
        lambda settings, **_kwargs: connect(
            settings, local_transport=checking_mentions(answering_both_prompts)
        ),
    )
    out = tmp_path / "hypotheses.json"
    assert (
        build.main(["hypotheses", str(FIXTURES / "findings_lab.csv"), *where(built), "--out", str(out)]) == 0
    )
    assert "notice" not in json.loads(out.read_text(encoding="utf-8"))


def test_the_intervention_rows_of_the_fixture_are_read():
    assert [f.marker_name for f in read_findings(SELF_REPORT) if is_intervention(f)] == [
        "sildenafil",
        "bupropion",
        "tadalafil",
    ]


# --- the vocabulary and the second check: interventions named but not listed -----------------


def test_an_intervention_of_the_vocabulary_named_but_not_listed_is_a_candidate(generate):
    named = {
        **REPLY,
        "test": [{"design": "A trial of cabergoline measuring androsterone.", "measure": "lab test"}],
    }
    [hypothesis] = generate(FakeServer(named))  # nothing in the library names cabergoline: no risk request
    [candidate] = hypothesis.candidates
    assert (candidate.intervention, candidate.risks_note) == ("cabergoline", NO_RISKS)


def test_the_vocabulary_leaves_out_causal_drugs_and_body_chemicals():
    names = {n.casefold() for n in vocabulary()}
    assert {"bupropion", "cabergoline", "pde5 inhibitor", "psilocybin"} <= names
    assert not names & {
        "paroxetine",
        "sertraline",
        "finasteride",
        "isotretinoin",
        "estradiol",
        "gaba",
        "time",
    }


def test_an_intervention_only_the_second_check_finds_is_a_candidate(generate):
    named = {
        **REPLY,
        "test": [{"design": "Give compound XR-17 and measure androsterone.", "measure": "lab test"}],
    }
    server = FakeServer(named, mentions=({"interventions": ["compound XR-17"]},))
    [hypothesis] = generate(server)
    assert [c.intervention for c in hypothesis.candidates] == ["compound XR-17"]
    [check] = server.mention_requests
    sentences = json.loads(check["messages"][1]["content"])["sentences"]
    assert "Give compound XR-17 and measure androsterone." in sentences
    assert hypothesis.record["mention_check"]["prompt_version"] == "mentions-1"  # type: ignore[index]


def test_a_name_the_second_check_invents_is_sent_back_then_dropped(generate):
    invented = {"interventions": ["unobtainium"]}
    server = FakeServer(REPLY, mentions=(invented, invented, invented))
    [hypothesis] = generate(server)
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.candidates == ()
    assert len(server.mention_requests) == 3
    assert "not found word for word" in server.mention_requests[1]["messages"][-1]["content"]
    assert hypothesis.record["mention_check"]["names_dropped"] == ["unobtainium"]  # type: ignore[index]


def test_a_hypothesis_the_second_check_cannot_be_run_on_is_reported_as_failed(generate):
    server = FakeServer(REPLY, mentions=("not json", "not json", "not json"))
    [failure] = generate(server)
    assert isinstance(failure, Failure)
    assert "interventions it names did not finish" in failure.reasons[-1]


def test_an_llm_error_on_the_second_check_fails_the_hypothesis(generate):
    def down_on_check(url: str, body: dict[str, Any]) -> dict[str, Any]:
        if body["messages"][0]["content"] == MENTION_SYSTEM:
            raise LLMError("no local model server answered")
        return {"model": "m", "choices": [{"message": {"content": json.dumps(REPLY)}}]}

    [failure] = generate(down_on_check)
    assert isinstance(failure, Failure)
    assert "no local model server answered" in failure.reasons[-1]
