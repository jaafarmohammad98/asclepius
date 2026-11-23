"""Hypotheses (M4-05b): an evidence bundle turned into the fixed five sections by the LLM.

The LLM is a fake local model server that answers with fixed replies, in turn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from test_evidence import built, where  # noqa: F401  (built is a fixture)
from test_search import WordEmbedder

from hypothesis_engine import build
from hypothesis_engine.evidence import EvidenceBundle, Passage, read_findings
from hypothesis_engine.hypotheses import PROMPT_VERSION, Failure, Hypothesis, generate_hypotheses
from hypothesis_engine.llm import LLMError, Settings, connect

FIXTURES = Path(__file__).parent / "fixtures"
LOCAL = Settings(backend="local", model="llama3.1:8b")

LOWER = Passage(
    "10:1", "10", False, "Neurosteroids in PSSD", "Androsterone was lower in men with PSSD.", "q", 0.03
)
HIGHER = Passage(
    "11:2", "11", False, "Androgens in PSSD", "Androsterone was higher in patients than controls.", "q", 0.02
)
PRIVATE = Passage(
    "private-ab12:1", "private-ab12", True, "My notes", "Unpublished androsterone data.", "q", 0.02
)


class FakeServer:
    """A local model server that answers each request with the next of `replies`."""

    def __init__(self, *replies: str | dict[str, Any]) -> None:
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.received: list[dict[str, Any]] = []

    def __call__(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        self.received.append(body)
        return {
            "model": "llama3.1:8b-instruct",
            "choices": [{"message": {"content": self.replies.pop(0)}, "finish_reason": "stop"}],
        }


def bundle(path: Path = FIXTURES / "findings_lab.csv", *passages: Passage) -> EvidenceBundle:
    finding = next(f for f in read_findings(path) if f.label == "suggestive")
    passages = passages or (LOWER, HIGHER)
    return EvidenceBundle(
        finding,
        (),
        tuple(p for p in passages if p is not HIGHER),
        tuple(p for p in passages if p is HIGHER),
        7,
        "bge-small",
    )


REPLY = {
    "mechanism": [
        {
            "claim": "Lower 5-alpha reductase activity lowers androsterone.",
            "passage_id": "10:1",
            "quote": "Androsterone was lower in men with PSSD.",
        }
    ],
    "supporting": [
        {"passage_id": "10:1", "quote": "Androsterone was lower in men with PSSD.", "why": "Same direction."}
    ],
    "contradicting": [
        {
            "passage_id": "11:2",
            "quote": "Androsterone was higher in patients than controls.",
            "why": "Opposite direction.",
        }
    ],
    "test": [{"design": "Measure urinary androsterone by GC-MS in new patients.", "measure": "lab test"}],
}


def generate(server: FakeServer, *bundles: EvidenceBundle) -> list[Hypothesis | Failure]:
    return generate_hypotheses(bundles or (bundle(),), connect(LOCAL, local_transport=server))


def test_a_well_formed_reply_gives_a_hypothesis_with_all_five_sections():
    [hypothesis] = generate(FakeServer(REPLY))
    assert isinstance(hypothesis, Hypothesis)
    finding = hypothesis.finding
    assert finding.label == "suggestive"
    assert finding.statement.startswith("Androsterone lower in PSSD than in healthy control (male)")
    assert [c.text for c in hypothesis.mechanism] == ["Lower 5-alpha reductase activity lowers androsterone."]
    assert [(c.paper_id, c.title) for c in hypothesis.supporting] == [("10", "Neurosteroids in PSSD")]
    assert [c.paper_id for c in hypothesis.contradicting] == ["11"]
    assert [t.measure for t in hypothesis.tests] == ["lab test"]
    assert list(hypothesis.as_dict()) == [
        "finding",
        "mechanism",
        "supporting",
        "contradicting",
        "test",
        "record",
    ]


def test_each_hypothesis_records_model_library_version_and_prompt_version():
    [hypothesis] = generate(FakeServer(REPLY))
    record = hypothesis.record
    assert record["model"] == "llama3.1:8b-instruct"  # the model that answered
    assert record["backend"] == "local"
    assert record["library_version"] == 7
    assert record["search_model"] == "bge-small"
    assert record["prompt_version"] == PROMPT_VERSION
    assert record["attempts"] == 1
    assert record["analysis_tool"] == "asclepius 0.1.0"


def test_a_malformed_reply_is_sent_back_with_what_was_wrong_and_retried():
    server = FakeServer("Here is a hypothesis: lower 5-alpha reductase.", REPLY)
    [hypothesis] = generate(server)
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.record["attempts"] == 2
    retry = server.received[1]["messages"]
    assert retry[-2] == {"role": "assistant", "content": "Here is a hypothesis: lower 5-alpha reductase."}
    assert "no JSON object" in retry[-1]["content"]


def test_a_reply_still_malformed_after_three_attempts_is_reported_as_failed():
    missing_test = {k: v for k, v in REPLY.items() if k != "test"}
    server = FakeServer("{not json}", missing_test, {**REPLY, "test": []})
    [failure] = generate(server)
    assert isinstance(failure, Failure)
    assert failure.record["attempts"] == 3
    assert len(failure.reasons) == 3
    assert "not valid JSON" in failure.reasons[0]
    assert "missing test" in failure.reasons[1]
    assert '"test" must not be empty' in failure.reasons[2]
    assert failure.as_dict()["finding"]["marker_id"] == "androsterone"


def test_a_citation_of_a_passage_not_in_the_bundle_is_malformed():
    invented = {**REPLY, "supporting": [{"passage_id": "99:1", "quote": "Made up.", "why": "Agrees."}]}
    [hypothesis] = generate(FakeServer(invented, REPLY))
    assert hypothesis.record["attempts"] == 2


def test_a_passage_filed_as_both_supporting_and_contradicting_is_malformed():
    twice = {**REPLY, "contradicting": REPLY["supporting"]}
    [failure] = generate(FakeServer(twice, twice, twice))
    assert isinstance(failure, Failure)
    assert "both supporting and contradicting" in failure.reasons[0]


def self_report_bundle() -> EvidenceBundle:
    return bundle(FIXTURES / "findings_self_report.csv")


SELF_REPORT_REPLY = {
    **REPLY,
    "test": [{"design": "Survey more people who took paroxetine.", "measure": "self-report"}],
}


def test_a_self_report_finding_says_so_first_in_section_one():
    [hypothesis] = generate(FakeServer(REPLY), self_report_bundle())
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.finding.labels[0] == "based on self-reports, no controls"
    assert "self-report, no controls" not in hypothesis.finding.labels
    assert hypothesis.finding.statement.startswith(
        "depression reported more often after paroxetine than after other causal drugs (all): 39% vs 17%"
    )


def test_a_self_report_hypothesis_tested_only_by_self_reports_is_rejected():
    server = FakeServer(SELF_REPORT_REPLY, REPLY)
    [hypothesis] = generate(server, self_report_bundle())
    assert hypothesis.record["attempts"] == 2
    assert "must not rely on self-reports" in server.received[1]["messages"][-1]["content"]
    assert "must not rely on self-reports" in server.received[0]["messages"][0]["content"]


def test_a_lab_finding_may_be_tested_by_self_reports():
    [hypothesis] = generate(FakeServer(SELF_REPORT_REPLY))
    assert isinstance(hypothesis, Hypothesis)


def test_the_prompt_holds_the_finding_and_public_passages_but_no_private_text():
    server = FakeServer(REPLY)
    generate(server, bundle(FIXTURES / "findings_lab.csv", LOWER, HIGHER, PRIVATE))
    sent = json.loads(server.received[0]["messages"][-1]["content"])
    assert sent["finding"]["marker_id"] == "androsterone"
    assert [p["passage_id"] for p in sent["passages"]] == ["10:1", "11:2"]
    assert "Unpublished" not in json.dumps(server.received)


def test_a_finding_with_nothing_to_cite_fails_without_asking_the_llm():
    server = FakeServer()
    [failure] = generate(server, bundle(FIXTURES / "findings_lab.csv", PRIVATE))
    assert isinstance(failure, Failure)
    assert server.received == []


def test_an_llm_error_reports_the_finding_as_failed_and_goes_on_to_the_next():
    def down(url: str, body: dict[str, Any]) -> dict[str, Any]:
        raise LLMError("no local model server answered")

    failure, other = generate_hypotheses([bundle(), bundle()], connect(LOCAL, local_transport=down))
    assert isinstance(failure, Failure) and isinstance(other, Failure)
    assert failure.reasons == ("no local model server answered",)


def test_a_reply_in_a_code_fence_is_read():
    [hypothesis] = generate(FakeServer(f"```json\n{json.dumps(REPLY)}\n```"))
    assert isinstance(hypothesis, Hypothesis)


def citing_the_first_passage(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """A model that answers any hypothesis request by citing the first passage it was given."""
    passage = json.loads(body["messages"][1]["content"])["passages"][0]
    quote = passage["text"].split(".")[0]
    reply = {
        "mechanism": [{"claim": "A claim.", "passage_id": passage["passage_id"], "quote": quote}],
        "supporting": [{"passage_id": passage["passage_id"], "quote": quote, "why": "Agrees."}],
        "contradicting": [],
        "test": [{"design": "Measure it again.", "measure": "lab test"}],
    }
    return {"model": "m", "choices": [{"message": {"content": json.dumps(reply)}}]}


def test_the_hypotheses_command_writes_hypotheses_and_failures_as_json(built, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setattr(build, "LocalEmbedder", lambda *_args, **_kwargs: WordEmbedder())
    monkeypatch.setattr(build, "load_settings", lambda *_args: LOCAL)
    monkeypatch.setattr(
        build,
        "connect",
        lambda settings, **_kwargs: connect(settings, local_transport=citing_the_first_passage),
    )
    out = tmp_path / "hypotheses.json"
    assert (
        build.main(["hypotheses", str(FIXTURES / "findings_lab.csv"), *where(built), "--out", str(out)]) == 0
    )
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [h["finding"]["marker_id"] for h in written["hypotheses"]] == [
        "androsterone",
        "androsterone/etiocholanolone",
    ]
    assert written["failed"] == []
    assert written["hypotheses"][0]["record"]["prompt_version"] == PROMPT_VERSION
    assert "2 hypotheses, 0 failed" in capsys.readouterr().err
