"""Citation checks (M4-06, spec H1): every quote is checked against the paper it cites.

How a quote is matched is tested on `quoted` alone. What happens to citations that fail is
tested through `generate_hypotheses`, against the library built from `test_evidence.PAPERS`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_evidence import where
from test_hypotheses import (
    FIXTURES,
    HIGHER,
    LOCAL,
    LOWER,
    REPLY,
    FakeServer,
    bundle,
)
from test_search import WordEmbedder

from hypothesis_engine import build
from hypothesis_engine.citations import quote_span, quoted
from hypothesis_engine.evidence import Passage
from hypothesis_engine.hypotheses import Failure, Hypothesis, generate_hypotheses
from hypothesis_engine.llm import LLMError, connect

TEXT = "Neurosteroids in post-finasteride syndrome. Androsterone was lower in men with PFS. 5α-reduc-\ntase"


@pytest.mark.parametrize(
    "quote",
    [
        "Androsterone was lower in men with PFS.",
        "androsterone was lower",  # case
        "Androsterone  was\nlower in men",  # white space
        "Andro-\nsterone was lower in men",  # broken across a line in the quote
        "5α-reductase",  # broken across a line in the paper
        "post finasteride syndrome",
        "postfinasteride syndrome",
        "post-finasteride syndrome",
        "Neuroﬁ",  # not a word of the text; see the next test for ligatures
    ][:-1],
)
def test_case_white_space_and_hyphenation_do_not_count(quote):
    assert quoted(quote, TEXT)


def test_a_ligature_from_pdf_text_reads_as_its_letters():
    assert quoted("finasteride", "post-ﬁnasteride")


@pytest.mark.parametrize(
    "quote",
    [
        "Androsterone was lower in every man with PFS.",  # invented words
        "drosterone was lower in men",  # starts inside a word
        "Androsterone was lower in me",  # ends inside a word
        "Andro sterone was lower",  # a word split that the text does not split
        "Androsterone waslower",  # two words joined
        "",
        " - ",
    ],
)
def test_a_quote_must_be_the_text_word_for_word(quote):
    assert not quoted(quote, TEXT)


@pytest.mark.parametrize(
    ("quote", "found"),
    [
        ("androsterone  was lower", "Androsterone was lower"),
        ("5α-reductase", "5α-reduc-\ntase"),
        ("post finasteride syndrome.", "post-finasteride syndrome."),
    ],
)
def test_the_span_is_where_the_text_holds_the_quote(quote, found):
    start, end = quote_span(quote, TEXT)
    assert TEXT[start:end] == found


def test_a_quote_not_in_the_text_has_no_span():
    assert quote_span("Androsterone was higher", TEXT) is None


def test_a_span_counts_characters_of_the_text_as_written():
    # "ﬁ" is one character in the text and two once read as letters: the span is the text's
    text = "post-ﬁnasteride lowered androsterone"
    start, end = quote_span("lowered androsterone", text)
    assert text[start:end] == "lowered androsterone"


# --- what happens to citations that fail ---------------------------------------------------

INVENTED = "Androsterone was lower in every woman with PFS."


def mechanism_quoting(quote: str) -> dict[str, Any]:
    return {**REPLY, "mechanism": [{**REPLY["mechanism"][0], "quote": quote}]}


def supporting_quoting(quote: str) -> dict[str, Any]:
    return {**REPLY, "supporting": [{**REPLY["supporting"][0], "quote": quote}]}


def test_a_reply_whose_quotes_are_all_found_is_not_sent_back(generate):
    server = FakeServer(REPLY)
    [hypothesis] = generate(server)
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.record["citations_removed"] == []
    assert len(server.received) == 1


def test_an_invented_quote_is_sent_back_and_a_corrected_reply_kept(generate):
    server = FakeServer(mechanism_quoting(INVENTED), REPLY)
    [hypothesis] = generate(server)
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.record["attempts"] == 2
    assert hypothesis.record["citations_removed"] == []
    sent_back = server.received[1]["messages"][-1]["content"]
    assert "a mechanism claim citing passage 10:1" in sent_back
    assert INVENTED in sent_back  # the LLM wrote it; it may see it again
    assert "not found word for word in paper 10" in sent_back
    assert "Copy each quote exactly" in sent_back


def test_a_real_paper_with_an_invented_quote_never_reaches_the_output(generate):
    server = FakeServer(*[supporting_quoting(INVENTED)] * 3)
    [hypothesis] = generate(server)
    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.record["attempts"] == 3
    assert hypothesis.supporting == ()  # kept: its mechanism claim is verified
    assert [c.paper_id for c in hypothesis.contradicting] == ["11"]
    assert hypothesis.record["citations_removed"] == [
        "a supporting entry citing passage 10:1 quotes words not found word for word in paper 10"
    ]
    assert "every woman" not in json.dumps(hypothesis.as_dict())


def test_a_fabricated_paper_identifier_never_reaches_the_output(generate):
    # an index out of step with the library could offer a passage of a paper it does not hold
    unknown = Passage("99999:1", "99999", False, "No such paper", "Androsterone was lower.", "q", 0.01)
    cites_it = {
        **REPLY,
        "mechanism": [*REPLY["mechanism"], {"claim": "C.", "passage_id": "99999:1", "quote": "Androsterone"}],
    }
    [hypothesis] = generate(
        FakeServer(*[cites_it] * 3), bundle(FIXTURES / "findings_lab.csv", LOWER, HIGHER, unknown)
    )
    assert isinstance(hypothesis, Hypothesis)
    assert [c.paper_id for c in hypothesis.mechanism] == ["10"]
    assert hypothesis.record["citations_removed"] == [
        "a mechanism claim citing passage 99999:1 cites paper 99999, which the library does not hold"
    ]


def test_a_hypothesis_left_with_no_mechanism_claim_is_reported_as_failed(generate):
    [failure] = generate(FakeServer(*[mechanism_quoting(INVENTED)] * 3))
    assert isinstance(failure, Failure)
    assert failure.record["attempts"] == 3
    assert failure.reasons[-1] == "no mechanism claim is left once unverified citations are removed (spec H1)"
    assert failure.reasons[0] == (
        "a mechanism claim citing passage 10:1 quotes words not found word for word in paper 10"
    )
    assert "every woman" not in json.dumps(failure.as_dict())


def test_a_retry_that_comes_back_worse_does_not_replace_the_better_reply(generate):
    both_invented = {**supporting_quoting(INVENTED), "mechanism": mechanism_quoting(INVENTED)["mechanism"]}
    [hypothesis] = generate(FakeServer(supporting_quoting(INVENTED), "{not json}", both_invented))
    assert isinstance(hypothesis, Hypothesis)  # reply 1: only its supporting entry failed
    assert [c.paper_id for c in hypothesis.mechanism] == ["10"]
    assert hypothesis.record["attempts"] == 3
    assert len(hypothesis.record["citations_removed"]) == 1


def test_of_two_replies_as_good_the_later_one_is_kept(generate):
    contradicting_invented = {
        **REPLY,
        "contradicting": [{**REPLY["contradicting"][0], "quote": "Androsterone was higher in all."}],
    }
    replies = (supporting_quoting(INVENTED), contradicting_invented, contradicting_invented)
    [hypothesis] = generate(FakeServer(*replies))
    assert [c.paper_id for c in hypothesis.supporting] == ["10"]
    assert hypothesis.contradicting == ()


def test_an_llm_error_on_a_retry_keeps_the_reply_already_given(generate, library, searcher):
    replies = iter([mechanism_quoting(INVENTED), REPLY])

    def then_down(url: str, body: dict[str, Any]) -> dict[str, Any]:
        reply = next(replies)
        if reply is REPLY:
            raise LLMError("no local model server answered")
        return {"model": "m", "choices": [{"message": {"content": json.dumps(reply)}}]}

    [failure] = generate_hypotheses([bundle()], connect(LOCAL, local_transport=then_down), library, searcher)
    assert isinstance(failure, Failure)  # the reply kept had no verified mechanism claim
    assert failure.reasons[1] == "no local model server answered"


def inventing_quotes(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """A model citing real passages of the bundle with quotes it made up, every time."""
    passage = json.loads(body["messages"][1]["content"])["passages"][0]["passage_id"]
    reply = {
        "mechanism": [{"claim": "A claim.", "passage_id": passage, "quote": "Invented words entirely."}],
        "supporting": [{"passage_id": passage, "quote": "More invented words.", "why": "Agrees."}],
        "contradicting": [],
        "test": [{"design": "Measure it again.", "measure": "lab test"}],
    }
    return {"model": "m", "choices": [{"message": {"content": json.dumps(reply)}}]}


def test_the_hypotheses_command_writes_no_unverified_citation(built, tmp_path, monkeypatch, capsys):
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
    assert "nvented" not in written
    assert "0 hypotheses, 2 failed" in capsys.readouterr().err
