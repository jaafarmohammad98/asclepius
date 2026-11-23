"""The hypotheses run shared by the command and the web app (M4-09): evidence, then the LLM, every
request checked by the private-text guard, with the passage and paper of every citation kept for
showing it."""

from __future__ import annotations

from functools import partial

import pytest
from test_hypotheses import FIXTURES, LOCAL, checking_mentions, citing_the_first_passage
from test_search import WordEmbedder

from hypothesis_engine.evidence import read_findings
from hypothesis_engine.hypotheses import Hypothesis
from hypothesis_engine.llm import connect
from hypothesis_engine.pipeline import run_hypotheses
from hypothesis_engine.privacy import PrivateTextLeak

LAB = FIXTURES / "findings_lab.csv"


CITING = checking_mentions(citing_the_first_passage)


def run(built, transport=CITING, **options):
    library, index = built
    return run_hypotheses(
        read_findings(LAB),
        LOCAL,
        library,
        index,
        WordEmbedder(),
        connect_llm=partial(connect, local_transport=transport),
        **options,
    )


def test_each_bundle_gets_its_result_in_order(built):
    made = run(built)
    assert len(made.results) == len(made.bundles) == 2
    assert [r.finding.marker_id for r in made.results] == [b.finding.marker_id for b in made.bundles]
    assert all(isinstance(r, Hypothesis) for r in made.results)
    assert (made.library_version, made.search_model) == (1, "test-words")


def test_every_cited_passage_is_kept_with_its_paper(built):
    made = run(built)
    cited = {c.passage_id for r in made.results for c in (*r.mechanism, *r.supporting)}
    assert cited and cited <= set(made.sources)
    source = made.sources["10:1"]
    assert source.paper_id == "10" and not source.private
    assert source.title == "Neurosteroids in post-finasteride syndrome"
    assert source.text.startswith("Androsterone was lower in men with PFS.")
    assert source.year == 2020


def test_progress_is_told_after_each_finding(built):
    told = []
    run(built, progress=lambda done, total: told.append((done, total)))
    assert told == [(0, 2), (1, 2), (2, 2)]


def test_a_request_with_private_text_stops_the_run(built, monkeypatch):
    from hypothesis_engine import pipeline

    class Refusing:
        def __init__(self, *_args):
            pass

        @classmethod
        def from_library(cls, _library):
            return cls()

        def check(self, _text):
            raise PrivateTextLeak("a request holding 70 words of private paper private-1")

    monkeypatch.setattr(pipeline, "PrivateTextGuard", Refusing)
    sent = []
    with pytest.raises(PrivateTextLeak):
        run(built, transport=lambda url, body: sent.append(body))
    assert sent == []
