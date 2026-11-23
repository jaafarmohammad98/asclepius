"""Hypotheses in the web app (M4-09): written from the finished run's findings by the hypothesis
engine, each linked to its finding, every citation shown with its passage and a link to its paper,
and exported as an HTML report with the run records.

The library is a few made-up paper records indexed with a word-counting embedder, and the LLM a
fake local model server that cites the first passage it is given."""

import json
import re
import time
import zlib
from functools import partial
from html import unescape

import numpy as np
import pytest
from hypothesis_engine import hypotheses as engine_hypotheses
from hypothesis_engine.hypotheses import CANDIDATE, NOTICE, PROMPT_VERSION
from hypothesis_engine.library import write_build
from hypothesis_engine.llm import connect
from hypothesis_engine.pubmed import Paper
from hypothesis_engine.search import build_index
from test_web import FAST, add_lab, finished, lab_files

from asclepius import synthetic
from asclepius.web import hypotheses as hypotheses_module
from asclepius.web import llm_settings
from asclepius.web.hypotheses import Engine
from asclepius.web.workspace import RunError, Workspace

PAPERS = [
    Paper(
        "31000001",
        "Neurosteroids after SSRIs",
        "Steroid levels differed in post-SSRI sexual dysfunction compared with healthy controls. "
        "Androsterone and etiocholanolone were measured in urine.",
        "10.1000/neuro.1",
        "PMC9000001",
        "J Steroid Res",
        2022,
    ),
    Paper(
        "31000002",
        "No difference in steroids",
        "No difference in steroid levels was seen between post-SSRI sexual dysfunction and healthy controls.",
        None,
        None,
        "Clin Endo",
        2021,
    ),
    Paper(
        "31000003",
        "Bupropion safety",
        "Seizures were reported with bupropion at high doses.",
        None,
        None,
        None,
        2019,
    ),
]


class Words:
    """Each word a direction: enough meaning for the search to rank by shared words."""

    name = "test-words"

    def _vector(self, text):
        vector = np.zeros(256, dtype=np.float32)
        for word in re.findall(r"\w+", text.casefold()):
            vector[zlib.crc32(word.encode()) % 256] += 1
        return vector / (np.linalg.norm(vector) or 1)

    def passages(self, texts):
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, 256))

    def query(self, text):
        return self._vector(text)


def first_sentence(passage):
    return passage["text"].split(".")[0]


def model(candidate=None):
    """A local model server citing the first passage it is given; naming `candidate` if set."""

    def answer(url, body):
        if len(body["messages"]) == 1:  # the settings screen's check
            return {"model": "fake-1", "choices": [{"message": {"content": "ready"}}]}
        system, content = body["messages"][0]["content"], body["messages"][1]["content"]
        if system == engine_hypotheses.MENTION_SYSTEM:
            reply = {"interventions": []}
        else:
            passage = json.JSONDecoder().raw_decode(content)[0]["passages"][0]
            cited = {"passage_id": passage["passage_id"], "quote": first_sentence(passage)}
            if system == engine_hypotheses.RISK_SYSTEM:
                reply = {"risks": [{"risk": "A risk.", **cited}]}
            else:
                reply = {
                    "mechanism": [{"claim": "Steroid synthesis is changed.", **cited}],
                    "supporting": [{**cited, "why": "Agrees."}],
                    "contradicting": [],
                    "test": [{"design": "Measure it in a new cohort.", "measure": "lab test"}],
                    "candidates": [{"intervention": candidate}] if candidate else [],
                }
        message = {"content": json.dumps(reply)}
        return {"model": "fake-1", "choices": [{"message": message, "finish_reason": "stop"}]}

    return answer


def build_library(folder):
    """The test library and its search index, in `folder`."""
    found_by = {p.pmid: [("PSSD", rank)] for rank, p in enumerate(PAPERS, start=1)}
    write_build(folder / "library.sqlite", PAPERS, found_by, "sha", "2026-11-22T00:00:00+00:00", [])
    build_index(folder / "library.sqlite", folder / "index.sqlite", Words())
    return folder


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("library"))


def engine(folder, settings=None, transport=None):
    """The engine on the test library, with the settings file `settings` (local, by default)."""
    path = folder / "llm.toml"
    if settings is None:
        llm_settings.save(path, {"backend": "local", "local": {"model": "llama3.1:8b"}}, env={})
    return Engine(
        library=folder / "library.sqlite",
        index=folder / "index.sqlite",
        settings=path,
        env={},
        embedder=lambda _models: Words(),
        connect_llm=partial(connect, local_transport=transport or model()),
    )


def lab_workspace(tmp_path, folder, **options):
    workspace = Workspace(tmp_path / "work", engine=engine(folder, **options))
    add_lab(workspace, lab_files(tmp_path, planted=[synthetic.PlantedDifference("androsterone", 0.4)]))
    workspace.load()
    workspace.start_run("PSSD", "healthy control", settings=FAST)
    finished(workspace)
    return workspace


def written(workspace, timeout=60):
    """Start the hypotheses and wait for them; their view."""
    workspace.start_hypotheses()
    deadline = time.monotonic() + timeout
    while workspace.hypotheses_status()["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    return workspace.hypotheses_view()


@pytest.fixture(scope="module")
def lab(tmp_path_factory, library):
    workspace = lab_workspace(tmp_path_factory.mktemp("lab"), library)
    return workspace, written(workspace)


def test_hypotheses_are_written_for_the_strong_and_suggestive_findings(lab):
    workspace, view = lab
    assert view["state"] == "finished", view["message"]
    shown = [f for s in workspace.findings()["sections"] for f in s["findings"]]
    worth = [f for f in shown if f["label"] in ("strong", "suggestive")]
    assert worth and len(view["hypotheses"]) + len(view["failed"]) == len(worth)
    assert view["hypotheses"], view["failed"]
    assert (view["done"], view["total"]) == (len(worth), len(worth))


def test_each_hypothesis_names_the_finding_it_comes_from(lab):
    workspace, view = lab
    rows = {
        (s["kind"], f["comparison"], f["marker_id"], f.get("sex"), f.get("group_a"), f.get("group_b"))
        for s in workspace.findings()["sections"]
        for f in s["findings"]
    }
    for hypothesis in view["hypotheses"]:
        finding = hypothesis["finding_ref"]
        key = tuple(finding[k] for k in ("section", "comparison", "marker_id", "sex", "group_a", "group_b"))
        assert key in rows
        assert hypothesis["key"] == "|".join(v or "" for v in key)


def test_each_hypothesis_has_the_five_sections(lab):
    _, view = lab
    for hypothesis in view["hypotheses"]:
        assert hypothesis["finding"]["label"] in ("strong", "suggestive")
        assert hypothesis["mechanism"] and hypothesis["supporting"] and hypothesis["test"]
        assert hypothesis["contradicting"] == []


def test_every_citation_shows_its_quote_in_its_passage_and_links_to_its_paper(lab):
    _, view = lab
    claim = view["hypotheses"][0]["mechanism"][0]
    source = view["sources"][claim["passage_id"]]
    start, end = claim["span"]
    assert source["text"][start:end] == claim["quote"]
    assert source["title"] == PAPERS[0].title and (source["journal"], source["year"]) == (
        "J Steroid Res",
        2022,
    )
    assert source["links"] == [
        {"label": "PubMed 31000001", "url": "https://pubmed.ncbi.nlm.nih.gov/31000001/"},
        {"label": "Full text (PMC9000001)", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9000001/"},
        {"label": "DOI 10.1000/neuro.1", "url": "https://doi.org/10.1000/neuro.1"},
    ]


def test_the_view_records_what_wrote_the_hypotheses(lab):
    workspace, view = lab
    record = view["record"]
    assert record["library_version"] == 1 and record["search_model"] == "test-words"
    assert record["settings"] == {
        "backend": "local",
        "model": "llama3.1:8b",
        "max_tokens": 16000,
        "url": "http://localhost:11434/v1",
    }
    assert record["dataset_fingerprint"] == workspace.export_record()["dataset"]["fingerprint"]
    assert view["hypotheses"][0]["record"]["prompt_version"] == PROMPT_VERSION
    assert view["notice"] is None  # no candidate, no intervention


def test_the_report_has_each_hypothesis_its_sources_and_the_run_records(lab):
    workspace, view = lab
    html = workspace.export_hypotheses()
    text = unescape(html)
    claim = view["hypotheses"][0]["mechanism"][0]
    assert claim["quote"] in text and "https://pubmed.ncbi.nlm.nih.gov/31000001/" in text
    for heading in (
        "The finding in the data",
        "Proposed mechanism",
        "Supporting papers",
        "Contradicting papers",
        "How to test it",
    ):
        assert heading in text
    embedded = re.search(r'<script type="application/json" id="run-record">(.*?)</script>', html, re.S)
    records = json.loads(embedded.group(1))
    assert (
        records["analysis"]["dataset"]["fingerprint"] == workspace.export_record()["dataset"]["fingerprint"]
    )
    assert records["hypotheses"]["library_version"] == 1
    assert [h["record"]["prompt_version"] for h in records["export"]["hypotheses"]] == [PROMPT_VERSION] * len(
        view["hypotheses"]
    )


def test_a_candidate_is_labelled_with_its_risks_under_the_notice(tmp_path, library):
    workspace = lab_workspace(tmp_path, library, transport=model(candidate="bupropion"))
    view = written(workspace)
    [candidate, *_] = view["hypotheses"][0]["candidates"]
    assert candidate["label"] == CANDIDATE and candidate["risks"]
    risk = candidate["risks"][0]
    start, end = risk["span"]
    assert view["sources"][risk["passage_id"]]["text"][start:end] == risk["quote"]
    assert view["notice"] == NOTICE
    report = unescape(workspace.export_hypotheses())
    assert report.index(NOTICE) < report.index("Proposed mechanism") and CANDIDATE in report


def test_hypotheses_wait_for_a_finished_run(tmp_path, library):
    workspace = Workspace(tmp_path / "work", engine=engine(library))
    with pytest.raises(RunError, match="Run the analysis"):
        workspace.start_hypotheses()


def test_with_no_llm_set_up_the_run_says_to_open_the_settings(tmp_path, library):
    folder = tmp_path / "engine"
    folder.mkdir()
    for name in ("library.sqlite", "index.sqlite"):
        (folder / name).write_bytes((library / name).read_bytes())
    view = written(lab_workspace(tmp_path, folder, settings="none"))
    assert view["state"] == "failed" and "Settings" in view["message"]
    assert view["hypotheses"] == []


def test_without_a_library_the_run_says_how_to_build_one(tmp_path, library):
    workspace = lab_workspace(tmp_path, tmp_path)  # no library in this folder
    view = written(workspace)
    assert view["state"] == "failed" and "asclepius-library build" in view["message"]


def test_loading_again_forgets_the_hypotheses(lab, tmp_path, library):
    workspace = lab_workspace(tmp_path, library)
    written(workspace)
    workspace.load()
    assert workspace.hypotheses_status()["state"] == "idle"
    assert workspace.hypotheses_view()["hypotheses"] == []


def test_a_private_passage_is_quoted_in_the_report_but_its_text_stays_out(lab):
    workspace, view = lab
    claim = view["hypotheses"][0]["mechanism"][0]
    private_text = "Unpublished results from our clinic that must stay on this computer."
    view = view | {
        "sources": {
            claim["passage_id"]: {
                **view["sources"][claim["passage_id"]],
                "private": True,
                "text": private_text,
                "links": [],
            }
        }
    }
    html = hypotheses_module.report_html(view, workspace.export_record(), {"hypotheses": []})
    assert claim["quote"] in unescape(html) and private_text not in unescape(html)
    assert "a private paper" in html
