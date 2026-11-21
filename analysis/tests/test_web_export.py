"""Export (M3-10c): the findings as CSV and a report as HTML, each with the tool and dataset version."""

import json
from html import escape

import pytest
from test_web import FAST, add_export, add_lab, finished, lab_files

from asclepius import synthetic
from asclepius.record import tool_version
from asclepius.web.export import read_findings_csv
from asclepius.web.workspace import RunError, Workspace


@pytest.fixture(scope="module")
def lab_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("lab")
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path, planted=[synthetic.PlantedDifference("androsterone", 0.4)]))
    workspace.load()
    workspace.start_run("PSSD", "healthy control", settings=FAST)
    finished(workspace)
    return workspace


@pytest.fixture(scope="module")
def self_report_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sr")
    workspace = Workspace(tmp_path / "work")
    add_export(workspace, synthetic.write_self_reports(tmp_path / "sr", authors=300, seed=1).export)
    workspace.load()
    workspace.start_run(settings=FAST)
    finished(workspace)
    return workspace


def shown_findings(view):
    """Every finding the page shows, without its plot, with the section it is in."""
    for s in view["sections"]:
        for f in s["findings"]:
            yield s["kind"], {k: v for k, v in f.items() if k != "plot"}


@pytest.mark.parametrize("run", ["lab_run", "self_report_run"])
def test_the_csv_export_reloads_without_loss(run, request):
    workspace = request.getfixturevalue(run)
    rows = read_findings_csv(workspace.export_csv())
    expected = list(shown_findings(workspace.findings()))
    assert expected, "the run has findings to export"
    assert len(rows) == len(expected)
    for row, (kind, finding) in zip(rows, expected, strict=True):
        assert row["section"] == kind
        for key, value in finding.items():
            if key == "zones":
                assert {z: row[f"zone_{z}"] for z in value} == value
            else:
                assert row[key] == value, key
                assert type(row[key]) is type(value), key


@pytest.mark.parametrize("run", ["lab_run", "self_report_run"])
def test_every_csv_row_names_the_tool_and_dataset_versions_and_its_labels(run, request):
    workspace = request.getfixturevalue(run)
    rows = read_findings_csv(workspace.export_csv())
    for row in rows:
        assert row["tool"] == tool_version()
        assert row["dataset_fingerprint"] == workspace.data.fingerprint
        assert "provisional, needs expert review" in row["labels"]
        is_self_report = run == "self_report_run"
        assert ("self-report, no controls" in row["labels"]) == is_self_report


def report_record(html):
    """The run record embedded in a report, as a program reads it back."""
    start = html.index('<script type="application/json" id="run-record">') + len(
        '<script type="application/json" id="run-record">'
    )
    return json.loads(html[start : html.index("</script>", start)])


@pytest.mark.parametrize("run", ["lab_run", "self_report_run"])
def test_the_report_contains_the_run_record_versions_and_labels(run, request):
    workspace = request.getfixturevalue(run)
    html = workspace.export_report()
    record = report_record(html)
    assert record == workspace.run_record()
    assert record["tool"] == tool_version()
    assert record["dataset"]["fingerprint"] == workspace.data.fingerprint
    assert [s["name"] for s in record["steps"]] == [s.name for s in workspace.steps]
    assert tool_version() in html and workspace.data.fingerprint in html
    assert "provisional, needs expert review" in html
    assert ("self-report, no controls" in html) == (run == "self_report_run")


def test_the_lab_report_has_each_finding_and_its_plot(lab_run):
    html = lab_run.export_report()
    view = lab_run.findings()
    for s in view["sections"]:
        assert escape(s["title"]) in html
    comparison = next(s for s in view["sections"] if s["kind"] == "comparison")
    androsterone = [f for f in comparison["findings"] if f["marker_id"] == "androsterone"]
    assert androsterone and escape(androsterone[0]["marker_name"]) in html
    plots = sum(len(s["findings"]) for s in view["sections"])
    assert html.count("<svg") == plots
    assert "<script src" not in html and "http://" not in html.replace("http://www.w3.org/2000/svg", "")


def test_the_report_records_which_step_ran_which_analysis(lab_run):
    steps = {s["name"]: s for s in lab_run.run_record()["steps"]}
    compare = steps["Compare PSSD with healthy control"]
    assert compare["state"] == "done"
    assert compare["record"]["groups"] == ["PSSD", "healthy control"]
    assert compare["record"]["dataset"]["fingerprint"] == lab_run.data.fingerprint
    assert "record" not in steps["Positions within reference ranges"]  # descriptions, no test


def test_nothing_is_exported_before_a_run_has_finished(tmp_path):
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path))
    workspace.load()
    with pytest.raises(RunError, match="before exporting"):
        workspace.export_csv()
    with pytest.raises(RunError, match="before exporting"):
        workspace.export_report()
