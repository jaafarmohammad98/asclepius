"""Export (M3-10c): the findings as CSV and a report as HTML, each with the tool and dataset version."""

import json
from html import escape

import pytest
from test_web import FAST, add_export, add_lab, finished, lab_files

from asclepius import synthetic
from asclepius.record import tool_version
from asclepius.web import workspace as workspace_module
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
    rows = [r for r in read_findings_csv(workspace.export_csv()) if r["row"] == "finding"]
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
        assert json.loads(row["settings"]) == workspace.export_record()["settings"]
        assert row["run_date"] == workspace.export_record()["date"]
        is_self_report = run == "self_report_run"
        assert ("self-report, no controls" in row["labels"]) == is_self_report


@pytest.mark.parametrize("run", ["lab_run", "self_report_run"])
def test_every_step_is_in_the_csv_with_its_statement(run, request):
    """A step with nothing reliable says so, and one with no findings, or not built, says why (S8)."""
    workspace = request.getfixturevalue(run)
    rows = read_findings_csv(workspace.export_csv())
    for s in workspace.findings()["sections"]:
        mine = [r for r in rows if r["section"] == s["kind"]]
        assert mine, s["kind"]
        if s["state"] == "done":
            assert {r["statement"] for r in mine} == {s["statement"]}
        else:
            assert [r["row"] for r in mine] == ["statement"]
            assert mine[0]["statement"] == f"{s['state']}: {s['message']}"


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
    assert record == workspace.export_record()
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
    steps = {s["name"]: s for s in lab_run.export_record()["steps"]}
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


def test_a_refused_step_does_not_export_a_count_below_5(tmp_path):
    """A group of 3 refuses the subgroups (too few patients) and the comparison (too small); the
    reasons say "fewer than 5", in the report and its record (spec S11)."""
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path, groups={"PSSD": 3, "healthy control": 40}))
    workspace.load()
    workspace.start_run("PSSD", "healthy control", settings=FAST)
    finished(workspace)
    html = workspace.export_report()
    messages = [s["message"] for s in report_record(html)["steps"]]
    assert any("Refused: fewer than 5 patients" in m for m in messages)
    assert any("'PSSD' has fewer than 5 reports" in m for m in messages)
    assert " 3 patients" not in html and "has 3 reports" not in html
    assert " 3 patients" not in workspace.export_csv()


def test_an_unexpected_error_is_not_exported_word_for_word(tmp_path, monkeypatch):
    """An error's text can name a file or a value, so the exports say only that the step failed."""

    def broken(data):
        raise OSError("/Users/someone/patients/jane-doe.csv: unreadable")

    monkeypatch.setattr(workspace_module, "lab_range_findings", broken)
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path))
    workspace.load()
    workspace.start_run("PSSD", "healthy control", settings=FAST)
    status = finished(workspace)
    assert "jane-doe" in status["steps"][-1]["message"]  # the page, on this computer, says why
    html, csv_text = workspace.export_report(), workspace.export_csv()
    assert "jane-doe" not in html and "jane-doe" not in csv_text
    step = report_record(html)["steps"][-1]
    assert step["state"] == "failed"
    assert step["message"] == "Unexpected error; the details are shown in the app only."


def test_a_hidden_intervention_count_reads_hidden_not_fewer_than_5():
    from asclepius.web.export import _hidden_counts

    rows = _hidden_counts([{"intervention": "x", "n_authors": None, "worse": None, "share_worse": None}])
    assert rows[0]["n_authors"] == "hidden" and rows[0]["worse"] == "hidden"
    assert rows[0]["share_worse"] is None
