import time

import pytest

from asclepius import synthetic
from asclepius.dataset import load_lab_reports
from asclepius.run import Settings
from asclepius.selfreport import load_self_reports
from asclepius.web.workspace import LoadError, RunError, Workspace


def lab_files(tmp_path, **options):
    """A synthetic extractor CSV and its metadata file, as a researcher would pick them."""
    options.setdefault("groups", {"PSSD": 40, "healthy control": 40})
    options.setdefault("seed", 1)
    return synthetic.write_lab_reports(tmp_path / "lab", **options)


def add_lab(workspace, out):
    workspace.add_file("reports", "reports.csv", out.reports.read_bytes())
    workspace.add_file("metadata", "metadata.csv", out.metadata.read_bytes())


def test_review_counts_lab_records_per_group_sex_and_tier(tmp_path):
    out = lab_files(tmp_path)
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, out)
    review = workspace.load()

    assert review["kind"] == "lab-report"
    totals = {row["group"]: row["records"] for row in review["groups"]}
    assert totals == {"PSSD": 40, "healthy control": 40}
    records = load_lab_reports([out.reports], metadata=out.metadata).records
    expected = records.groupby(["group", "sex", "tier"]).size()
    cells = {(c["group"], c["sex"], c["tier"]): c["records"] for c in review["records"]}
    assert set(cells) == set(expected.index)
    for key, n in expected.items():
        assert cells[key] in (n, None)
        if n < 5:
            assert cells[key] is None  # no count below 5 is shown (spec S11)
    assert review["choices"] == ["PSSD", "healthy control"]


def add_export(workspace, export):
    for path in sorted(export.glob("*.csv")):
        workspace.add_file("export", path.name, path.read_bytes())


def test_review_counts_self_report_authors_per_condition_and_exclusions(tmp_path):
    out = synthetic.write_self_reports(tmp_path / "sr", authors=300, unverified_posts=10, seed=1)
    workspace = Workspace(tmp_path / "work")
    add_export(workspace, out.export)
    review = workspace.load()

    data = load_self_reports(out.export)
    assert review["kind"] == "self-report"
    expected = data.authors["condition"].value_counts()
    conditions = {row["condition"]: row["authors"] for row in review["conditions"]}
    assert set(conditions) == set(expected.index)
    for condition, n in expected.items():
        assert conditions[condition] == (n if n >= 5 else None)
    assert review["authors"] == len(data.authors)
    assert review["excluded"] == {"unverified": 10, "no_author": data.excluded["no_author"]}
    assert review["label"] == "self-report, no controls"


def test_no_files_picked_is_said_plainly(tmp_path):
    with pytest.raises(LoadError, match="No files were picked"):
        Workspace(tmp_path / "work").load()


def test_lab_files_and_an_export_folder_are_not_loaded_together(tmp_path):
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path))
    add_export(workspace, synthetic.write_self_reports(tmp_path / "sr", seed=1).export)
    with pytest.raises(LoadError, match="never pooled"):
        workspace.load()


def test_an_export_folder_missing_a_table_names_it(tmp_path):
    export = synthetic.write_self_reports(tmp_path / "sr", seed=1).export
    (export / "labs.csv").unlink()
    workspace = Workspace(tmp_path / "work")
    add_export(workspace, export)
    with pytest.raises(LoadError, match=r"labs\.csv"):
        workspace.load()


def test_a_broken_dataset_file_is_named_as_picked(tmp_path):
    workspace = Workspace(tmp_path / "work")
    workspace.add_file("reports", "march.csv", b"record_id,marker_id\nr001,x\n")
    with pytest.raises(LoadError, match=r"march\.csv"):
        workspace.load()


def test_picking_again_replaces_the_files(tmp_path):
    workspace = Workspace(tmp_path / "work")
    workspace.add_file("reports", "march.csv", b"not a dataset")
    workspace.clear()
    add_lab(workspace, lab_files(tmp_path))
    assert workspace.load()["kind"] == "lab-report"


FAST = Settings(permutations=999, bootstrap=200)


def finished(workspace, timeout=120):
    """Poll the run's status, as the page does, until it has finished."""
    deadline = time.monotonic() + timeout
    while (status := workspace.status())["state"] == "running":
        assert time.monotonic() < deadline, status
        time.sleep(0.05)
    return status


def steps(status):
    return {step["name"]: step for step in status["steps"]}


def test_a_lab_run_goes_through_every_step_and_says_what_each_found(tmp_path):
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path, planted=[synthetic.PlantedDifference("androsterone", 0.4)]))
    workspace.load()
    workspace.start_run("PSSD", "healthy control", settings=FAST)
    status = finished(workspace)

    assert status["state"] == "finished"
    assert status["done"] == status["total"] == len(status["steps"])
    by_name = steps(status)
    main = by_name["Compare PSSD with healthy control"]
    assert main["state"] == "done"
    assert "strong" in main["message"] or main["message"] == "nothing reliable yet"
    assert by_name["Look for subgroups of PSSD"]["state"] == "done"
    assert by_name["Positions within reference ranges"]["state"] == "done"
    cross = by_name["Compare PSSD, PFS and PAS"]
    assert cross["state"] == "refused" and "at least two" in cross["message"]
    assert workspace.results["Compare PSSD with healthy control"].record["groups"] == [
        "PSSD",
        "healthy control",
    ]


def test_a_group_too_small_is_refused_in_plain_words_and_the_run_still_finishes(tmp_path):
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path, groups={"PSSD": 40, "healthy control": 8}))
    workspace.load()
    workspace.start_run("PSSD", "healthy control", settings=FAST)
    status = finished(workspace)

    main = steps(status)["Compare PSSD with healthy control"]
    assert main["state"] == "refused"
    assert "'healthy control' has 8 reports" in main["message"]
    assert status["state"] == "finished"


def test_a_self_report_run_says_which_analyses_are_not_built_yet(tmp_path):
    workspace = Workspace(tmp_path / "work")
    add_export(workspace, synthetic.write_self_reports(tmp_path / "sr", seed=1).export)
    workspace.load()
    workspace.start_run(settings=FAST)
    status = finished(workspace)

    by_name = steps(status)
    assert by_name["Typed-in lab values within reference ranges"]["state"] == "done"
    assert by_name["Drug and symptom profile"]["state"] == "done"
    assert [s["state"] for s in status["steps"]].count("not built yet") == 2
    assert status["label"] == "self-report, no controls"


def test_a_run_needs_a_loaded_dataset_and_two_different_groups(tmp_path):
    workspace = Workspace(tmp_path / "work")
    with pytest.raises(RunError, match="Load a dataset"):
        workspace.start_run("PSSD", "healthy control")
    add_lab(workspace, lab_files(tmp_path))
    workspace.load()
    with pytest.raises(RunError, match="two different groups"):
        workspace.start_run("PSSD", "PSSD")
    assert workspace.status()["state"] == "idle"


def test_a_total_does_not_give_away_the_hidden_counts_beside_it(tmp_path):
    """A total minus the counts shown would give back a hidden count below 5 (spec S11), so the
    counts hidden beside a total add up to 0 or at least 5."""
    out = lab_files(tmp_path, groups={"PSSD": 40, "healthy control": 40, "PFS": 12})
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, out)
    review = workspace.load()

    records = load_lab_reports([out.reports], metadata=out.metadata).records
    true = records.groupby(["group", "sex", "tier"]).size()
    extra = 0
    for group, n in records["group"].value_counts().items():
        assert {r["group"]: r["records"] for r in review["groups"]}[group] == n
        cells = [c for c in review["records"] if c["group"] == group]
        hidden = sum(int(true[(group, c["sex"], c["tier"])]) for c in cells if c["records"] is None)
        assert hidden == 0 or hidden >= 5
        extra += any(c["records"] is None and true[(group, c["sex"], c["tier"])] >= 5 for c in cells)
    assert extra  # some count of 5 or more had to be hidden too


def test_the_self_report_total_does_not_give_away_a_small_condition(tmp_path):
    out = synthetic.write_self_reports(tmp_path / "sr", authors=300, seed=1)
    workspace = Workspace(tmp_path / "work")
    add_export(workspace, out.export)
    review = workspace.load()

    counts = load_self_reports(out.export).authors["condition"].value_counts()
    assert review["authors"] == counts.sum()
    hidden = sum(int(counts[c["condition"]]) for c in review["conditions"] if c["authors"] is None)
    assert hidden == 0 or hidden >= 5


def test_subgroups_are_looked_for_among_the_patient_group_picked(tmp_path):
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path))
    workspace.load()
    workspace.start_run("healthy control", "PSSD", settings=FAST)
    assert "Look for subgroups of PSSD" in steps(finished(workspace))


def test_with_no_patient_group_picked_there_are_no_subgroups_to_look_for(tmp_path):
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path, groups={"SSRI-exposed control": 40, "healthy control": 40}))
    workspace.load()
    workspace.start_run("SSRI-exposed control", "healthy control", settings=FAST)
    step = steps(finished(workspace))["Look for subgroups"]
    assert step["state"] == "refused" and "PSSD, PFS or PAS" in step["message"]


def test_a_second_run_cannot_start_while_one_is_running(tmp_path):
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path))
    workspace.load()
    workspace.start_run("PSSD", "healthy control", settings=FAST)
    with pytest.raises(RunError, match="already in progress"):
        workspace.start_run("PSSD", "healthy control", settings=FAST)
    with pytest.raises(LoadError, match="in progress"):
        workspace.load()
    finished(workspace)
