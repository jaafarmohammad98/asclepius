import hashlib
import json
from importlib.metadata import version

import pandas as pd
import pytest

from asclepius import synthetic
from asclepius.dataset import load_lab_reports
from asclepius.record import ReproductionError, export_run, reproduce
from asclepius.run import Settings, run_analysis

GROUPS = ("PSSD", "healthy control")
FAST = Settings(permutations=999, bootstrap=200, seed=3)


@pytest.fixture
def written(tmp_path):
    planted = [synthetic.PlantedDifference("androsterone", 0.5, group="PSSD")]
    return synthetic.write_lab_reports(tmp_path / "data", planted=planted, seed=1)


def run(written, settings=FAST):
    data = load_lab_reports([written.reports], metadata=written.metadata)
    return run_analysis(data, *GROUPS, settings)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_run_record_names_files_tool_settings_and_date(written):
    record = run(written).record
    assert record["dataset"]["files"] == [
        {"role": "lab reports", "sha256": sha256(written.reports)},
        {"role": "metadata", "sha256": sha256(written.metadata)},
    ]
    assert "reports.csv" not in json.dumps(record)  # a file name could name a patient
    assert len(record["code_sha256"]) == 64
    assert record["dataset"]["kind"] == "lab-report"
    assert len(record["dataset"]["fingerprint"]) == 64
    assert record["tool"] == f"asclepius {version('asclepius')}"
    assert record["groups"] == list(GROUPS)
    assert record["settings"] == FAST.record()
    assert pd.Timestamp(record["date"]).date() == pd.Timestamp.now(tz="UTC").date()
    assert record["status"] == "provisional, needs expert review"
    json.dumps(record)  # it can be written out as it is


def test_the_same_inputs_and_run_record_give_identical_output(written):
    first = run(written)
    again = reproduce(json.loads(json.dumps(first.record)), [written.reports], metadata=written.metadata)
    pd.testing.assert_frame_equal(again.findings, first.findings)
    pd.testing.assert_frame_equal(again.refused, first.refused)
    pd.testing.assert_frame_equal(again.skipped, first.skipped)
    assert again.statement == first.statement
    assert {**again.record, "date": None} == {**first.record, "date": None}


def test_reproducing_with_changed_data_is_refused(written):
    record = run(written).record
    written.metadata.write_text(written.metadata.read_text().replace(",A\n", ",B\n", 1))
    with pytest.raises(ReproductionError, match="metadata.csv"):
        reproduce(record, [written.reports], metadata=written.metadata)


def test_reproducing_with_a_missing_file_is_refused(written):
    record = run(written).record
    with pytest.raises(ReproductionError, match="has 2 files, and 1 were given"):
        reproduce(record, [written.reports])


def test_reproducing_with_other_code_is_refused(written):
    record = {**run(written).record, "code_sha256": "0" * 64}
    with pytest.raises(ReproductionError, match="other asclepius code"):
        reproduce(record, [written.reports], metadata=written.metadata)


def test_reproducing_with_another_tool_version_is_refused(written):
    record = {**run(written).record, "tool": "asclepius 0.0.1"}
    with pytest.raises(ReproductionError, match="0.0.1"):
        reproduce(record, [written.reports], metadata=written.metadata)


def test_every_export_carries_the_run_record(written, tmp_path):
    result = run(written)
    out = export_run(result, tmp_path / "export")
    files = sorted(p.name for p in out.iterdir())
    assert files == ["findings.csv", "refused.csv", "run_record.json", "skipped.csv"]
    record = json.loads((out / "run_record.json").read_text())
    assert {k: v for k, v in record.items() if k != "outputs"} == result.record
    assert record["outputs"] == {name: sha256(out / name) for name in files if name.endswith(".csv")}
    findings = pd.read_csv(out / "findings.csv")
    assert len(findings) == len(result.findings)
    assert {"effect_size", "tier_a_effect_size", "tier_a_result", "label"} <= set(findings.columns)


def test_reproduced_exports_are_byte_identical_apart_from_the_date(written, tmp_path):
    first = export_run(run(written), tmp_path / "first")
    record = json.loads((first / "run_record.json").read_text())
    again = export_run(reproduce(record, [written.reports], metadata=written.metadata), tmp_path / "again")
    for name in ("findings.csv", "refused.csv", "skipped.csv"):
        assert (again / name).read_bytes() == (first / name).read_bytes()
