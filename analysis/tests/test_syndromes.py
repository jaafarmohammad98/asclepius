import json

import numpy as np
import pandas as pd
import pytest

from asclepius import synthetic
from asclepius.dataset import load_lab_reports
from asclepius.record import export_run, reproduce
from asclepius.run import GroupTooSmall, Settings, benjamini_hochberg
from asclepius.syndromes import compare_syndromes, range_shift

FAST = Settings(permutations=999, bootstrap=200)
ALL = {"PSSD": 100, "PFS": 100, "PAS": 100, "healthy control": 100}


def lab_dataset(tmp_path, groups=ALL, planted=(), **settings):
    out = synthetic.write_lab_reports(tmp_path, groups=groups, planted=list(planted), **settings)
    return load_lab_reports([out.reports], metadata=out.metadata)


def summary_row(result, marker_id, sex):
    rows = result.summary[(result.summary["marker_id"] == marker_id) & (result.summary["sex"] == sex)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_a_marker_lower_in_every_syndrome_is_shared(tmp_path):
    planted = [synthetic.PlantedDifference("androsterone", 0.5, group=g) for g in ("PSSD", "PFS", "PAS")]
    data = lab_dataset(tmp_path, planted=planted, tier_a_share=0.5, seed=1)
    result = compare_syndromes(data, settings=FAST)
    for sex in ("male", "female"):
        row = summary_row(result, "androsterone", sex)
        assert row["pattern"] == "shared"
        assert row["shared_by"] == "PSSD, PFS, PAS"
        assert (row["direction_PSSD"], row["direction_PFS"], row["direction_PAS"]) == ("lower",) * 3
        assert row["label"] in ("strong", "suggestive")
        assert row["differ"] == ""
        assert row["n_PSSD"] >= 10 and row["n_reference"] >= 10


def test_a_marker_lower_in_one_syndrome_only_differs(tmp_path):
    planted = [synthetic.PlantedDifference("etiocholanolone", 0.4, group="PFS")]
    data = lab_dataset(tmp_path, planted=planted, tier_a_share=0.5, seed=1)
    result = compare_syndromes(data, settings=FAST)
    row = summary_row(result, "etiocholanolone", "male")
    assert row["pattern"] == "differs"
    assert row["direction_PFS"] == "lower"
    assert (row["direction_PSSD"], row["direction_PAS"]) == ("no clear shift", "no clear shift")
    assert "PSSD higher than PFS" in row["differ"] and "PFS lower than PAS" in row["differ"]
    assert row["shared_by"] == ""
    # the ratio built on it moves the other way, in PFS alone
    ratio = summary_row(result, "androsterone/etiocholanolone", "male")
    assert ratio["measure"] == "enzyme-activity ratio"
    assert ratio["direction_PFS"] == "higher" and ratio["pattern"] == "differs"


@pytest.mark.parametrize("seed", range(3))
def test_null_data_says_nothing_reliable_yet(tmp_path, seed):
    data = lab_dataset(tmp_path, groups={g: 40 for g in ALL}, null=True, seed=seed)
    result = compare_syndromes(data, settings=FAST)
    assert set(result.findings["label"]) == {"noise-level"}
    assert result.statement == "nothing reliable yet"
    assert set(result.summary["pattern"]) == {"no clear shift"}


def test_every_test_is_corrected_together(tmp_path):
    data = lab_dataset(tmp_path, groups={g: 40 for g in ALL}, null=True, seed=0)
    result = compare_syndromes(data, settings=FAST)
    # 34 markers and ratios, 2 sexes, 3 syndromes against controls and 3 pairs between them
    assert len(result.findings) == 34 * 2 * 6
    assert set(result.findings["comparison"]) == {
        "cross-syndrome: vs reference",
        "cross-syndrome: between syndromes",
    }
    p = result.findings["p_value"].to_numpy()
    assert result.findings["q_value"].tolist() == pytest.approx(benjamini_hochberg(p).tolist())
    assert len(result.summary) == 34 * 2
    assert set(result.summary.columns) >= {f"n_{s}" for s in ("PSSD", "PFS", "PAS", "reference")}


def test_a_syndrome_with_too_few_reports_is_left_out(tmp_path):
    data = lab_dataset(tmp_path, groups={"PSSD": 40, "PFS": 40, "PAS": 6, "healthy control": 40}, seed=0)
    result = compare_syndromes(data, settings=FAST)
    assert "PAS" not in set(result.findings["group_a"]) | set(result.findings["group_b"])
    left_out = result.refused[result.refused["group_a"] == "PAS"]
    assert len(left_out) == 1
    assert left_out["n_a"].item() == 6
    assert "at least 10" in left_out["reason"].item()
    assert not any("PAS" in c for c in result.summary.columns)
    assert result.record["groups"] == ["PSSD", "PFS", "healthy control"]


def test_fewer_than_two_syndromes_is_refused(tmp_path):
    data = lab_dataset(tmp_path, groups={"PSSD": 40, "PFS": 3, "healthy control": 40}, seed=0)
    with pytest.raises(GroupTooSmall, match="at least two") as refused:
        compare_syndromes(data, settings=FAST)
    assert "fewer than 5" in str(refused.value)  # a count below 5 is not shown (S11)


def test_a_reference_group_with_too_few_reports_is_refused(tmp_path):
    data = lab_dataset(tmp_path, groups={"PSSD": 40, "PFS": 40, "healthy control": 7}, seed=0)
    with pytest.raises(GroupTooSmall, match="'healthy control' has 7 reports"):
        compare_syndromes(data, settings=FAST)
    with pytest.raises(ValueError, match="No loaded report"):
        compare_syndromes(data, reference="SSRI-exposed control", settings=FAST)
    with pytest.raises(ValueError, match="must be a control group"):
        compare_syndromes(data, reference="PAS", settings=FAST)


def test_a_sex_with_too_few_values_is_not_tested(tmp_path):
    # 20% of 40 is 8 women per group: fewer than 10, so no test among women
    data = lab_dataset(tmp_path, groups={g: 40 for g in ALL}, female_share=0.2, seed=0)
    result = compare_syndromes(data, settings=FAST)
    assert set(result.findings["sex"]) == {"male"}
    women = result.refused[result.refused["sex"] == "female"]
    assert len(women) == 34 * 6
    female = result.summary[result.summary["sex"] == "female"]
    assert len(female) == 34
    assert set(female["pattern"]) == {"not tested"}
    assert set(female["direction_PSSD"]) == {"not tested"}
    assert set(female["n_PSSD"]) == {8}


SYNDROMES_ONLY = {"PSSD": 100, "PFS": 100, "PAS": 100}


def test_without_controls_each_syndrome_is_compared_with_the_reference_range(tmp_path):
    # at a third of normal, most androsterone values fall below the range
    planted = [synthetic.PlantedDifference("androsterone", 0.3, group=g) for g in ("PSSD", "PFS", "PAS")]
    data = lab_dataset(tmp_path, groups=SYNDROMES_ONLY, planted=planted, tier_a_share=0.5, seed=1)
    result = compare_syndromes(data, reference="reference range", settings=FAST)
    row = summary_row(result, "androsterone", "male")
    assert row["pattern"] == "shared" and row["shared_by"] == "PSSD, PFS, PAS"
    assert row["direction_PSSD"] == "lower" and row["label"] in ("strong", "suggestive")
    assert row["n_PSSD"] == 70
    vs = result.findings[result.findings["comparison"] == "cross-syndrome: vs reference"]
    assert set(vs["group_b"]) == {"reference range"}
    test = vs[(vs["marker_id"] == "androsterone") & (vs["sex"] == "male") & (vs["group_a"] == "PFS")].iloc[0]
    # effect: share above the range minus share below it; fewer than 5 above, so both shares are
    # hidden (one would give the other back with the effect, S11)
    assert -1 <= test["ci_low"] <= test["effect_size"] < -0.5 and test["tier_a_result"] == "holds"
    assert test[["share_above", "share_below"]].isna().all()
    # ratios have no range: only compared between syndromes
    ratio = summary_row(result, "androsterone/etiocholanolone", "male")
    assert ratio["direction_PSSD"] == "not tested"
    assert result.record["groups"] == ["PSSD", "PFS", "PAS", "reference range"]


def test_without_controls_null_data_says_nothing_reliable_yet(tmp_path):
    data = lab_dataset(tmp_path, groups=SYNDROMES_ONLY, null=True, seed=0)
    result = compare_syndromes(data, reference="reference range", settings=FAST)
    assert result.statement == "nothing reliable yet"


def test_a_range_shift_is_share_above_minus_share_below():
    zones = pd.Series(["above"] * 2 + ["below"] * 8 + ["within"] * 10)
    shift = range_shift(zones, FAST, np.random.default_rng(0))
    assert shift["n"] == 20
    assert shift["effect_size"] == pytest.approx((2 - 8) / 20)
    # exact sign test: 2 of 10 outside values above, two-sided P = 2 * 56 / 1024
    assert shift["p_value"] == pytest.approx(112 / 1024)
    assert shift["ci_low"] < -0.3 < shift["ci_high"]


@pytest.mark.parametrize("reference", ["healthy control", "reference range"])
def test_a_cross_syndrome_run_is_exported_and_reproduced(tmp_path, reference):
    groups = {g: 40 for g in ALL}
    if reference == "reference range":
        del groups["healthy control"]
    out = synthetic.write_lab_reports(tmp_path / "data", groups=groups, seed=3)
    data = load_lab_reports([out.reports], metadata=out.metadata)
    result = compare_syndromes(data, reference=reference, settings=FAST)
    assert result.record["analysis"] == "cross-syndrome"
    assert result.record["reference"] == reference
    folder = export_run(result, tmp_path / "first")
    assert (folder / "summary.csv").exists()
    record = json.loads((folder / "run_record.json").read_text())
    assert "summary.csv" in record["outputs"]
    again = reproduce(record, [out.reports], metadata=out.metadata)
    pd.testing.assert_frame_equal(again.findings, result.findings)
    pd.testing.assert_frame_equal(again.summary, result.summary)
    second = export_run(again, tmp_path / "second")
    for name in ("findings.csv", "refused.csv", "summary.csv"):
        assert (second / name).read_bytes() == (folder / name).read_bytes()
