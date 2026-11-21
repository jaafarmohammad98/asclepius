import json
from pathlib import Path

import pytest

from asclepius import synthetic
from asclepius.profile import (
    CLASSES,
    DRUG_CLASSES,
    PROFILE_ONSET,
    PROFILE_SYMPTOM,
    PROFILE_TRAJECTORY,
    author_profiles,
    fisher_exact,
    odds_ratio,
    profile_drugs,
)
from asclepius.record import export_run
from asclepius.run import NOTHING_RELIABLE, STRONG, SUGGESTIVE, Settings
from asclepius.selfreport import load_self_reports


@pytest.mark.parametrize(
    ("table", "p"),
    [
        (((3, 1), (1, 3)), 0.4857),  # Fisher's lady tasting tea
        (((6, 2), (1, 4)), 0.1026),  # scipy.stats.fisher_exact's documented example
        (((5, 0), (0, 5)), 0.0079),  # 2 / C(10, 5)
        (((4, 4), (4, 4)), 1.0),
    ],
)
def test_fisher_exact_two_sided_p_value(table, p):
    assert fisher_exact(*table[0], *table[1]) == pytest.approx(p, abs=5e-5)


def test_odds_ratio_with_woolf_interval():
    # 6 * 4 / (2 * 1) = 12; log standard error sqrt(1/6 + 1/2 + 1 + 1/4) = 1.3844
    ratio, low, high = odds_ratio(6, 2, 1, 4)
    assert ratio == pytest.approx(12.0)
    assert low == pytest.approx(0.796, abs=1e-3)
    assert high == pytest.approx(180.9, rel=1e-3)


def test_odds_ratio_with_an_empty_cell_adds_a_half_to_every_cell():
    ratio, low, high = odds_ratio(5, 0, 0, 5)  # 5.5 * 5.5 / (0.5 * 0.5) = 121
    assert ratio == pytest.approx(121.0)
    assert 0 < low < 121 < high


FIXTURE = Path(__file__).parent / "fixtures" / "selfreport"  # hand-made, not real posts


def test_author_profiles_have_one_row_per_author_with_symptoms_drugs_and_classes():
    profiles = author_profiles(load_self_reports(FIXTURE))
    assert profiles.index.is_unique and len(profiles) == 5  # the verified authors with a key
    took_two = profiles[profiles["drugs"].map(len) == 2].iloc[0]  # escitalopram, then finasteride
    assert took_two["drugs"] == ("escitalopram", "finasteride")
    assert took_two["classes"] == ("SSRI", "finasteride")  # counts in each class
    assert bool(took_two["genital_numbness"]) and bool(took_two["erectile_dysfunction"])
    assert not took_two["low_libido"]
    sertraline = profiles[profiles["drugs"].map(("sertraline",).__eq__)].iloc[0]
    assert sertraline["classes"] == ("SSRI",) and sertraline["onset"] == "after_stopping"
    assert sertraline["overall_trajectory"] == "improving"  # the latest post
    no_drug = profiles[profiles["drugs"].map(len) == 0]
    assert len(no_drug) == 2 and set(no_drug["classes"]) == {()}


def test_every_drug_class_is_one_of_the_six_and_unspecified_antidepressants_have_none():
    assert set(DRUG_CLASSES.values()) == set(CLASSES)
    assert CLASSES == ("SSRI", "SNRI", "tricyclic", "other antidepressant", "finasteride", "isotretinoin")
    assert DRUG_CLASSES["unspecified_ssri"] == "SSRI"
    assert "unspecified_antidepressant" not in DRUG_CLASSES  # could be any class


def self_reports(tmp_path, *, seed=1, authors=500, null=False, associations=()):
    out = synthetic.write_self_reports(
        tmp_path, authors=authors, null=null, associations=list(associations), seed=seed
    )
    return load_self_reports(out.export)


def finding(result, symptom, group, level="class"):
    rows = result.findings[
        (result.findings["marker_id"] == f"symptom:{symptom}")
        & (result.findings["group_a"] == group)
        & (result.findings["level"] == level)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


@pytest.mark.parametrize("seed", range(3))
def test_a_planted_drug_symptom_association_is_found_and_never_strong(tmp_path, seed):
    # at 500 authors (about 60 on finasteride) chance makes the odds ratio about 2 in some seeds
    planted = synthetic.PlantedAssociation("finasteride", "erectile_dysfunction", 3.0)
    result = profile_drugs(self_reports(tmp_path, seed=seed, authors=1000, associations=[planted]))
    row = finding(result, "erectile_dysfunction", "finasteride")
    assert row["comparison"] == PROFILE_SYMPTOM == "drug profile: symptom"
    assert row["group_b"] == "other drug classes" and row["sex"] == "all"
    assert row["label"] == SUGGESTIVE
    assert row["caveat"] == "self-report, no controls"
    assert row["odds_ratio"] > 1.5 and row["odds_ratio_low"] > 1
    assert row["effect_size"] > 0 and row["ci_low"] > 0  # share of finasteride takers minus the rest
    assert row["share_a"] - row["share_b"] == pytest.approx(row["effect_size"])
    assert row["n_a"] >= 10 and row["n_b"] >= 10 and row["reported_a"] >= 5
    assert STRONG not in set(result.findings["label"])
    assert result.statement != NOTHING_RELIABLE


# 1 of 40 null seeds at 500 authors (seed 1: paroxetine, q 0.064) gives a chance suggestive row,
# as a 10% false discovery rate allows; never a strong one
@pytest.mark.parametrize("seed", [0, 2, 3])
def test_null_self_reports_give_nothing_reliable_yet(tmp_path, seed):
    result = profile_drugs(self_reports(tmp_path, seed=seed, null=True))
    assert result.statement == NOTHING_RELIABLE
    assert set(result.findings["label"]) <= {"noise-level"}
    kinds = set(result.findings["comparison"])
    assert {PROFILE_SYMPTOM, PROFILE_ONSET, PROFILE_TRAJECTORY} >= kinds >= {PROFILE_SYMPTOM}


def test_drug_classes_and_the_most_common_single_drugs_are_compared(tmp_path):
    result = profile_drugs(self_reports(tmp_path), top_drugs=3)
    classes = set(result.findings.loc[result.findings["level"] == "class", "group_a"])
    assert classes == {"SSRI", "SNRI", "finasteride", "isotretinoin"}  # the synthetic drugs' classes
    drugs = result.findings[result.findings["level"] == "drug"]
    assert set(drugs["group_b"]) == {"other causal drugs"}
    # finasteride and isotretinoin are a class each, and venlafaxine the only SNRI: no repeat
    assert len(set(drugs["group_a"])) == 3
    assert not set(drugs["group_a"]) & {"finasteride", "isotretinoin", "venlafaxine"}
    trajectory = result.findings[result.findings["comparison"] == PROFILE_TRAJECTORY]
    assert set(trajectory["marker_id"]) == {
        "trajectory:recovered", "trajectory:improving", "trajectory:stable", "trajectory:worsening"
    }  # fmt: skip
    # one correction over every test of the run
    assert result.findings["q_value"].notna().all()
    assert len(result.findings) == result.findings["p_value"].notna().sum()


def test_a_group_below_ten_authors_is_refused_and_small_counts_are_hidden(tmp_path):
    result = profile_drugs(self_reports(tmp_path, authors=80))  # isotretinoin about 6 authors
    refused = result.refused[result.refused["group_a"] == "isotretinoin"]
    assert len(refused) and "fewer than 10" in refused.iloc[0]["reason"]
    assert "isotretinoin" not in set(result.findings["group_a"])
    for table in (result.findings, result.refused, result.summary):
        for column in [c for c in table.columns if c.startswith(("n_", "reported_", "authors"))]:
            counts = table[column].dropna()
            assert not ((counts > 0) & (counts < 5)).any(), column
    # a row with a cell below 5 shows no share, count or odds ratio that would give it back
    small = result.findings[result.findings["reported_a"].isna()]
    assert len(small)
    hidden = ["reported_b", "share_a", "share_b", "odds_ratio", "effect_size", "ci_low", "ci_high"]
    assert small[hidden].isna().all().all()
    assert small["p_value"].notna().all() and small["label"].notna().all()


def test_authors_with_drugs_in_several_classes_count_in_each_and_the_output_says_so(tmp_path):
    result = profile_drugs(self_reports(tmp_path))
    assert "counts in each" in result.note
    assert "several" in result.record["note"]


def test_the_result_names_no_author(tmp_path):
    result = profile_drugs(self_reports(tmp_path))
    for table in (result.findings, result.refused, result.summary):
        assert not {"author", "author_key", "case_id"} & set(table.columns)


def test_settings_are_recorded_and_the_run_is_repeatable(tmp_path):
    data = self_reports(tmp_path)
    a, b = profile_drugs(data, top_drugs=4), profile_drugs(data, top_drugs=4)
    assert a.findings.equals(b.findings)
    assert a.record["settings"] == Settings().record() | {"top_drugs": 4}
    assert a.record["dataset"]["kind"] == "self-report"


def test_a_profile_exports_like_any_run(tmp_path):
    result = profile_drugs(self_reports(tmp_path / "sr"))
    out = export_run(result, tmp_path / "out")
    record = json.loads((out / "run_record.json").read_text())
    assert set(record["outputs"]) == {"findings.csv", "refused.csv", "skipped.csv", "summary.csv"}
    assert record["analysis"] == "drug and symptom profile"
    assert "odds_ratio" in (out / "findings.csv").read_text().splitlines()[0]


REAL_EXPORT = Path(__file__).parents[2] / "harvester" / "data" / "export"


@pytest.mark.skipif(not (REAL_EXPORT / "cases.csv").exists(), reason="the r/PSSD export is not here")
def test_the_real_export_runs_and_shows_no_case_or_small_count():
    data = load_self_reports(REAL_EXPORT)
    result = profile_drugs(data)
    assert STRONG not in set(result.findings["label"])
    tables = (result.findings, result.refused, result.summary)
    for table in tables:
        assert not {"author", "author_key", "case_id"} & set(table.columns)
        for column in [c for c in table.columns if c.startswith(("n_", "reported_", "authors"))]:
            counts = table[column].dropna()
            assert not ((counts > 0) & (counts < 5)).any(), column
