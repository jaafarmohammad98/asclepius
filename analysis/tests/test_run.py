import pandas as pd
import pytest

from asclepius import ratios, synthetic
from asclepius.compare import compare_groups
from asclepius.dataset import load_lab_reports
from asclepius.refrange import combine_findings
from asclepius.run import NOTHING_RELIABLE, GroupTooSmall, Settings, label_findings, run_analysis

GROUPS = ("PSSD", "healthy control")


def lab_dataset(tmp_path, **settings):
    out = synthetic.write_lab_reports(tmp_path, **settings)
    return load_lab_reports([out.reports], metadata=out.metadata)


@pytest.mark.parametrize("seed", range(20))
def test_null_data_gives_no_strong_finding(tmp_path, seed):
    # half the reports in tier A, so the Tier A re-run can test most rows. At 40 per group and a
    # quarter tier A, every row is "too small to test" and this check could not fail.
    data = lab_dataset(
        tmp_path, groups={"PSSD": 100, "healthy control": 100}, null=True, tier_a_share=0.5, seed=seed
    )
    findings = run_analysis(data, *GROUPS, Settings(bootstrap=200)).findings
    assert (findings["tier_a_result"] != "Tier A too small to test").mean() > 0.9
    assert not (findings["label"] == "strong").any()


def test_null_data_says_nothing_reliable_yet(tmp_path):
    data = lab_dataset(tmp_path, null=True, seed=1)
    result = run_analysis(data, *GROUPS)
    assert set(result.findings["label"]) == {"noise-level"}
    assert result.statement == NOTHING_RELIABLE == "nothing reliable yet"


def test_every_finding_shows_a_full_dataset_and_a_tier_a_result(tmp_path):
    planted = [synthetic.PlantedDifference("androsterone", 0.5, group="PSSD")]
    data = lab_dataset(
        tmp_path, groups={"PSSD": 100, "healthy control": 100}, planted=planted, tier_a_share=0.5, seed=1
    )
    findings = run_analysis(data, *GROUPS).findings.set_index(["marker_id", "sex"])
    tier_a = data.records[data.records["tier"] == "A"]
    for sex in ("male", "female"):
        row = findings.loc[("androsterone", sex)]
        expected_n = int(((tier_a["group"] == "PSSD") & (tier_a["sex"] == sex)).sum())
        assert row["tier_a_n_a"] == expected_n < row["n_a"]
        assert row["tier_a_effect_size"] < -0.33 and row["tier_a_p_value"] < 0.05
        assert row["tier_a_result"] == "holds"
        assert row["holds_in_tier_a"]
        assert row["label"] == "strong"


def test_a_finding_is_strong_only_where_it_holds_in_tier_a(tmp_path):
    planted = [synthetic.PlantedDifference("androsterone", 0.5, group="PSSD")]
    data = lab_dataset(tmp_path, groups={"PSSD": 100, "healthy control": 100}, planted=planted, seed=1)
    result = run_analysis(data, *GROUPS)
    found = result.findings.set_index(["marker_id", "sex"])
    for marker in ("androsterone", "androsterone/etiocholanolone"):
        # about 25 tier A reports per group, 10 PSSD men: enough to test
        men = found.loc[(marker, "male")]
        assert (men["tier_a_result"], men["label"]) == ("holds", "strong")
        # 9 PSSD and 5 control women in tier A: survives correction, but cannot be checked
        women = found.loc[(marker, "female")]
        assert women["q_value"] <= 0.05 and abs(women["effect_size"]) >= 0.33
        assert (women["tier_a_n_a"], women["tier_a_n_b"]) == (9, 5)
        assert (women["tier_a_result"], women["label"]) == ("Tier A too small to test", "suggestive")
        assert pd.isna(women["tier_a_effect_size"]) and pd.isna(women["tier_a_p_value"])
    assert result.statement != NOTHING_RELIABLE


def test_with_no_tier_a_records_every_finding_says_tier_a_is_too_small(tmp_path):
    planted = [synthetic.PlantedDifference("androsterone", 0.5, group="PSSD")]
    data = lab_dataset(tmp_path, planted=planted, tier_a_share=0, seed=1)
    findings = run_analysis(data, *GROUPS).findings
    assert set(findings["tier_a_result"]) == {"Tier A too small to test"}
    assert (findings[["tier_a_n_a", "tier_a_n_b"]] == 0).all().all()
    assert "strong" not in set(findings["label"])


def test_a_finding_that_disappears_in_tier_a_is_not_strong():
    findings = pd.DataFrame({"p_value": [0.001], "effect_size": [0.6], "holds_in_tier_a": [False]})
    assert label_findings(findings, Settings())["label"].item() == "suggestive"


def test_one_correction_covers_markers_ratios_and_both_sexes(tmp_path):
    planted = [synthetic.PlantedDifference("androsterone", 0.5, group="PSSD")]
    data = lab_dataset(tmp_path, planted=planted, seed=2)
    findings = run_analysis(data, *GROUPS).findings
    # 26 markers and 8 ratios, in each sex: one run of 68 tests
    assert len(findings) == 68
    assert set(findings["comparison"]) == {"group comparison", "enzyme-activity ratio"}
    assert set(findings["sex"]) == {"male", "female"}
    combined = combine_findings(compare_groups(data, *GROUPS), ratios.compare_ratios(data, *GROUPS).rows)
    assert findings["q_value"].tolist() == label_findings(combined, Settings())["q_value"].tolist()
    # correcting each call or each sex on its own would give other q-values
    for part in ("comparison", "sex"):
        alone = pd.concat(
            label_findings(rows.drop(columns=["q_value", "label"]), Settings())
            for _, rows in findings.groupby(part)
        ).sort_index()
        assert not (alone["q_value"] == findings["q_value"]).all()


def test_benjamini_hochberg_q_values():
    # worked example: p sorted 0.01, 0.02, 0.03, 0.04 of 4 tests -> each p * 4 / rank, made monotone
    table = pd.DataFrame({"p_value": [0.04, 0.01, 0.03, 0.02], "effect_size": [0.5] * 4})
    assert label_findings(table, Settings())["q_value"].tolist() == pytest.approx([0.04] * 4)
    table = pd.DataFrame({"p_value": [0.001, 0.5, 0.04], "effect_size": [0.5] * 3})
    assert label_findings(table, Settings())["q_value"].tolist() == pytest.approx([0.003, 0.5, 0.06])


def test_labels_follow_the_thresholds():
    table = pd.DataFrame(
        {
            "p_value": [0.001, 0.001, 0.001, 0.08, 0.5],
            "effect_size": [-0.4, 0.2, 0.4, 0.9, 0.9],
            "holds_in_tier_a": [True, True, False, True, True],
        }
    )
    # one test per row, so q equals p
    labels = [label_findings(table.iloc[[i]], Settings())["label"].item() for i in range(len(table))]
    assert labels == ["strong", "suggestive", "suggestive", "suggestive", "noise-level"]


def test_self_report_findings_are_never_strong():
    table = pd.DataFrame({"p_value": [0.0001], "effect_size": [0.9], "holds_in_tier_a": [True]})
    labelled = label_findings(table, Settings(), self_report=True)
    assert labelled["label"].item() == "suggestive"
    assert labelled["caveat"].item() == "self-report, no controls"


def test_reference_range_rows_are_not_tests():
    tests = pd.DataFrame({"comparison": ["group comparison"] * 2, "p_value": [0.01, 0.02]})
    ranges = pd.DataFrame({"comparison": ["reference range"], "share_below": [0.2]})
    labelled = label_findings(combine_findings(tests, ranges), Settings())
    assert labelled["q_value"].iloc[:2].tolist() == pytest.approx([0.02, 0.02])  # two tests, not three
    assert labelled["q_value"].isna().iloc[2]
    assert labelled["label"].isna().iloc[2]


def test_a_group_of_nine_is_refused_with_a_clear_message(tmp_path):
    data = lab_dataset(tmp_path, groups={"PSSD": 9, "healthy control": 40}, seed=1)
    with pytest.raises(GroupTooSmall) as refused:
        run_analysis(data, *GROUPS)
    message = str(refused.value)
    assert "'PSSD' has 9 reports" in message
    assert "at least 10" in message


def test_a_sex_with_too_few_reports_is_not_tested(tmp_path):
    # 40 per group at 15% women: about 6 women each, too few to compare
    data = lab_dataset(tmp_path, female_share=0.15, seed=1)
    result = run_analysis(data, *GROUPS)
    assert set(result.findings["sex"]) == {"male"}
    assert set(result.refused["sex"]) == {"female"}
    assert (result.refused[["n_a", "n_b"]].min(axis=1) < 10).all()
    assert result.refused["reason"].str.contains("fewer than 10").all()
    # the refused rows do not count as tests in the correction
    assert result.findings["q_value"].notna().all()


def test_settings_are_recorded_with_the_run_and_marked_provisional(tmp_path):
    data = lab_dataset(tmp_path, null=True, seed=1)
    settings = Settings(suggestive_q=0.2, permutations=999, bootstrap=200)
    result = run_analysis(data, *GROUPS, settings)
    assert result.settings == settings.record()
    assert result.settings["suggestive_q"] == 0.2
    assert result.settings["min_group_size"] == 10
    assert "Benjamini-Hochberg" in str(result.settings["correction"])
    assert result.settings["status"] == "provisional, needs expert review"


def test_settings_must_make_sense():
    with pytest.raises(ValueError):
        Settings(strong_q=0.2, suggestive_q=0.1)
    with pytest.raises(ValueError):
        Settings(min_group_size=0)
