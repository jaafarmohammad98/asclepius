from pathlib import Path

import pandas as pd
import pytest

from asclepius import ratios, synthetic
from asclepius.dataset import LabDataset, load_lab_reports

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = FIXTURES / "dutch_samples.csv"  # the public sample reports
README = Path(__file__).parents[1] / "README.md"


def one_report(values: dict[str, float | str | None]) -> LabDataset:
    """A dataset with one report holding `values` (marker id -> number, "<x", ">x" or None)."""
    rows = []
    for marker, value in values.items():
        comparator, number = None, value
        if isinstance(value, str):
            comparator, number = value[0], float(value[1:])
        rows.append(
            {
                "record": "1/r001",
                "marker_id": marker,
                "marker_name": marker,
                "comparator": comparator,
                "value": number,
                "unit": "ng/mg",
            }
        )
    markers = pd.DataFrame(rows).astype({"value": "float64"})
    records = pd.DataFrame(
        [{"record": "1/r001", "sex": "male", "age_years": 30, "group": "PSSD", "tier": "B"}]
    )
    return LabDataset(markers=markers, fingerprint="x", records=records)


def one_ratio(values: dict[str, float | str | None], ratio_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The row of `ratio_id` computed for `one_report(values)` (empty when skipped), and all skips."""
    computed = ratios.compute_ratios(one_report(values))
    row = computed.data.markers[computed.data.markers["marker_id"] == ratio_id]
    return row, computed.skipped


def test_the_list_is_the_spec_list_and_marked_provisional():
    pairs = {(r.numerator, r.denominator) for r in ratios.RATIOS}
    assert pairs == {
        ("androsterone", "etiocholanolone"),
        ("5a_androstanediol", "5b_androstanediol"),
        ("a_tetrahydrocortisol_a_thf", "b_tetrahydrocortisol_b_thf"),
        ("a_pregnanediol", "b_pregnanediol"),
        ("24hr_free_cortisol", "24hr_free_cortisone"),
        ("2_methoxy_e1", "2_oh_e1"),
        ("2_oh_e1", "16_oh_e1"),
        ("2_oh_e1", "4_oh_e1"),
    }
    assert ratios.STATUS == "provisional, needs expert review"
    # the synthetic generator writes every component, so the ratios can be tested
    assert {m for r in ratios.RATIOS for m in (r.numerator, r.denominator)} <= set(synthetic.MARKER_IDS)


def test_every_ratio_is_computed_for_the_public_samples():
    computed = ratios.compute_ratios(load_lab_reports([SAMPLES]))
    values = computed.data.markers
    assert set(values["marker_id"]) == {r.id for r in ratios.RATIOS}
    assert values.groupby("marker_id").size().eq(3).all()  # three sample reports
    assert computed.skipped.empty
    # r001: androsterone 1810 / etiocholanolone 680
    first = values[(values["record"] == "1/r001") & (values["marker_id"] == "androsterone/etiocholanolone")]
    assert first["value"].item() == pytest.approx(1810 / 680)
    assert first["marker_name"].item() == "Androsterone / etiocholanolone"


def test_a_ratio_is_skipped_and_reported_when_a_component_is_missing_or_zero():
    computed = ratios.compute_ratios(
        one_report({"androsterone": 10, "2_oh_e1": 0, "16_oh_e1": 2, "4_oh_e1": 1, "2_methoxy_e1": 1})
    )
    assert computed.data.markers.empty
    skipped = computed.skipped.set_index("ratio_id")
    assert skipped.loc["androsterone/etiocholanolone", "reason"] == ratios.MISSING
    assert skipped.loc["2_oh_e1/16_oh_e1", "reason"] == ratios.ZERO
    assert skipped.loc["2_methoxy_e1/2_oh_e1", "reason"] == ratios.ZERO
    assert skipped.loc["a_pregnanediol/b_pregnanediol", "reason"] == ratios.MISSING
    assert (skipped["n_reports"] == 1).all()
    # a blank value counts as missing
    _, skipped = one_ratio({"androsterone": None, "etiocholanolone": 2}, "androsterone/etiocholanolone")
    assert ratios.MISSING in set(skipped["reason"])


def test_a_limit_in_one_component_gives_a_limit_on_the_ratio():
    # below 10 over 2 is below 5; 10 over below 2 is above 5
    row, _ = one_ratio({"androsterone": "<10", "etiocholanolone": 2}, "androsterone/etiocholanolone")
    assert (row["comparator"].item(), row["value"].item()) == ("<", 5)
    row, _ = one_ratio({"androsterone": 10, "etiocholanolone": "<2"}, "androsterone/etiocholanolone")
    assert (row["comparator"].item(), row["value"].item()) == (">", 5)
    row, _ = one_ratio({"androsterone": "<10", "etiocholanolone": ">2"}, "androsterone/etiocholanolone")
    assert (row["comparator"].item(), row["value"].item()) == ("<", 5)


def test_two_limits_the_same_way_give_no_ratio():
    row, skipped = one_ratio({"androsterone": "<10", "etiocholanolone": "<2"}, "androsterone/etiocholanolone")
    assert row.empty
    reason = skipped.loc[skipped["ratio_id"] == "androsterone/etiocholanolone", "reason"]
    assert reason.item() == ratios.BOTH_LIMITS


def test_ratios_go_through_the_group_comparison(tmp_path):
    out = synthetic.write_lab_reports(
        tmp_path,
        groups={"PSSD": 100, "healthy control": 100},
        planted=[synthetic.PlantedDifference("androsterone", 0.5)],
        seed=3,
    )
    data = load_lab_reports([out.reports], metadata=out.metadata)
    findings = ratios.compare_ratios(data, "PSSD", "healthy control", permutations=999, bootstrap=200)
    table = findings.rows
    assert findings.skipped.empty
    assert set(table["comparison"]) == {ratios.KIND}
    assert set(table["status"]) == {ratios.STATUS}
    assert set(table["marker_id"]) == {r.id for r in ratios.RATIOS}
    # androsterone halved in PSSD: its ratio over etiocholanolone is the strongest finding, in both sexes
    top = table.head(2)
    assert set(top["marker_id"]) == {"androsterone/etiocholanolone"}
    assert (top["effect_size"] < -0.33).all()
    assert (top["p_value"] < 0.01).all()


def test_the_readme_table_is_generated_from_the_list():
    table = ratios.documentation_table()
    assert ratios.STATUS in table
    for r in ratios.RATIOS:
        assert r.name in table and r.proxy_for in table
    assert table in README.read_text(encoding="utf-8")
