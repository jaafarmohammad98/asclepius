import shutil
from pathlib import Path

import pytest

from asclepius.compare import compare_groups
from asclepius.dataset import load_lab_reports
from asclepius.refrange import combine_findings, lab_range_findings, self_report_range_findings, zone
from asclepius.selfreport import load_self_reports

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = FIXTURES / "dutch_samples.csv"  # the public sample reports


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5, "below"), (10, "within"), (15, "within"), (20, "within"), (21, "above")],
)
def test_a_number_is_below_within_or_above_the_range(value, expected):
    assert zone(value, 10, 20) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("<10", "below"),
        ("<8", "below"),
        ("<15", "unclear"),
        (">20", "above"),
        (">12", "unclear"),
        (">5", "unclear"),
    ],
)
def test_a_limit_is_placed_only_when_certain(value, expected):
    assert zone(value, 10, 20) == expected


def test_a_range_with_one_end_has_no_below_or_above_on_the_other_side():
    assert zone(3, None, 15) == "within"
    assert zone("<12", None, 15) == "within"
    assert zone(16, None, 15) == "above"
    assert zone(0.5, 1, None) == "below"


def test_every_sample_marker_with_a_range_gets_the_position_the_lab_printed():
    data = load_lab_reports([SAMPLES])
    patients = lab_range_findings(data).patients
    assert len(patients) == len(data.markers)  # every sample row has a range
    # the lab's own status, e.g. "Below luteal range", "Low end of range" (still within)
    printed = data.markers.set_index(["record", "marker_id"])["status"].str.split().str[0]
    expected = printed.map({"Below": "below", "Above": "above"}).fillna("within")
    got = patients.set_index(["record", "marker_id"])["zone"]
    assert got.sort_index().tolist() == expected.sort_index().tolist()


def test_position_is_0_at_the_low_end_and_1_at_the_high_end():
    patients = lab_range_findings(load_lab_reports([SAMPLES])).patients
    row = patients.set_index(["record", "marker_id"]).loc[("1/r001", "b_pregnanediol")]
    assert row["position"] == pytest.approx((196 - 600) / (2000 - 600))


HEADER = "record_id,sex,age_years,collection_month,report_month,marker_id,marker_name,category,comparator,value,unit,ref_low,ref_high,status"


def small_dataset(tmp_path):
    """Five PSSD men with marker x (range 10-20) and y (no range), one healthy-control man."""
    rows = [HEADER]
    for n, (comparator, value) in enumerate([("", 5), ("", 15), ("", 15), ("", 25), ("<", 15), ("", 12)], 1):
        rows.append(f"r{n:03},male,30,,,x,X,,{comparator},{value},ng/mg,10,20,")
        rows.append(f"r{n:03},male,30,,,y,Y,,,{value},ng/mg,,,")
    reports = tmp_path / "reports.csv"
    reports.write_text("\n".join(rows) + "\n")
    meta = tmp_path / "meta.csv"
    groups = ["PSSD"] * 5 + ["healthy control"]
    meta.write_text("record_id,group\n" + "".join(f"r{n:03},{g}\n" for n, g in enumerate(groups, 1)))
    return load_lab_reports([reports], metadata=meta)


def test_each_group_gets_its_share_below_within_and_above_with_an_interval(tmp_path):
    rows = lab_range_findings(small_dataset(tmp_path)).rows
    pssd = rows[(rows["group_a"] == "PSSD") & (rows["marker_id"] == "x")].iloc[0]
    assert pssd["comparison"] == "reference range"
    assert (pssd["sex"], pssd["n_a"], pssd["n_unclear"]) == ("male", 4, 1)  # "<15" may be below or within
    assert (pssd["share_below"], pssd["share_within"], pssd["share_above"]) == (0.25, 0.5, 0.25)
    # Wilson score interval for 1 of 4
    assert (pssd["below_low"], pssd["below_high"]) == pytest.approx((0.0456, 0.6994), abs=1e-4)
    assert set(rows["group_a"]) == {"PSSD", "healthy control"}


def test_markers_without_a_range_are_skipped_and_reported(tmp_path):
    findings = lab_range_findings(small_dataset(tmp_path))
    assert "y" not in set(findings.rows["marker_id"]) | set(findings.patients["marker_id"])
    assert findings.skipped.to_dict("records") == [{"marker_id": "y", "marker_name": "Y", "n_values": 6}]


def test_reference_range_rows_are_marked_apart_from_group_comparisons_in_the_findings_table(tmp_path):
    data = small_dataset(tmp_path)
    table = combine_findings(
        compare_groups(data, "PSSD", "healthy control", bootstrap=10), lab_range_findings(data).rows
    )
    assert table["comparison"].value_counts().to_dict() == {"group comparison": 2, "reference range": 2}
    assert table.loc[table["comparison"] == "reference range", "effect_size"].isna().all()


def self_reports(tmp_path):
    """The fixture export plus typed-in values: author 1 (PSSD man) gives testosterone twice, once in
    range and once above it; author 2 (PAS woman) once in range; the man with no condition gives DHT with no range."""
    export = shutil.copytree(FIXTURES / "selfreport", tmp_path / "export")
    with open(export / "labs.csv", "a") as labs:
        labs.write("c02,Total  Testosterone ,1000,ng/dL,264,916,high,blood\n")
        labs.write("c03,total testosterone,30,ng/dL,15,70,normal,blood\n")
        labs.write("c09,dht,40,pg/mL,,,,\n")
    return load_self_reports(export)


def test_typed_in_values_with_a_range_are_placed_per_author_as_self_report_rows(tmp_path):
    findings = self_report_range_findings(self_reports(tmp_path))
    rows = findings.rows.set_index(["marker_id", "group_a"])
    assert set(findings.rows["comparison"]) == {"reference range, self-report"}
    # author 1's two testosterone values disagree, so they count once, as unclear
    assert rows.loc[("total testosterone", "PSSD"), ["n_a", "n_unclear"]].tolist() == [0, 1]
    women = rows.loc[("total testosterone", "PAS")]
    assert (women["sex"], women["n_a"], women["share_within"]) == ("female", 1, 1.0)
    # a range with only a high end still places a value above it
    assert rows.loc[("prolactin", "PSSD"), "share_above"] == 1.0
    # dht is skipped (that man's value has no range; the PFS post is unverified, so not loaded)
    assert findings.skipped.to_dict("records") == [{"marker_id": "dht", "marker_name": "dht", "n_values": 1}]


def test_a_typed_in_value_at_a_range_end_follows_its_flag_like_a_lab_status(tmp_path):
    export = shutil.copytree(FIXTURES / "selfreport", tmp_path / "export")
    with open(export / "labs.csv", "a") as labs:
        labs.write("c03,shbg,70,nmol/L,15,70,high,blood\n")  # probably rounded down to the range end
        labs.write("c03,lh,15,IU/L,15,70,,blood\n")  # no flag: the end is in the range
    rows = self_report_range_findings(load_self_reports(export)).rows.set_index("marker_id")
    assert rows.loc["shbg", "share_above"] == 1.0
    assert rows.loc["lh", "share_within"] == 1.0
