from pathlib import Path

import pytest

from asclepius.dataset import DatasetError, load_lab_reports

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = FIXTURES / "dutch_samples.csv"  # `npm run extract -- --csv` on the public samples


def test_extractor_csv_loads_without_changes():
    data = load_lab_reports([SAMPLES])
    assert data.kind == "lab-report"
    assert len(data.markers) == 140
    assert data.markers["record"].nunique() == 3
    first = data.markers.iloc[0]
    assert first["marker_id"] == "b_pregnanediol"
    assert first["value"] == 196.0
    assert first["unit"] == "ng/mg"
    assert first["sex"] == "female"
    assert first["age_years"] == 26


HEADER = (
    "record_id,sex,age_years,collection_month,report_month,marker_id,marker_name,category,"
    "comparator,value,unit,ref_low,ref_high,status"
)
GOOD_ROW = "r001,male,30,2024-01,2024-02,cortisol,Cortisol,Glucocorticoids,,12.5,ng/mg,5,20,Within range"


def write(tmp_path, name, *rows, header=HEADER):
    path = tmp_path / name
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_records_from_several_files_stay_apart(tmp_path):
    a = write(tmp_path, "a.csv", GOOD_ROW)
    b = write(tmp_path, "b.csv", GOOD_ROW)
    data = load_lab_reports([a, b])
    assert sorted(data.markers["record"]) == ["1/r001", "2/r001"]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (GOOD_ROW.replace("12.5", "twelve"), "line 2, column value: 'twelve' is not a number"),
        (GOOD_ROW.replace("12.5", ""), "line 2, column value: is empty"),
        (GOOD_ROW.replace("ng/mg", "ng per mg"), "column unit: 'ng per mg' is not a unit"),
        (GOOD_ROW.replace("male", "M"), "column sex: 'M' is not one of"),
        (GOOD_ROW.replace(",30,", ",30.5,"), "column age_years: '30.5' is not an age"),
        (GOOD_ROW.replace("2024-01", "Jan 2024"), "column collection_month: 'Jan 2024' is not a month"),
        (GOOD_ROW.replace(",5,20,", ",20,5,"), "low end is above its high end"),
    ],
)
def test_bad_value_is_rejected_naming_file_line_and_column(tmp_path, row, expected):
    path = write(tmp_path, "bad.csv", GOOD_ROW, row)
    with pytest.raises(DatasetError, match="bad.csv") as err:
        load_lab_reports([path])
    assert expected.replace("line 2", "line 3") in str(err.value)


def test_missing_column_is_named(tmp_path):
    path = write(tmp_path, "bad.csv", "r001,male", header="record_id,sex")
    with pytest.raises(DatasetError, match="missing column.*marker_id"):
        load_lab_reports([path])


def test_unexpected_column_is_named(tmp_path):
    path = write(tmp_path, "bad.csv", GOOD_ROW + ",x", header=HEADER + ",patient_name")
    with pytest.raises(DatasetError, match="unexpected column.*patient_name"):
        load_lab_reports([path])


def test_empty_and_header_only_files_are_rejected(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DatasetError, match="empty.csv: the file is empty"):
        load_lab_reports([empty])
    with pytest.raises(DatasetError, match="no rows"):
        load_lab_reports([write(tmp_path, "header.csv")])


def test_mixed_units_for_one_marker_are_rejected(tmp_path):
    a = write(tmp_path, "a.csv", GOOD_ROW)
    b = write(tmp_path, "b.csv", GOOD_ROW.replace("ng/mg", "ug/mg"))
    with pytest.raises(
        DatasetError, match=r"cortisol is recorded in more than one unit \(ng/mg in a.csv; ug/mg in b.csv\)"
    ):
        load_lab_reports([a, b])


def test_any_unit_the_extractor_reads_is_accepted(tmp_path):
    path = write(tmp_path, "a.csv", GOOD_ROW.replace("ng/mg", "µg/mg"), GOOD_ROW.replace("cortisol", "dhea"))
    assert list(load_lab_reports([path]).markers["unit"]) == ["µg/mg", "ng/mg"]


def test_a_repeated_identical_row_is_kept_once(tmp_path):
    path = write(tmp_path, "a.csv", GOOD_ROW, GOOD_ROW)
    assert len(load_lab_reports([path]).markers) == 1


def test_two_different_rows_for_one_marker_in_one_report_are_rejected(tmp_path):
    other = GOOD_ROW.replace(",12.5,", ",30,")
    path = write(tmp_path, "a.csv", GOOD_ROW, other)
    with pytest.raises(
        DatasetError, match=r"a.csv, line 3, column value: r001 already has a row for cortisol \(line 2\)"
    ):
        load_lab_reports([path])


def test_one_report_must_give_one_sex_and_age(tmp_path):
    second = GOOD_ROW.replace("cortisol,Cortisol", "cortisone,Cortisone")
    with pytest.raises(DatasetError, match=r"a.csv, line 3, column sex: r001 is male on line 2"):
        load_lab_reports([write(tmp_path, "a.csv", GOOD_ROW, second.replace(",male,", ",female,"))])
    with pytest.raises(DatasetError, match=r"b.csv, line 3, column age_years: r001 is 30 on line 2"):
        load_lab_reports([write(tmp_path, "b.csv", GOOD_ROW, second.replace(",30,", ",31,"))])
    # a blank cell is "not given", not a disagreement
    blank = second.replace(",male,", ",,").replace(",30,", ",,")
    assert len(load_lab_reports([write(tmp_path, "c.csv", GOOD_ROW, blank)]).markers) == 2
