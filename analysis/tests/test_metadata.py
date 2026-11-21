from pathlib import Path

import pytest

from asclepius.dataset import DatasetError, load_lab_reports

SAMPLES = Path(__file__).parent / "fixtures" / "dutch_samples.csv"  # records r001, r002, r004

HEADER = (
    "record_id,sex,age_years,collection_month,report_month,marker_id,marker_name,category,"
    "comparator,value,unit,ref_low,ref_high,status"
)


def report_file(tmp_path, name, *record_ids):
    rows = [
        f"{r},male,30,2024-01,2024-02,cortisol,Cortisol,Glucocorticoids,,12.5,ng/mg,5,20," for r in record_ids
    ]
    path = tmp_path / name
    path.write_text("\n".join([HEADER, *rows]) + "\n")
    return path


def metadata_file(tmp_path, *lines, name="metadata.csv"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def records(data):
    return data.records.set_index("record")


def test_without_metadata_every_record_is_unassigned_tier_b():
    data = load_lab_reports([SAMPLES])
    table = records(data)
    assert sorted(table.index) == ["1/r001", "1/r002", "1/r004"]
    assert set(table["group"]) == {"unassigned"}
    assert set(table["tier"]) == {"B"}
    assert table.at["1/r001", "sex"] == "female"


def test_metadata_gives_group_tier_drug_months_and_scores(tmp_path):
    meta = metadata_file(
        tmp_path,
        "record_id,group,tier,drug,months_since_stopping,score_sexual,score_cognitive",
        "r001,PSSD,A,sertraline,18,7,4",
        "r002,healthy control,B,,,,",
    )
    table = records(load_lab_reports([SAMPLES], metadata=meta))
    assert table.at["1/r001", "group"] == "PSSD"
    assert table.at["1/r001", "tier"] == "A"
    assert table.at["1/r001", "drug"] == "sertraline"
    assert table.at["1/r001", "months_since_stopping"] == 18
    assert table.at["1/r001", "score_sexual"] == 7
    assert table.at["1/r002", "group"] == "healthy control"
    assert table.isna().at["1/r002", "drug"]
    assert table.isna().at["1/r002", "score_cognitive"]


def test_record_missing_from_metadata_goes_to_unassigned(tmp_path):
    meta = metadata_file(tmp_path, "record_id,group", "r001,PFS")
    table = records(load_lab_reports([SAMPLES], metadata=meta))
    assert table.at["1/r001", "group"] == "PFS"
    assert table.at["1/r002", "group"] == "unassigned"
    assert table.at["1/r004", "group"] == "unassigned"
    assert set(table["tier"]) == {"B"}  # tier column absent: default B


def test_every_record_has_a_group_and_a_tier(tmp_path):
    meta = metadata_file(tmp_path, "record_id,group,tier", "r001,PAS,", "r002,,C")
    table = records(load_lab_reports([SAMPLES], metadata=meta))
    assert table["group"].notna().all() and table["tier"].notna().all()
    assert table.at["1/r001", "tier"] == "B"
    assert table.at["1/r002", "group"] == "unassigned"
    assert table.at["1/r002", "tier"] == "C"


def test_group_names_are_matched_ignoring_case_and_spacing(tmp_path):
    meta = metadata_file(tmp_path, "record_id,group", "r001,  ssri-EXPOSED   control ", "r002,pssd")
    table = records(load_lab_reports([SAMPLES], metadata=meta))
    assert table.at["1/r001", "group"] == "SSRI-exposed control"
    assert table.at["1/r002", "group"] == "PSSD"


@pytest.mark.parametrize("group", ["controls", "unassigned", "Post-SSRI"])
def test_unknown_group_name_is_rejected(tmp_path, group):
    meta = metadata_file(tmp_path, "record_id,group", "r001,PSSD", f"r002,{group}")
    with pytest.raises(DatasetError, match="metadata.csv, line 3, column group") as err:
        load_lab_reports([SAMPLES], metadata=meta)
    assert "healthy control" in str(err.value)  # names the allowed list


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("r001,PSSD,D,,,", "column tier: 'D' is not one of A, B, C"),
        ("r001,PSSD,A,,-3,", "column months_since_stopping: '-3' is not"),
        ("r001,PSSD,A,,soon,", "column months_since_stopping: 'soon' is not"),
        ("r001,PSSD,A,,,lots", "column score_sexual: 'lots' is not a number"),
        (",PSSD,A,,,", "column record_id: is empty"),
    ],
)
def test_bad_metadata_value_is_rejected_naming_line_and_column(tmp_path, line, expected):
    meta = metadata_file(tmp_path, "record_id,group,tier,drug,months_since_stopping,score_sexual", line)
    with pytest.raises(DatasetError, match="metadata.csv, line 2") as err:
        load_lab_reports([SAMPLES], metadata=meta)
    assert expected in str(err.value)


def test_metadata_needs_record_id_and_group_columns_and_no_unknown_ones(tmp_path):
    with pytest.raises(DatasetError, match="missing column"):
        load_lab_reports([SAMPLES], metadata=metadata_file(tmp_path, "record_id,tier", "r001,A"))
    with pytest.raises(DatasetError, match="unexpected column.*sex"):
        load_lab_reports([SAMPLES], metadata=metadata_file(tmp_path, "record_id,group,sex", "r001,PSSD,male"))


def test_extra_metadata_rows_are_left_out_and_reported(tmp_path):
    meta = metadata_file(tmp_path, "record_id,group", "r001,PSSD", "r099,PFS", "r100,PAS")
    data = load_lab_reports([SAMPLES], metadata=meta)
    assert len(data.records) == 3
    assert data.unmatched == ("r099", "r100")


def test_identical_duplicate_rows_are_accepted(tmp_path):
    meta = metadata_file(tmp_path, "record_id,group,tier", "r001,PSSD,A", "r001,pssd,A")
    table = records(load_lab_reports([SAMPLES], metadata=meta))
    assert table.at["1/r001", "group"] == "PSSD"


def test_conflicting_rows_are_rejected_naming_both_lines(tmp_path):
    meta = metadata_file(tmp_path, "record_id,group,tier", "r001,PSSD,A", "r002,PFS,B", "r001,PFS,A")
    with pytest.raises(DatasetError, match="metadata.csv: lines 2 and 4 give record r001 different group"):
        load_lab_reports([SAMPLES], metadata=meta)


def test_bare_record_id_matches_when_only_one_file_has_it(tmp_path):
    a = report_file(tmp_path, "a.csv", "r001", "r002")
    b = report_file(tmp_path, "b.csv", "r001", "r003")
    meta = metadata_file(tmp_path, "record_id,group", "r003,PSSD", "2/r001,PFS", "1/r001,PAS")
    table = records(load_lab_reports([a, b], metadata=meta))
    assert table.at["2/r003", "group"] == "PSSD"
    assert table.at["2/r001", "group"] == "PFS"
    assert table.at["1/r001", "group"] == "PAS"
    assert table.at["1/r002", "group"] == "unassigned"


def test_bare_record_id_in_several_files_is_ambiguous(tmp_path):
    a = report_file(tmp_path, "a.csv", "r001")
    b = report_file(tmp_path, "b.csv", "r001")
    meta = metadata_file(tmp_path, "record_id,group", "r001,PSSD")
    with pytest.raises(DatasetError, match=r"line 2.*r001 is in more than one file.*1/r001 or 2/r001"):
        load_lab_reports([a, b], metadata=meta)


def test_qualified_and_bare_ids_for_the_same_record_must_agree(tmp_path):
    meta = metadata_file(tmp_path, "record_id,group", "r001,PSSD", "1/r001,PFS")
    with pytest.raises(DatasetError, match="lines 2 and 3 give record 1/r001 different group"):
        load_lab_reports([SAMPLES], metadata=meta)


def test_metadata_changes_the_fingerprint(tmp_path):
    one = metadata_file(tmp_path, "record_id,group", "r001,PSSD", name="one.csv")
    two = metadata_file(tmp_path, "record_id,group", "r001,PFS", name="two.csv")
    plain = load_lab_reports([SAMPLES]).fingerprint
    assert load_lab_reports([SAMPLES], metadata=one).fingerprint != plain
    assert (
        load_lab_reports([SAMPLES], metadata=one).fingerprint
        != load_lab_reports([SAMPLES], metadata=two).fingerprint
    )


def test_empty_metadata_file_is_rejected(tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text("")
    with pytest.raises(DatasetError, match="metadata.csv: the file is empty"):
        load_lab_reports([SAMPLES], metadata=path)


def test_line_numbers_count_blank_lines(tmp_path):
    meta = metadata_file(tmp_path, "record_id,group", "", "r001,PSSD", "r002,bogus")
    with pytest.raises(DatasetError, match="line 4, column group"):
        load_lab_reports([SAMPLES], metadata=meta)


def test_missing_metadata_file_is_a_dataset_error(tmp_path):
    with pytest.raises(DatasetError, match="nothere.csv: the file does not exist"):
        load_lab_reports([SAMPLES], metadata=tmp_path / "nothere.csv")


def test_score_names_may_use_capitals(tmp_path):
    meta = metadata_file(tmp_path, "record_id,group,score_HAMD", "r001,PSSD,12")
    table = records(load_lab_reports([SAMPLES], metadata=meta))
    assert table.at["1/r001", "score_HAMD"] == 12
