import shutil
from pathlib import Path

import pandas as pd
import pytest

from asclepius.dataset import DatasetError
from asclepius.selfreport import load_self_reports

EXPORT = Path(__file__).parent / "fixtures" / "selfreport"  # hand-made, not real posts


def test_each_author_counts_once():
    data = load_self_reports(EXPORT)
    assert data.kind == "self-report"
    assert len(data.authors) == 5
    assert data.authors["author"].is_unique


def test_unverified_cases_are_excluded_unless_asked_for():
    data = load_self_reports(EXPORT, include_unverified=True)
    assert len(data.authors) == 6
    assert "PFS" in set(data.authors["condition"])


def by_drug(data, drug):
    """The one author whose causal drugs include `drug` (the fixture gives each drug to one author)."""
    authors = data.causal_drugs.loc[data.causal_drugs["drug"] == drug, "author"]
    assert len(authors) == 1
    return data.authors.set_index("author").loc[authors.iloc[0]]


def test_conditions_map_to_the_plans_groups():
    data = load_self_reports(EXPORT)
    assert by_drug(data, "sertraline")["condition"] == "PSSD"
    assert by_drug(data, "isotretinoin")["condition"] == "PAS"  # the harvester's PSA
    assert by_drug(data, "escitalopram")["condition"] == "multiple"  # PSSD in one post, PFS in another
    assert sorted(data.authors["condition"]) == ["PAS", "PSSD", "multiple", "none", "unclear"]


def test_an_authors_posts_merge_into_one_profile():
    a = by_drug(load_self_reports(EXPORT), "sertraline")  # posted in 2022 and 2024
    assert a["n_cases"] == 2
    assert (a["first_year"], a["last_year"]) == (2022, 2024)
    assert a["sex"] == "male"  # given once, missing once
    assert a["onset"] == "after_stopping"
    assert a["overall_trajectory"] == "improving"  # the 2024 post, not the 2022 one
    assert a["age"] == 26
    assert a["months_since_stopping"] == 30


def test_conflicting_or_missing_values_stay_unknown():
    d = by_drug(load_self_reports(EXPORT), "escitalopram")
    assert pd.isna(d["sex"]) and pd.isna(d["onset"]) and pd.isna(d["age"])
    assert d["overall_trajectory"] == "stable"


def items_of(data, table, drug, *columns):
    author = by_drug(data, drug).name
    frame = getattr(data, table)
    return frame.loc[frame["author"] == author, list(columns)].to_dict("records")


def test_a_symptom_reported_twice_counts_once_at_its_worst_severity():
    data = load_self_reports(EXPORT)
    rows = items_of(data, "symptoms", "sertraline", "symptom", "severity")
    assert [row["symptom"] for row in rows] == ["emotional_blunting", "genital_numbness"]
    assert pd.isna(rows[0]["severity"]) and rows[1]["severity"] == "severe"
    assert list(data.symptoms["severity"].cat.categories) == ["mild", "moderate", "severe"]


def test_an_intervention_tried_twice_keeps_the_latest_clear_outcome():
    data = load_self_reports(EXPORT)
    assert items_of(data, "interventions", "sertraline", "intervention", "outcome") == [
        {"intervention": "cold showers", "outcome": "unclear"},
        {"intervention": "tadalafil", "outcome": "partial_improvement"},
    ]
    assert items_of(data, "interventions", "escitalopram", "intervention", "outcome") == [
        {"intervention": "bupropion", "outcome": "worse"}
    ]


def test_typed_in_lab_values_are_kept_per_author():
    data = load_self_reports(EXPORT)
    assert [
        tuple(lab.values()) for lab in items_of(data, "labs", "sertraline", "test", "value", "ref_high")
    ] == [
        ("prolactin", 22.0, 15.0),
        ("total testosterone", 450.0, 916.0),
    ]
    assert len(data.labs) == 2  # the unverified case's lab is left out


IDS = ("c0", "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb")


def broken_export(tmp_path, table, old, new):
    """A copy of the fixture with one line of one table changed."""
    export = tmp_path / "export"
    shutil.copytree(EXPORT, export)
    path = export / f"{table}.csv"
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))
    return export


@pytest.mark.parametrize(
    ("table", "old", "new", "expected"),
    [
        ("cases", "PSA,female", "PAS,female", "cases.csv, line 4, column condition: 'PAS' is not one of"),
        (
            "cases",
            "unclear,True",
            "unclear,yes",
            "cases.csv, line 4, column evidence_ok: 'yes' is not one of",
        ),
        (
            "cases",
            "c03,bbbbbbbbbbbbbbbb,2023",
            "c03,bbbbbbbbbbbbbbbb,23rd",
            "line 4, column year: '23rd' is not a year",
        ),
        ("cases", "female,,", "female,old,", "line 4, column age: 'old' is not a number"),
        ("cases", "c03,", "c02,", "cases.csv, line 4: the same case_id appears on an earlier line"),
        ("cases", "c03,bbbbbbbbbbbbbbbb", ",bbbbbbbbbbbbbbbb", "cases.csv, line 4, column case_id: is empty"),
        (
            "symptoms",
            "low_libido,moderate",
            "low_libido,very",
            "symptoms.csv, line 5, column severity: 'very'",
        ),
        (
            "symptoms",
            "c03,low_libido",
            "c99,low_libido",
            "symptoms.csv, line 5: refers to a case not in cases.csv",
        ),
        ("interventions", "zinc,worse", "zinc,cured", "interventions.csv, line 5, column outcome: 'cured'"),
        ("labs", "450.0", "lots", "labs.csv, line 2, column value: 'lots' is not a number"),
        ("causal_drugs", "c03,isotretinoin", "c03,", "causal_drugs.csv, line 4, column drug: is empty"),
    ],
)
def test_a_malformed_export_is_rejected_without_naming_a_case_or_author(tmp_path, table, old, new, expected):
    export = broken_export(tmp_path, table, old, new)
    with pytest.raises(DatasetError) as err:
        load_self_reports(export)
    message = str(err.value)
    assert expected in message
    assert not any(i in message for i in IDS + ("c99",))


def test_a_missing_table_is_named(tmp_path):
    export = tmp_path / "export"
    shutil.copytree(EXPORT, export)
    (export / "labs.csv").unlink()
    with pytest.raises(DatasetError, match="labs.csv is missing"):
        load_self_reports(export)


def test_posts_without_an_author_are_left_out_and_counted():
    data = load_self_reports(EXPORT)
    assert "paroxetine" not in set(data.causal_drugs["drug"])  # its only post has no author (deleted account)
    assert data.excluded == {"no_author": 1, "unverified": 1}


def test_a_post_without_a_year_or_condition_still_loads(tmp_path):
    export = broken_export(
        tmp_path, "cases", "c08,eeeeeeeeeeeeeeee,2025,comment,unclear", "c08,eeeeeeeeeeeeeeee,,comment,"
    )
    data = load_self_reports(export)
    assert len(data.authors) == 5
    assert "unclear" in set(data.authors["condition"])


def test_the_dataset_records_its_version_and_settings(tmp_path):
    data = load_self_reports(EXPORT)
    assert len(data.fingerprint) == 64
    assert [(f.role, f.name, len(f.sha256)) for f in data.files] == [
        (name, f"{name}.csv", 64) for name in ("cases", "causal_drugs", "symptoms", "interventions", "labs")
    ]
    assert data.settings == {"include_unverified": False}
    assert load_self_reports(EXPORT, include_unverified=True).settings == {"include_unverified": True}
    changed = broken_export(tmp_path, "labs", "450.0", "451.0")
    assert load_self_reports(changed).fingerprint != data.fingerprint
