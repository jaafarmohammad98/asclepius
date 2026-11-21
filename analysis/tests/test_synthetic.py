import json

import numpy as np
import pandas as pd
import pytest

from asclepius.dataset import load_lab_reports
from asclepius.selfreport import load_self_reports
from asclepius.synthetic import (
    PlantedAssociation,
    PlantedDifference,
    PlantedOutcome,
    write_lab_reports,
    write_self_reports,
)


def test_same_seed_gives_the_same_lab_file(tmp_path):
    a = write_lab_reports(tmp_path / "a", seed=7)
    b = write_lab_reports(tmp_path / "b", seed=7)
    c = write_lab_reports(tmp_path / "c", seed=8)
    assert a.reports.read_bytes() == b.reports.read_bytes()
    assert a.reports.read_bytes() != c.reports.read_bytes()


def test_lab_file_loads_through_the_dataset_loader(tmp_path):
    out = write_lab_reports(tmp_path, groups={"PSSD": 12, "healthy control": 15}, seed=1)
    data = load_lab_reports([out.reports])
    assert data.markers["record"].nunique() == 27


def _log_mean_ratio(out, marker, sex, group="PSSD", other="healthy control"):
    """Geometric mean of `group` over `other` for one marker within one sex, from loaded data."""
    data = load_lab_reports([out.reports]).markers
    groups = pd.read_csv(out.groups).set_index("record_id")["group"]
    rows = data[(data["marker_id"] == marker) & (data["sex"] == sex)]
    logs = np.log(rows["value"]).groupby(rows["record_id"].map(groups)).mean()
    return float(np.exp(logs[group] - logs[other]))


def test_planted_difference_appears_at_its_size_within_each_sex(tmp_path):
    planted = [PlantedDifference(marker="androsterone", fold=0.5)]
    big = {"PSSD": 400, "healthy control": 400}
    out = write_lab_reports(tmp_path, groups=big, planted=planted, female_share=0.5, seed=3)
    for sex in ("male", "female"):
        assert _log_mean_ratio(out, "androsterone", sex) == pytest.approx(0.5, rel=0.12)
        assert _log_mean_ratio(out, "etiocholanolone", sex) == pytest.approx(1.0, rel=0.12)


def test_planted_difference_can_be_limited_to_one_sex(tmp_path):
    planted = [PlantedDifference(marker="testosterone", fold=2.0, sex="female")]
    big = {"PSSD": 400, "healthy control": 400}
    out = write_lab_reports(tmp_path, groups=big, planted=planted, female_share=0.5, seed=4)
    assert _log_mean_ratio(out, "testosterone", "female") == pytest.approx(2.0, rel=0.12)
    assert _log_mean_ratio(out, "testosterone", "male") == pytest.approx(1.0, rel=0.12)


def test_null_mode_has_no_differences_and_says_so(tmp_path):
    # large enough that sampling noise on the widest range (2-OH-E1 in men) stays well under 12%
    big = {"PSSD": 1500, "healthy control": 1500}
    out = write_lab_reports(tmp_path, groups=big, null=True, female_share=0.5, seed=5)
    for marker in ("androsterone", "24hr_free_cortisol", "2_oh_e1"):
        assert _log_mean_ratio(out, marker, "male") == pytest.approx(1.0, rel=0.12)
    truth = json.loads(out.truth.read_text())
    assert truth["null"] is True
    assert truth["planted"] == []
    with pytest.raises(ValueError, match="Null mode"):
        write_lab_reports(tmp_path, null=True, planted=[PlantedDifference("androsterone", 0.5)])


def test_values_follow_each_sexs_reference_range(tmp_path):
    out = write_lab_reports(tmp_path, groups={"healthy control": 600}, female_share=0.5, seed=6)
    data = load_lab_reports([out.reports]).markers
    testosterone = data[data["marker_id"] == "testosterone"]
    # DUTCH ranges: men 25-115 ng/mg, women 2.3-14 ng/mg; about 95% of healthy values fall inside
    for sex, (low, high) in {"male": (25, 115), "female": (2.3, 14)}.items():
        rows = testosterone[testosterone["sex"] == sex]
        assert (rows["ref_low"] == low).all() and (rows["ref_high"] == high).all()
        inside = rows["value"].between(low, high).mean()
        assert 0.9 < inside < 0.99


def test_unknown_marker_or_group_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown marker"):
        write_lab_reports(tmp_path, planted=[PlantedDifference("cortisol", 0.5)])
    with pytest.raises(ValueError, match="Unknown group"):
        write_lab_reports(tmp_path, groups={"PSSD": 10, "controls": 10})


# ---------------------------------------------------------------- self-reports

TABLES = ("cases", "causal_drugs", "symptoms", "interventions", "labs")


def test_same_seed_gives_the_same_self_report_tables(tmp_path):
    a = write_self_reports(tmp_path / "a", authors=80, seed=7)
    b = write_self_reports(tmp_path / "b", authors=80, seed=7)
    c = write_self_reports(tmp_path / "c", authors=80, seed=8)
    for name in TABLES:
        assert (a.export / f"{name}.csv").read_bytes() == (b.export / f"{name}.csv").read_bytes()
    assert (a.export / "cases.csv").read_bytes() != (c.export / "cases.csv").read_bytes()


def test_self_report_tables_load_with_repeat_posts_merged_per_author(tmp_path):
    out = write_self_reports(tmp_path, authors=200, repeat_share=0.4, deleted_posts=12, seed=1)
    data = load_self_reports(out.export)
    assert len(data.authors) == 200
    assert data.excluded == {"no_author": 12, "unverified": 0}
    repeaters = (data.authors["n_cases"] > 1).mean()
    assert 0.3 < repeaters < 0.5
    cases = pd.read_csv(out.export / "cases.csv")
    assert len(cases) > 200 + 12


def _odds_ratio(data, drug, symptom):
    took = set(data.causal_drugs.loc[data.causal_drugs["drug"] == drug, "author"])
    has = set(data.symptoms.loc[data.symptoms["symptom"] == symptom, "author"])
    everyone = set(data.authors["author"])
    a = len(took & has)
    b = len(took - has)
    c = len(has - took)
    d = len(everyone - took - has)
    return (a * d) / (b * c)


def _shares(data, intervention):
    rows = data.interventions[data.interventions["intervention"] == intervention]["outcome"]
    improved = rows.isin(["major_improvement", "partial_improvement"]).mean()
    return improved, (rows == "worse").mean()


def test_planted_drug_symptom_association_is_recovered_per_author(tmp_path):
    planted = [PlantedAssociation(drug="finasteride", symptom="erectile_dysfunction", odds_ratio=4.0)]
    out = write_self_reports(tmp_path, authors=4000, associations=planted, seed=2)
    data = load_self_reports(out.export)
    assert _odds_ratio(data, "finasteride", "erectile_dysfunction") == pytest.approx(4.0, rel=0.35)
    assert _odds_ratio(data, "finasteride", "brain_fog") == pytest.approx(1.0, rel=0.35)
    assert _odds_ratio(data, "sertraline", "erectile_dysfunction") == pytest.approx(1.0, rel=0.35)


def test_planted_intervention_outcome_shares_are_recovered_per_author(tmp_path):
    planted = [PlantedOutcome(intervention="tadalafil", improved=0.6, worse=0.05)]
    out = write_self_reports(tmp_path, authors=4000, outcomes=planted, seed=3)
    data = load_self_reports(out.export)
    improved, worse = _shares(data, "tadalafil")
    assert improved == pytest.approx(0.6, abs=0.06)
    assert worse == pytest.approx(0.05, abs=0.04)
    improved, worse = _shares(data, "bupropion")  # not planted: the default shares
    assert improved == pytest.approx(0.25, abs=0.06)
    assert worse == pytest.approx(0.15, abs=0.06)


def test_self_report_null_mode_plants_nothing(tmp_path):
    out = write_self_reports(tmp_path, authors=4000, null=True, seed=4)
    data = load_self_reports(out.export)
    assert _odds_ratio(data, "finasteride", "erectile_dysfunction") == pytest.approx(1.0, rel=0.35)
    for intervention in ("tadalafil", "exercise"):
        improved, _ = _shares(data, intervention)
        assert improved == pytest.approx(0.25, abs=0.06)
    truth = json.loads(out.truth.read_text())
    assert truth["null"] is True and truth["associations"] == [] and truth["outcomes"] == []
    with pytest.raises(ValueError, match="Null mode"):
        write_self_reports(tmp_path, null=True, outcomes=[PlantedOutcome("tadalafil", 0.5, 0.1)])


def test_unknown_drug_symptom_or_intervention_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown drug"):
        write_self_reports(tmp_path, associations=[PlantedAssociation("aspirin", "brain_fog", 2)])
    with pytest.raises(ValueError, match="Unknown symptom"):
        write_self_reports(tmp_path, associations=[PlantedAssociation("sertraline", "hiccups", 2)])
    with pytest.raises(ValueError, match="add to at most 1"):
        write_self_reports(tmp_path, outcomes=[PlantedOutcome("tadalafil", 0.8, 0.4)])
