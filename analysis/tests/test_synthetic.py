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
    PlantedScore,
    PlantedSubgroup,
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
    data = load_lab_reports([out.reports], metadata=out.metadata)
    assert data.markers["record"].nunique() == 27
    assert data.unmatched == ()
    assert data.records["group"].value_counts().to_dict() == {"healthy control": 15, "PSSD": 12}
    assert set(data.records["tier"]) == {"A", "B"}


def test_about_tier_a_share_of_records_are_tier_a(tmp_path):
    out = write_lab_reports(tmp_path, groups={"PSSD": 200, "healthy control": 200}, tier_a_share=0.25, seed=2)
    tiers = load_lab_reports([out.reports], metadata=out.metadata).records["tier"]
    assert 0.18 < (tiers == "A").mean() < 0.32


def _log_mean_ratio(out, marker, sex, group="PSSD", other="healthy control"):
    """Geometric mean of `group` over `other` for one marker within one sex, from loaded data."""
    data = load_lab_reports([out.reports], metadata=out.metadata)
    groups = data.records.set_index("record")["group"]
    rows = data.markers[(data.markers["marker_id"] == marker) & (data.markers["sex"] == sex)]
    logs = np.log(rows["value"]).groupby(rows["record"].map(groups)).mean()
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


def test_symptom_scores_and_drugs_go_in_the_metadata(tmp_path):
    groups = {"PSSD": 300, "PFS": 300}
    planted = [PlantedScore("libido", 3.0, group="PFS")]
    drugs = {"PSSD": ["sertraline", "paroxetine"], "PFS": ["finasteride"]}
    out = write_lab_reports(
        tmp_path, groups=groups, scores=["libido", "fatigue"], planted_scores=planted, drugs=drugs, seed=1
    )
    records = load_lab_reports([out.reports], metadata=out.metadata).records
    assert records[["score_libido", "score_fatigue"]].stack().between(0, 10).all()
    means = records.groupby("group")[["score_libido", "score_fatigue"]].mean()
    assert means.loc["PFS", "score_libido"] - means.loc["PSSD", "score_libido"] == pytest.approx(3, abs=0.4)
    assert abs(means.loc["PFS", "score_fatigue"] - means.loc["PSSD", "score_fatigue"]) < 0.4
    taken = records.groupby("group")["drug"].agg(lambda d: set(d))
    assert taken["PSSD"] == {"sertraline", "paroxetine"} and taken["PFS"] == {"finasteride"}
    assert json.loads(out.truth.read_text())["planted_scores"] == [
        {"score": "libido", "shift": 3.0, "group": "PFS"}
    ]


def test_scores_and_drugs_leave_the_reports_unchanged(tmp_path):
    plain = write_lab_reports(tmp_path / "a", seed=2)
    scored = write_lab_reports(tmp_path / "b", seed=2, scores=["libido"], drugs={"PSSD": ["sertraline"]})
    assert plain.reports.read_bytes() == scored.reports.read_bytes()
    assert plain.metadata.read_text().splitlines()[0] == "record_id,group,tier"


def test_unknown_score_or_drug_group_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Planted score 'mood' is not one of the scores"):
        write_lab_reports(tmp_path, scores=["libido"], planted_scores=[PlantedScore("mood", 1.0)])
    with pytest.raises(ValueError, match="Drugs for 'PFS', which is not one of the groups"):
        write_lab_reports(tmp_path, drugs={"PFS": ["finasteride"]})
    with pytest.raises(ValueError, match="score names"):
        write_lab_reports(tmp_path, scores=["sexual function"])


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
    out = write_self_reports(
        tmp_path, authors=200, repeat_share=0.4, deleted_posts=12, unverified_posts=7, seed=1
    )
    data = load_self_reports(out.export)
    assert len(data.authors) == 200
    assert data.excluded == {"no_author": 12, "unverified": 7}
    assert len(load_self_reports(out.export, include_unverified=True).authors) == 207
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


SEXUAL = ("genital_numbness", "erectile_dysfunction", "anorgasmia")
EMOTIONAL = ("anhedonia", "emotional_blunting", "brain_fog")


def test_planted_symptom_subgroups_report_their_own_symptoms(tmp_path):
    planted = [
        PlantedSubgroup(share=0.5, symptoms=SEXUAL, onset_during=0.6),
        PlantedSubgroup(share=0.5, symptoms=EMOTIONAL),
    ]
    out = write_self_reports(tmp_path, authors=2000, subgroups=planted, seed=5)
    data = load_self_reports(out.export)
    reported = pd.crosstab(data.symptoms["author"], data.symptoms["symptom"]).gt(0)
    reported = reported.reindex(data.authors["author"], fill_value=False)
    sexual = reported[list(SEXUAL)].sum(axis=1) >= 2
    emotional = reported[list(EMOTIONAL)].sum(axis=1) >= 2
    assert sexual.mean() == pytest.approx(0.5, abs=0.06)  # about 0.9 of half, plus a few others
    assert (sexual & emotional).mean() < 0.05  # the subgroups' symptoms rarely go together
    # within a subgroup each listed symptom is common, the other subgroup's rare
    assert reported.loc[sexual, "anorgasmia"].mean() > 0.7
    assert reported.loc[sexual, "anhedonia"].mean() < 0.2
    onset = data.authors.set_index("author")["onset"]
    assert (onset[sexual.to_numpy()] == "during_use").mean() == pytest.approx(0.6, abs=0.08)
    assert (onset[emotional.to_numpy()] == "during_use").mean() < 0.05
    truth = json.loads(out.truth.read_text())
    assert truth["null"] is False and len(truth["subgroups"]) == 2


def test_symptom_subgroups_are_checked(tmp_path):
    with pytest.raises(ValueError, match="Unknown symptom"):
        write_self_reports(tmp_path, subgroups=[PlantedSubgroup(1.0, ("hiccups",))])
    with pytest.raises(ValueError, match="add to at most 1"):
        write_self_reports(
            tmp_path, subgroups=[PlantedSubgroup(0.7, SEXUAL), PlantedSubgroup(0.7, EMOTIONAL)]
        )
    with pytest.raises(ValueError, match="Null mode"):
        write_self_reports(tmp_path, null=True, subgroups=[PlantedSubgroup(0.5, SEXUAL)])
