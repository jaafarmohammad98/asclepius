import json
from pathlib import Path

import numpy as np
import pytest

from asclepius import synthetic
from asclepius.record import export_run
from asclepius.run import STRONG, SUGGESTIVE
from asclepius.selfreport import load_self_reports
from asclepius.symptom_subgroups import (
    SymptomClusterSettings,
    TooFewAuthors,
    cluster_symptoms,
    swap_randomise,
)

SEXUAL = ("genital_numbness", "erectile_dysfunction", "anorgasmia")
EMOTIONAL = ("anhedonia", "emotional_blunting", "brain_fog")


def self_reports(tmp_path, *, seed=1, authors=400, null=False, subgroups=()):
    out = synthetic.write_self_reports(
        tmp_path, authors=authors, null=null, subgroups=list(subgroups), seed=seed
    )
    return load_self_reports(out.export)


def planted(onset_during=(0.0, 0.0)):
    return [
        synthetic.PlantedSubgroup(share=0.5, symptoms=SEXUAL, onset_during=onset_during[0]),
        synthetic.PlantedSubgroup(share=0.5, symptoms=EMOTIONAL, onset_during=onset_during[1]),
    ]


def top_symptoms(result, cluster, n=3):
    rows = result.summary[result.summary["cluster"] == cluster]
    return set(rows.sort_values("difference", ascending=False)["symptom"].head(n))


@pytest.mark.parametrize("seed", range(3))
def test_two_planted_symptom_subgroups_are_recovered(tmp_path, seed):
    result = cluster_symptoms(self_reports(tmp_path, seed=seed, subgroups=planted()))
    assert result.k == 2 and result.clusters == "2 stable clusters"
    described = [top_symptoms(result, c) for c in (1, 2)]
    assert sorted(described, key=sorted) == sorted([set(SEXUAL), set(EMOTIONAL)], key=sorted)
    assert (result.sizes["n_authors"] >= 30).all()


@pytest.mark.parametrize("seed", range(5))
def test_self_reports_with_no_subgroups_give_no_stable_clusters(tmp_path, seed):
    result = cluster_symptoms(self_reports(tmp_path, seed=seed, null=True))
    assert result.k is None and result.clusters == "no stable clusters"
    assert result.statement == "no stable clusters"
    assert result.reason
    assert result.findings.empty and result.summary.empty


@pytest.mark.parametrize("seed", range(3))
def test_a_cluster_with_more_onsets_during_use_is_found_and_never_strong(tmp_path, seed):
    result = cluster_symptoms(self_reports(tmp_path, seed=seed, subgroups=planted(onset_during=(0.5, 0.15))))
    assert result.k == 2
    sexual = 1 if top_symptoms(result, 1) == set(SEXUAL) else 2
    rows = result.findings[result.findings["marker_id"] == "onset:during_use"]
    assert len(rows) == 1  # with 2 clusters, cluster 1 against cluster 2
    row = rows.iloc[0]
    assert row["comparison"] == "symptom subgroup: onset"
    assert (row["group_a"], row["group_b"]) == ("cluster 1", "cluster 2")
    assert row["label"] == SUGGESTIVE and row["caveat"] == "self-report, no controls"
    assert np.sign(row["effect_size"]) == (1 if sexual == 1 else -1)
    assert STRONG not in set(result.findings["label"])
    assert result.statement.startswith("2 stable clusters; ")
    # drug classes and trajectories are tested too, in the same correction
    assert {"symptom subgroup: drug class", "symptom subgroup: trajectory"} <= set(
        result.findings["comparison"]
    )
    assert result.findings["q_value"].notna().any()


def test_clusters_that_differ_by_nothing_planted_give_nothing_reliable(tmp_path):
    result = cluster_symptoms(self_reports(tmp_path, seed=2, subgroups=planted()))
    assert result.k == 2
    assert result.statement == "2 stable clusters; nothing reliable yet"


def test_authors_with_fewer_than_two_symptoms_are_left_out_and_counted(tmp_path):
    data = self_reports(tmp_path, null=True)
    result = cluster_symptoms(data)
    reported = data.symptoms[data.symptoms["symptom"].isin(result.symptoms)]
    per_author = reported.groupby("author")["symptom"].nunique()
    fewer = len(data.authors) - int((per_author >= 2).sum())
    assert fewer >= 5 and result.left_out == fewer
    assert result.record["analysis"] == "symptom subgroups"


def test_clustering_fewer_than_30_authors_is_refused(tmp_path):
    with pytest.raises(TooFewAuthors, match="at least 30"):
        cluster_symptoms(self_reports(tmp_path, authors=25, null=True))


def test_swap_randomisation_keeps_every_total_and_breaks_the_links():
    rng = np.random.default_rng(0)
    x = np.zeros((200, 6), dtype=bool)
    x[:100, :3] = True  # half report symptoms 0-2, half 3-5: never together
    x[100:, 3:] = True
    swapped = swap_randomise(x, rounds=20, rng=rng)
    assert (swapped.sum(axis=1) == x.sum(axis=1)).all()
    assert (swapped.sum(axis=0) == x.sum(axis=0)).all()
    assert (swapped[:, 0] & swapped[:, 3]).sum() > 30  # now often together


def test_three_planted_subgroups_compare_each_cluster_with_the_others(tmp_path):
    groups = [
        synthetic.PlantedSubgroup(1 / 3, ("genital_numbness", "erectile_dysfunction", "anorgasmia")),
        synthetic.PlantedSubgroup(1 / 3, ("anhedonia", "emotional_blunting", "depression")),
        synthetic.PlantedSubgroup(1 / 3, ("brain_fog", "fatigue", "insomnia")),
    ]
    result = cluster_symptoms(self_reports(tmp_path, authors=600, subgroups=groups))
    assert result.k == 3
    assert set(result.findings["group_b"]) == {"other clusters"}
    assert set(result.findings["group_a"]) == {"cluster 1", "cluster 2", "cluster 3"}


def test_no_output_names_an_author_or_shows_a_count_below_5(tmp_path):
    result = cluster_symptoms(self_reports(tmp_path, authors=150, subgroups=planted((0.5, 0.0))))
    assert result.k == 2
    tables = (result.findings, result.refused, result.summary, result.sizes)
    for table in tables:
        assert not {"author", "author_key", "case_id"} & set(table.columns)
        for column in [c for c in table.columns if c.startswith(("n_", "reported_"))]:
            counts = table[column].dropna()
            assert not ((counts > 0) & (counts < 5)).any(), column
    # isotretinoin has about 10 authors here: refused or hidden, never a small count
    small = result.findings[result.findings["reported_a"].isna()]
    assert small[["share_a", "share_b", "p_value", "odds_ratio"]].isna().all().all()


def test_settings_are_recorded_the_run_is_repeatable_and_it_exports(tmp_path):
    data = self_reports(tmp_path / "sr", subgroups=planted())
    a, b = cluster_symptoms(data), cluster_symptoms(data)
    assert a.findings.equals(b.findings) and a.summary.equals(b.summary)
    assert a.record["settings"]["clusters"] == SymptomClusterSettings().record()
    assert a.record["settings"]["min_group_size"] == 10
    out = export_run(a, tmp_path / "out")
    record = json.loads((out / "run_record.json").read_text())
    assert set(record["outputs"]) == {"findings.csv", "refused.csv", "skipped.csv", "summary.csv"}


REAL_EXPORT = Path(__file__).parents[2] / "harvester" / "data" / "export"


@pytest.mark.skipif(not (REAL_EXPORT / "cases.csv").exists(), reason="the r/PSSD export is not here")
def test_the_real_export_runs_and_every_cluster_has_at_least_30_authors():
    result = cluster_symptoms(load_self_reports(REAL_EXPORT))
    assert result.left_out is not None
    if result.k is not None:
        assert (result.sizes["n_authors"] >= 30).all()
    assert STRONG not in set(result.findings["label"])
    for table in (result.findings, result.refused, result.summary, result.sizes):
        assert not {"author", "author_key", "case_id"} & set(table.columns)
