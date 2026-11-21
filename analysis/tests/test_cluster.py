import dataclasses

import numpy as np
import pytest

from asclepius import synthetic
from asclepius.cluster import NO_STABLE, ClusterSettings, TooFewPatients, cluster_patients
from asclepius.dataset import load_lab_reports


def lab_dataset(tmp_path, **settings):
    out = synthetic.write_lab_reports(tmp_path, **settings)
    return load_lab_reports([out.reports], metadata=out.metadata)


def test_refuses_fewer_than_30_patients(tmp_path):
    data = lab_dataset(tmp_path, groups={"PSSD": 29, "healthy control": 40}, null=True)
    with pytest.raises(TooFewPatients, match="29 patients.*at least 30"):
        cluster_patients(data, groups=["PSSD"])


# A subgroup with lower 5-alpha reduced metabolites: four markers at 40% of everyone else's.
FIVE_ALPHA = ("androsterone", "5a_androstanediol", "a_tetrahydrocortisol_a_thf", "a_pregnanediol")


def two_subgroups(tmp_path, size=50, fold=0.4, seed=0, **settings):
    """Two groups of `size`, PFS with `FIVE_ALPHA` at `fold`. Clustering never sees the group, so
    PSSD and PFS stand in for two hidden subgroups of one condition."""
    planted = [synthetic.PlantedDifference(m, fold, group="PFS") for m in FIVE_ALPHA]
    return lab_dataset(tmp_path, groups={"PSSD": size, "PFS": size}, planted=planted, seed=seed, **settings)


def agreement(result, data):
    """Share of patients whose cluster is the one most of their true group fell in."""
    truth = data.records.set_index("record")["group"]
    table = result.assignments.join(truth, on="record")
    majority = table.groupby("group")["cluster"].agg(lambda c: c.mode()[0])
    return (table["cluster"] == table["group"].map(majority)).mean(), majority.nunique()


@pytest.mark.parametrize("seed", range(5))
def test_planted_subgroups_are_recovered(tmp_path, seed):
    # 50 each: found in 20 of 20 seeds, 93% to 99% of patients in their own subgroup's cluster.
    # At 30 each it is 16 of 20, at 90% to 98%; the other 4 say "no stable clusters".
    data = two_subgroups(tmp_path, seed=seed)
    result = cluster_patients(data, groups=["PSSD", "PFS"])
    assert result.k == 2
    assert result.statement == "2 stable clusters"
    share, distinct = agreement(result, data)
    assert distinct == 2 and share >= 0.9
    assert result.sizes["n_patients"].sum() == 100


def test_the_profile_shows_what_sets_a_cluster_apart(tmp_path):
    data = two_subgroups(tmp_path)
    result = cluster_patients(data, groups=["PSSD", "PFS"])
    truth = data.records.set_index("record")["group"]
    pfs = result.assignments.join(truth, on="record").query("group == 'PFS'")["cluster"].mode()[0]
    profile = result.profile[result.profile["cluster"] == pfs].set_index("marker_id")["median_score"]
    assert (profile[list(FIVE_ALPHA)] < -0.5).all()
    assert profile.drop(list(FIVE_ALPHA)).filter(regex="^[^/]*$").abs().max() < 0.5  # other raw markers
    assert profile["androsterone/etiocholanolone"] < -0.5  # and the ratios they are part of


@pytest.mark.parametrize("seed", range(10))
def test_no_subgroups_gives_no_stable_clusters(tmp_path, seed):
    # 0 of 20 seeds give clusters, at 60 or 100 patients
    data = lab_dataset(tmp_path, groups={"PSSD": 60}, null=True, seed=seed)
    result = cluster_patients(data)
    assert result.statement == NO_STABLE == "no stable clusters"
    assert result.k is None and result.assignments.empty and result.sizes.empty
    assert result.reason


@pytest.mark.parametrize("seed", range(5))
def test_one_stretched_group_is_not_subgroups(tmp_path, seed):
    # Every marker of a patient times one shared factor (as when one person makes more of
    # everything): k-means splits high from low the same way every time, so the split is stable,
    # but no tighter than in one normal group. About 1 seed in 20 still gives clusters, as a 5%
    # test allows (seed 6 here).
    data = lab_dataset(tmp_path, groups={"PSSD": 60}, null=True, seed=seed)
    factor = np.exp(np.random.default_rng(seed).normal(0, 0.6, len(data.records)))
    factor = data.markers["record"].map(dict(zip(data.records["record"], factor, strict=True)))
    data = dataclasses.replace(data, markers=data.markers.assign(value=data.markers["value"] * factor))
    result = cluster_patients(data)
    assert result.statement == NO_STABLE
    assert "stable but no tighter than in one group" in result.reason


def test_the_same_settings_give_the_same_result(tmp_path):
    data = two_subgroups(tmp_path)
    settings = ClusterSettings(seed=3)
    first, second = (cluster_patients(data, settings=settings) for _ in range(2))
    assert first.assignments.equals(second.assignments)
    assert first.stability.equals(second.stability) and first.null_p == second.null_p
    assert first.settings["status"] == "provisional, needs expert review"


def test_reports_with_no_sex_are_left_out(tmp_path):
    data = lab_dataset(tmp_path, groups={"PSSD": 35}, null=True)
    records = data.records.assign(sex=[None] * 6 + list(data.records["sex"][6:]))
    data = dataclasses.replace(data, records=records)
    with pytest.raises(TooFewPatients, match="29 patients \\(6 more left out for having no sex\\)"):
        cluster_patients(data)


def test_30_per_subgroup_is_found_in_most_seeds(tmp_path):
    found = []
    for seed in range(20):
        data = two_subgroups(tmp_path / str(seed), size=30, seed=seed)
        result = cluster_patients(data, groups=["PSSD", "PFS"])
        if result.k is not None:
            assert result.k == 2 and agreement(result, data)[0] >= 0.9
            found.append(seed)
    assert len(found) == 16  # the others say "no stable clusters"


def test_tiers_limit_who_is_clustered(tmp_path):
    data = two_subgroups(tmp_path, size=100, seed=1, tier_a_share=0.5)
    result = cluster_patients(data, groups=["PSSD", "PFS"], tiers=["A"])
    tier_a = data.records.loc[data.records["tier"] == "A", "record"]
    assert result.k == 2 and set(result.assignments["record"]) == set(tier_a)
    with pytest.raises(TooFewPatients):
        cluster_patients(two_subgroups(tmp_path / "small", size=40), tiers=["A"])


def test_too_few_usable_markers_says_no_stable_clusters(tmp_path):
    data = lab_dataset(tmp_path, groups={"PSSD": 40}, null=True)
    kept = data.markers[data.markers["marker_id"] == "testosterone"]
    result = cluster_patients(dataclasses.replace(data, markers=kept))
    assert result.statement == NO_STABLE
    assert "two markers" in result.reason and result.features == ("testosterone",)


def test_counts_below_5_are_not_shown(tmp_path):
    # 10% women: each cluster has fewer than 5 of them (spec S11)
    planted = [synthetic.PlantedDifference(m, 0.4, group="PFS") for m in FIVE_ALPHA]
    data = lab_dataset(tmp_path, groups={"PSSD": 40, "PFS": 40}, planted=planted, female_share=0.1, seed=0)
    result = cluster_patients(data)
    women = result.sizes[result.sizes["sex"] == "female"]["n_patients"]
    assert women.isna().all() and result.sizes["n_patients"].dropna().ge(5).all()


def test_small_counts_of_missing_values_are_not_shown(tmp_path):
    data = lab_dataset(tmp_path, groups={"PSSD": 40}, null=True)

    def without(marker, reports):
        gone = (data.markers["marker_id"] == marker) & data.markers["record"].isin(
            data.records["record"][:reports]
        )
        return cluster_patients(dataclasses.replace(data, markers=data.markers[~gone]))

    assert without("dhea_s", 3).filled is None  # 3 values filled: fewer than 5 (spec S11)
    assert without("dhea_s", 6).filled == 6
    dropped = without("dhea_s", 20).dropped  # missing for more than 20%: left out
    assert dropped.to_dict("records") == [{"marker_id": "dhea_s", "n_missing": 20}]


def test_the_result_carries_a_run_record(tmp_path):
    data = two_subgroups(tmp_path)
    record = cluster_patients(data, groups=["PSSD", "PFS"]).record
    assert record["dataset"]["fingerprint"] == data.fingerprint
    assert record["tool"] and record["groups"] == ["PFS", "PSSD"]
    assert record["settings"]["min_patients"] == 30 and record["settings"]["tiers"] is None
