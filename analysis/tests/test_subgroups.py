import dataclasses
import json

import numpy as np
import pytest

from asclepius import synthetic
from asclepius.cluster import cluster_patients
from asclepius.dataset import load_lab_reports
from asclepius.record import export_run, reproduce
from asclepius.run import Settings, benjamini_hochberg, run_analysis
from asclepius.subgroups import SUBGROUP_DRUG, SUBGROUP_KINDS, SUBGROUP_SCORE, link_subgroups

# PSSD and PFS stand in for two hidden subgroups of one condition, as in test_cluster: clustering
# never sees the group. PFS has four 5-alpha reduced markers at 40%, which makes the clusters.
FIVE_ALPHA = ("androsterone", "5a_androstanediol", "a_tetrahydrocortisol_a_thf", "a_pregnanediol")
FAST = Settings(permutations=999, bootstrap=200)


def subgroups(tmp_path, size=50, seed=0, **settings):
    planted = [synthetic.PlantedDifference(m, 0.4, group="PFS") for m in FIVE_ALPHA]
    out = synthetic.write_lab_reports(
        tmp_path, groups={"PSSD": size, "PFS": size}, planted=planted, seed=seed, **settings
    )
    data = load_lab_reports([out.reports], metadata=out.metadata)
    return data, cluster_patients(data, groups=["PSSD", "PFS"])


def row(links, marker_id):
    rows = links.rows[links.rows["marker_id"] == marker_id]
    assert len(rows) == 1
    return rows.iloc[0]


@pytest.mark.parametrize("seed", range(3))
def test_a_planted_symptom_difference_is_found_in_its_cluster(tmp_path, seed):
    data, clusters = subgroups(
        tmp_path,
        seed=seed,
        scores=["libido", "fatigue"],
        planted_scores=[synthetic.PlantedScore("libido", 3.0, group="PFS")],
    )
    assert clusters.k == 2
    links = link_subgroups(data, clusters, FAST)
    libido, fatigue = row(links, "score_libido"), row(links, "score_fatigue")
    assert libido["comparison"] == SUBGROUP_SCORE == "subgroup: symptom score"
    assert (libido["group_a"], libido["group_b"]) == ("cluster 1", "cluster 2")
    assert (
        libido["p_value"] <= 0.001
    )  # the smallest 999 permutations give and abs(libido["effect_size"]) > 0.5
    assert abs(fatigue["effect_size"]) < 0.33  # its p-value is 0.009 in seed 2: chance, uncorrected
    # the cluster most PFS patients are in has the higher libido score
    truth = data.records.set_index("record")["group"]
    pfs = clusters.assignments.join(truth, on="record").query("group == 'PFS'")["cluster"].mode()[0]
    higher = libido["median_a"] > libido["median_b"]
    assert higher == (pfs == 1)
    assert libido["n_a"] + libido["n_b"] == 100 and libido["sex"] == "all"
    assert libido["subgroups_of"] == "PFS, PSSD"


def test_a_drug_taken_by_one_cluster_is_found(tmp_path):
    # a quarter of PSSD and three quarters of PFS took finasteride, so each cluster has at least 5
    # who did and 5 who did not, and both shares can be shown (spec S11)
    drugs = {"PSSD": ["sertraline"] * 3 + ["finasteride"], "PFS": ["finasteride"] * 3 + ["sertraline"]}
    data, clusters = subgroups(tmp_path, drugs=drugs)
    links = link_subgroups(data, clusters, FAST)
    finasteride = row(links, "drug:finasteride")
    assert finasteride["comparison"] == SUBGROUP_DRUG == "subgroup: drug taken"
    assert finasteride["marker_name"] == "finasteride"
    assert finasteride["p_value"] <= 0.001  # the smallest 999 permutations give
    # on yes/no values, Cliff's delta is the difference in shares
    assert finasteride["effect_size"] == pytest.approx(finasteride["share_a"] - finasteride["share_b"])
    assert abs(finasteride["effect_size"]) > 0.3


def test_shares_below_5_are_hidden_and_rare_drugs_refused(tmp_path):
    # PFS all took finasteride: its cluster has fewer than 5 who did not. About 6 of PSSD's 50
    # are on fluoxetine, fewer than the 10 a test needs (spec S4).
    # 14 or so PSSD took it too, so its cluster's share alone could be shown
    drugs = {"PSSD": ["sertraline"] * 15 + ["fluoxetine"] * 3 + ["finasteride"] * 7, "PFS": ["finasteride"]}
    data, clusters = subgroups(tmp_path, drugs=drugs)
    links = link_subgroups(data, clusters, FAST)
    finasteride = row(links, "drug:finasteride")
    # both hidden: with one shown, the other is effect_size + share_b, and times n_a a count
    assert np.isnan(finasteride["share_a"]) and np.isnan(finasteride["share_b"])
    assert abs(finasteride["effect_size"]) > 0.5
    refused = links.refused.set_index("marker_id")
    assert "drug:fluoxetine" not in set(links.rows["marker_id"])
    assert refused.loc["drug:fluoxetine", "reason"] == "fewer than 10 patients took it (spec S4)"
    assert links.refused["n_a"].dropna().ge(5).all() and links.rows["tier_a_n_a"].dropna().ge(5).all()


def test_no_stable_clusters_gives_nothing_to_link(tmp_path):
    out = synthetic.write_lab_reports(tmp_path, groups={"PSSD": 60}, null=True, scores=["libido"])
    data = load_lab_reports([out.reports], metadata=out.metadata)
    links = link_subgroups(data, cluster_patients(data, groups=["PSSD"]), FAST)
    assert links.rows.empty and links.statement == "no stable clusters, so nothing to link"


def test_clusters_from_another_dataset_are_refused(tmp_path):
    data, clusters = subgroups(tmp_path / "a")
    other, _ = subgroups(tmp_path / "b", seed=1)
    with pytest.raises(ValueError, match="different dataset"):
        link_subgroups(other, clusters, FAST)


def test_the_tier_a_rerun_keeps_the_clusters(tmp_path):
    data, clusters = subgroups(
        tmp_path,
        size=100,
        seed=1,
        tier_a_share=0.5,
        scores=["libido"],
        planted_scores=[synthetic.PlantedScore("libido", 3.0, group="PFS")],
    )
    libido = row(link_subgroups(data, clusters, FAST), "score_libido")
    tier_a = data.records.loc[data.records["tier"] == "A", "record"]
    assert libido["tier_a_n_a"] + libido["tier_a_n_b"] == len(tier_a)
    assert libido["tier_a_result"] == "holds" and libido["holds_in_tier_a"]
    assert np.sign(libido["tier_a_effect_size"]) == np.sign(libido["effect_size"])


def test_with_more_than_two_clusters_each_is_compared_with_the_rest(tmp_path):
    data, clusters = subgroups(tmp_path, scores=["libido"])
    three = dataclasses.replace(
        clusters, k=3, assignments=clusters.assignments.assign(cluster=[1, 2, 3, 1] * 25)
    )
    rows = link_subgroups(data, three, FAST).rows
    assert list(rows["group_a"]) == ["cluster 1", "cluster 2", "cluster 3"]
    assert set(rows["group_b"]) == {"other clusters"} and (rows["n_a"] + rows["n_b"] == 100).all()


def test_subgroup_findings_join_the_run_and_carry_labels(tmp_path):
    data, clusters = subgroups(
        tmp_path,
        size=100,
        seed=1,
        tier_a_share=0.5,
        scores=["libido", "fatigue"],
        planted_scores=[synthetic.PlantedScore("libido", 3.0, group="PFS")],
    )
    links = link_subgroups(data, clusters, FAST)
    result = run_analysis(data, "PSSD", "PFS", FAST, subgroups=links)
    plain = run_analysis(data, "PSSD", "PFS", FAST)
    findings = result.findings
    linked = findings[findings["comparison"].isin(SUBGROUP_KINDS)]
    assert len(linked) == 2 and len(findings) == len(plain.findings) + 2
    libido = linked.set_index("marker_id").loc["score_libido"]
    assert libido["label"] == "strong"
    assert linked.set_index("marker_id").loc["score_fatigue", "label"] == "noise-level"
    # one correction over all 70 tests, not the 68 markers and ratios and the 2 links apart
    assert np.allclose(findings["q_value"], benjamini_hochberg(findings["p_value"].to_numpy()))
    assert not np.allclose(findings["q_value"][:68], plain.findings["q_value"])
    assert result.record["subgroups"]["statement"] == "2 stable clusters"


def test_links_made_with_other_settings_are_refused_by_the_run(tmp_path):
    data, clusters = subgroups(tmp_path, scores=["libido"])
    links = link_subgroups(data, clusters, FAST)
    with pytest.raises(ValueError, match="same settings"):
        run_analysis(data, "PSSD", "PFS", Settings(permutations=999, bootstrap=100), subgroups=links)


def test_a_run_with_subgroups_reproduces_from_its_record(tmp_path):
    planted = [synthetic.PlantedDifference(m, 0.4, group="PFS") for m in FIVE_ALPHA]
    out = synthetic.write_lab_reports(
        tmp_path / "data", groups={"PSSD": 50, "PFS": 50}, planted=planted, scores=["libido"]
    )
    data = load_lab_reports([out.reports], metadata=out.metadata)
    links = link_subgroups(data, cluster_patients(data, groups=["PSSD", "PFS"]), FAST)
    result = run_analysis(data, "PSSD", "PFS", FAST, subgroups=links)
    record = json.loads((export_run(result, tmp_path / "export") / "run_record.json").read_text())
    again = reproduce(record, [out.reports], metadata=out.metadata)
    assert again.findings.equals(result.findings) and again.refused.equals(result.refused)


def test_a_run_says_when_there_were_no_subgroups_to_link(tmp_path):
    out = synthetic.write_lab_reports(tmp_path, groups={"PSSD": 60, "healthy control": 40}, null=True)
    data = load_lab_reports([out.reports], metadata=out.metadata)
    links = link_subgroups(data, cluster_patients(data, groups=["PSSD"]), FAST)
    result = run_analysis(data, "PSSD", "healthy control", FAST, subgroups=links)
    assert result.statement.endswith("; subgroups: no stable clusters, so nothing to link")
