"""The findings view (M3-10b): what the page shows of a finished run."""

import json

import numpy as np
import pandas as pd
import pytest
from test_subgroups import FIVE_ALPHA
from test_web import FAST, add_export, add_lab, finished, lab_files

from asclepius import synthetic
from asclepius.cluster import cluster_patients
from asclepius.dataset import load_lab_reports
from asclepius.run import run_analysis
from asclepius.subgroups import link_subgroups
from asclepius.web.findings import cluster_sizes, distribution, finding_view, findings_view
from asclepius.web.workspace import DONE, Step, Subgroups, Workspace


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory):
    """A finished lab run with one planted difference: androsterone 60% lower in PSSD."""
    tmp_path = tmp_path_factory.mktemp("planted")
    out = lab_files(tmp_path, planted=[synthetic.PlantedDifference("androsterone", 0.4)])
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, out)
    workspace.load()
    workspace.start_run("PSSD", "healthy control", settings=FAST)
    finished(workspace)
    return workspace.findings()


def section(view, kind):
    return next(s for s in view["sections"] if s["kind"] == kind)


def test_a_planted_difference_is_a_finding_with_its_sizes_label_tier_a_result_and_plot(planted_run):
    comparison = section(planted_run, "comparison")
    assert comparison["state"] == "done"
    assert not comparison["nothing_reliable"]
    rows = [f for f in comparison["findings"] if f["marker_id"] == "androsterone" and f["sex"] == "male"]
    assert len(rows) == 1
    finding = rows[0]
    # 40 per group, 30% women: 28 men in each (synthetic.write_lab_reports)
    assert (finding["n_a"], finding["n_b"]) == (28, 28)
    assert finding["label"] in ("strong", "suggestive")
    assert finding["effect_size"] < -0.33
    assert finding["tier_a_result"] in ("holds", "does not hold", "Tier A too small to test")
    plot = finding["plot"]
    assert plot["type"] == "distribution"
    assert [side["group"] for side in plot["sides"]] == ["PSSD", "healthy control"]
    pssd, control = plot["sides"]
    assert pssd["n"] == 28
    assert pssd["q10"] <= pssd["q25"] <= pssd["median"] <= pssd["q75"] <= pssd["q90"]
    assert pssd["median"] == pytest.approx(finding["median_a"])
    assert pssd["median"] < control["median"]  # 60% lower


def run(tmp_path, *, compare=("PSSD", "healthy control"), **options):
    """The findings of a finished run comparing `compare`, on `lab_files(**options)`."""
    workspace = Workspace(tmp_path / "work")
    add_lab(workspace, lab_files(tmp_path, **options))
    workspace.load()
    workspace.start_run(*compare, settings=FAST)
    finished(workspace)
    return workspace.findings()


def every_finding(view):
    for s in view["sections"]:
        yield from s["findings"]


def counts_in(value):
    """Every count anywhere in the view: values of keys naming a count."""
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                isinstance(item, int)
                and not isinstance(item, bool)
                and (
                    key == "n"
                    or key.startswith(("n_", "tier_a_n", "reported_"))
                    or key == "authors"
                    or key in ("below", "within", "above", "unclear")
                )
            ):
                yield key, item
            else:
                yield from counts_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from counts_in(item)


def test_a_null_dataset_says_nothing_reliable_yet(tmp_path):
    view = run(tmp_path, null=True)
    assert view["nothing_reliable"] is True
    assert view["statement"] == "nothing reliable yet"
    comparison = section(view, "comparison")
    assert comparison["nothing_reliable"] is True
    assert comparison["statement"] == "nothing reliable yet"
    assert comparison["findings"]  # the tests are still listed, each labelled noise-level
    assert {f["label"] for f in comparison["findings"]} == {"noise-level"}


def test_a_planted_run_does_not_say_nothing_reliable(planted_run):
    assert planted_run["nothing_reliable"] is False
    assert planted_run["statement"] is None


def test_every_finding_shows_its_sizes_and_a_label(planted_run):
    findings = list(every_finding(planted_run))
    assert findings
    for finding in findings:
        assert finding["label"]
        assert "n_a" in finding
        assert "plot" in finding
        if finding.get("group_b") not in (None, "reference range"):
            assert "n_b" in finding
            assert finding["tier_a_result"]


def test_no_count_below_5_is_shown(planted_run, tmp_path):
    for view in (planted_run, run(tmp_path, tier_a_share=0.1)):
        for key, count in counts_in(view):
            assert count == 0 or count >= 5, key  # 0 is no one


def test_findings_are_ordered_strongest_first_then_by_effect_size(planted_run):
    rank = {"strong": 0, "suggestive": 1, "noise-level": 2}
    findings = section(planted_run, "comparison")["findings"]
    keys = [(rank[f["label"]], -abs(f["effect_size"])) for f in findings]
    assert keys == sorted(keys)


def test_a_ratio_finding_plots_the_ratio_values(planted_run):
    ratios = [
        f
        for f in section(planted_run, "comparison")["findings"]
        if f["comparison"] == "enzyme-activity ratio"
    ]
    assert ratios
    for finding in ratios:
        a, b = finding["plot"]["sides"]
        assert a["median"] == pytest.approx(finding["median_a"])
        assert b["median"] == pytest.approx(finding["median_b"])
        assert finding["status"] == "provisional, needs expert review"


def test_reference_range_positions_show_counts_per_zone_labelled_as_descriptions(planted_run):
    ranges = section(planted_run, "ranges")
    assert ranges["nothing_reliable"] is None  # descriptions, not tests
    assert ranges["findings"]
    for finding in ranges["findings"]:
        assert finding["label"] == "description, not a test"
        side = finding["plot"]["sides"][0]
        assert finding["plot"]["type"] == "zones"
        shown = [side[z] for z in ("below", "within", "above", "unclear") if side[z] is not None]
        hidden_total = (finding["n_a"] or 0) - sum(shown)
        assert finding["n_a"] is None or hidden_total == 0 or hidden_total >= 5


def test_subgroup_links_are_labelled_with_the_comparison_and_plot_each_cluster(tmp_path):
    """Stable clusters, as in test_subgroups: PSSD and PFS clustered together stand in for two
    hidden subgroups (the app clusters one group, where synthetic data has none to find)."""
    planted = [synthetic.PlantedDifference(m, 0.4, group="PFS") for m in FIVE_ALPHA]
    out = synthetic.write_lab_reports(
        tmp_path,
        groups={"PSSD": 50, "PFS": 50},
        planted=planted,
        scores=["libido"],
        planted_scores=[synthetic.PlantedScore("libido", 3.0, group="PFS")],
        drugs={"PSSD": ["sertraline", "fluoxetine"], "PFS": ["finasteride"]},
    )
    data = load_lab_reports([out.reports], metadata=out.metadata)
    clusters = cluster_patients(data, groups=["PSSD", "PFS"])
    links = link_subgroups(data, clusters, FAST)
    comparison = run_analysis(data, "PSSD", "PFS", FAST, subgroups=links)
    steps = [Step("Look for subgroups", "subgroups", None, DONE), Step("Compare", "comparison", None, DONE)]
    results = {"Look for subgroups": Subgroups(clusters, links), "Compare": comparison}
    view = findings_view(data, steps, results)

    # without the comparison, the links have no label (spec S5), so none is shown
    alone = section(findings_view(data, steps[:1], results), "subgroups")
    assert alone["findings"] == [] and "not shown" in alone["note"]
    assert alone["clusters"]["statement"] == "2 stable clusters"

    subgroups = section(view, "subgroups")
    assert subgroups["clusters"]["statement"] == "2 stable clusters"
    assert subgroups["clusters"]["sizes"]
    kinds = {f["comparison"] for f in subgroups["findings"]}
    assert kinds == {"subgroup: symptom score", "subgroup: drug taken"}
    for finding in subgroups["findings"]:
        assert finding["label"] in ("strong", "suggestive", "noise-level")
        assert finding["n_a"] and finding["n_b"]
    libido = next(f for f in subgroups["findings"] if f["marker_id"] == "score_libido")
    assert libido["label"] in ("strong", "suggestive")
    a, b = libido["plot"]["sides"]
    assert libido["plot"]["type"] == "distribution"
    assert (a["group"], b["group"]) == ("cluster 1", "cluster 2")
    assert (a["n"], b["n"]) == (libido["n_a"], libido["n_b"])
    assert a["median"] == pytest.approx(libido["median_a"])
    drug = next(f for f in subgroups["findings"] if f["comparison"] == "subgroup: drug taken")
    assert drug["plot"]["type"] == "shares"
    # the comparison section does not repeat them
    assert not any(f["comparison"].startswith("subgroup:") for f in section(view, "comparison")["findings"])


def test_the_cross_syndrome_view_reads_each_marker_across_syndromes(tmp_path):
    view = run(tmp_path, groups={"PSSD": 40, "PFS": 40, "healthy control": 40}, seed=2)
    syndromes = section(view, "syndromes")
    assert syndromes["state"] == "done"
    assert syndromes["reference"] == "healthy control"
    assert syndromes["syndromes"] == ["PSSD", "PFS"]
    assert syndromes["summary"]
    assert {"pattern", "n_PSSD", "direction_PFS"} <= set(syndromes["summary"][0])
    kinds = {f["comparison"] for f in syndromes["findings"]}
    assert kinds == {"cross-syndrome: vs reference", "cross-syndrome: between syndromes"}


def test_a_refused_step_says_why_in_its_section(tmp_path):
    view = run(tmp_path, groups={"PSSD": 40, "healthy control": 40})
    syndromes = section(view, "syndromes")
    assert syndromes["state"] == "refused"
    assert "at least two" in syndromes["message"]
    assert syndromes["findings"] == []


def test_every_self_report_section_carries_the_self_report_label(tmp_path):
    workspace = Workspace(tmp_path / "work")
    add_export(workspace, synthetic.write_self_reports(tmp_path / "sr", authors=600, seed=1).export)
    workspace.load()
    workspace.start_run(settings=FAST)
    finished(workspace)
    view = workspace.findings()

    assert view["label"] == "self-report, no controls"
    kinds = [s["kind"] for s in view["sections"]]
    assert kinds == [
        "self-report ranges",
        "drug and symptom profile",
        "symptom subgroups",
        "intervention outcomes",
    ]
    for s in view["sections"]:
        assert "self-report, no controls" in s["labels"]
    ranges = section(view, "self-report ranges")
    assert ranges["findings"]
    for finding in ranges["findings"]:
        assert finding["caveat"] == "self-report, no controls"
        assert finding["plot"]["type"] == "zones"  # counts only
    for key, count in counts_in(view):
        assert count == 0 or count >= 5, key
    profile = section(view, "drug and symptom profile")
    assert profile["state"] == "done" and profile["findings"] and profile["groups"]
    assert "counts in each" in profile["note"]
    for finding in profile["findings"]:
        assert finding["caveat"] == "self-report, no controls" and finding["label"] != "strong"
        assert finding["plot"]["type"] == "shares"
        assert finding["plot"]["caption"]


def syndrome_row(n, above, below):
    """A syndrome against the reference ranges, as `compare_syndromes` gives it."""
    return pd.Series(
        {
            "comparison": "cross-syndrome: vs reference",
            "marker_id": "x",
            "marker_name": "X",
            "sex": "male",
            "group_a": "PSSD",
            "group_b": "reference range",
            "n_a": n,
            "share_above": above / n,
            "share_below": below / n,
            "effect_size": (above - below) / n,
            "ci_low": -0.1,
            "ci_high": 0.2,
            "p_value": 0.5,
            "label": "noise-level",
        }
    )


def test_a_syndrome_shift_hides_a_within_count_below_5_and_all_that_gives_it_back():
    view = finding_view(syndrome_row(14, above=5, below=6), values=None)
    assert view["share_above"] is None and view["share_below"] is None
    assert view["effect_size"] is None  # with n, above - below and the shares, within comes back
    assert all(view["plot"]["sides"][0][z] is None for z in ("below", "within", "above"))
    assert view["n_a"] == 14 and view["label"] == "noise-level"


def test_a_syndrome_shift_with_every_zone_large_enough_is_shown():
    view = finding_view(syndrome_row(30, above=5, below=6), values=None)
    side = view["plot"]["sides"][0]
    assert (side["below"], side["within"], side["above"]) == (6, 19, 5)
    assert view["effect_size"] == pytest.approx(-1 / 30)


def test_a_drug_link_with_hidden_shares_hides_its_effect_too():
    row = pd.Series(
        {
            "comparison": "subgroup: drug taken",
            "marker_id": "drug:x",
            "marker_name": "x",
            "sex": "all",
            "group_a": "cluster 1",
            "group_b": "cluster 2",
            "n_a": 30,
            "n_b": 31,
            "share_a": np.nan,
            "share_b": np.nan,
            "effect_size": 0.4,
            "ci_low": 0.2,
            "ci_high": 0.6,
            "tier_a_effect_size": 0.5,
            "tier_a_ci_low": 0.1,
            "tier_a_ci_high": 0.8,
            "label": "suggestive",
        }
    )
    view = finding_view(row, values=None)
    for column in (
        "effect_size",
        "ci_low",
        "ci_high",
        "tier_a_effect_size",
        "tier_a_ci_low",
        "tier_a_ci_high",
    ):
        assert view[column] is None, column
    assert view["label"] == "suggestive" and (view["n_a"], view["n_b"]) == (30, 31)


def test_a_percentile_plot_needs_10_values():
    assert distribution("PSSD", pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9]))["median"] is None
    assert distribution("PSSD", pd.Series([float(v) for v in range(10)]))["median"] == 4.5


def test_cluster_sizes_hide_a_sex_whose_count_the_cluster_total_gives_back():
    sizes = pd.DataFrame(
        {
            "cluster": [1, 1, 2, 2],
            "sex": ["female", "male", "female", "male"],
            "n_patients": [None, 38, 14, 32],
        }
    )
    shown_sizes = {(r["cluster"], r["sex"]): r["n"] for r in cluster_sizes(sizes)}
    assert shown_sizes[(1, "male")] is None  # the cluster's total is shown with its links
    assert shown_sizes[(2, "female")] == 14


def test_the_symptom_subgroups_section_shows_clusters_their_symptoms_and_links(tmp_path):
    groups = [
        synthetic.PlantedSubgroup(
            0.5, ("genital_numbness", "erectile_dysfunction", "anorgasmia"), onset_during=0.5
        ),
        synthetic.PlantedSubgroup(0.5, ("anhedonia", "emotional_blunting", "brain_fog"), onset_during=0.15),
    ]
    workspace = Workspace(tmp_path / "work")
    add_export(
        workspace, synthetic.write_self_reports(tmp_path / "sr", authors=400, subgroups=groups, seed=1).export
    )
    workspace.load()
    workspace.start_run(settings=FAST)
    finished(workspace)
    view = workspace.findings()

    subgroups = section(view, "symptom subgroups")
    assert subgroups["state"] == "done"
    assert subgroups["clusters"]["statement"] == "2 stable clusters"
    assert [c["n"] for c in subgroups["clusters"]["sizes"]] and all(
        c["n"] >= 30 for c in subgroups["clusters"]["sizes"]
    )
    assert {r["symptom"] for r in subgroups["symptoms"] if r["cluster"] == 1}
    assert "left out" in subgroups["note"]
    onset = [f for f in subgroups["findings"] if f["marker_id"] == "onset:during_use"]
    assert onset and onset[0]["label"] == "suggestive" and onset[0]["plot"]["type"] == "shares"
    for finding in subgroups["findings"]:
        assert finding["caveat"] == "self-report, no controls" and finding["label"] != "strong"
    for key, count in counts_in(view):
        assert count == 0 or count >= 5, key


def test_the_intervention_section_ranks_reported_outcomes_and_lists_safety_signals(tmp_path):
    planted = [
        synthetic.PlantedOutcome("tadalafil", improved=0.6, worse=0.05),
        synthetic.PlantedOutcome("cyproheptadine", improved=0.1, worse=0.45),
    ]
    workspace = Workspace(tmp_path / "work")
    add_export(
        workspace,
        synthetic.write_self_reports(tmp_path / "sr", authors=1000, outcomes=planted, seed=1).export,
    )
    workspace.load()
    workspace.start_run(settings=FAST)
    finished(workspace)
    view = workspace.findings()

    s = section(view, "intervention outcomes")
    assert s["state"] == "done"
    assert "reported outcome, not evidence that it works" in s["labels"]
    assert s["ranking"][0]["intervention"] == "tadalafil" and s["ranking"][0]["rank"] == 1
    assert "cyproheptadine" in {r["intervention"] for r in s["safety"]}
    for r in s["ranking"] + s["safety"]:
        for key in ("n_clear", "major_improvement", "partial_improvement", "no_effect", "worse", "unclear"):
            assert r[key] is None or r[key] == 0 or r[key] >= 5, key
    text = json.dumps(s).lower()
    assert not any(word in text for word in ("recommend", "effective", "efficacy", "should try"))
    for key, count in counts_in(view):
        assert count == 0 or count >= 5, key
