import numpy as np
import pytest

from asclepius import synthetic
from asclepius.compare import compare_groups, compare_values
from asclepius.dataset import load_lab_reports


def lab_dataset(tmp_path, **settings):
    out = synthetic.write_lab_reports(tmp_path, **settings)
    return load_lab_reports([out.reports], metadata=out.metadata)


def test_effect_size_is_cliffs_delta_of_a_against_b():
    # every a below every b
    assert compare_values([1, 2, 3], [4, 5, 6]).effect_size == -1
    # pairs (1,2) lower, (3,2) higher: one each way
    assert compare_values([1, 3], [2]).effect_size == 0
    # 5 vs (1,2,5): two higher, one tie
    assert compare_values([5], [1, 2, 5]).effect_size == pytest.approx(2 / 3)


def test_small_groups_get_the_exact_permutation_p_value():
    # 3 vs 3 completely separated: 2 of the 20 ways to split six values are this extreme
    assert compare_values([1, 2, 3], [4, 5, 6]).p_value == pytest.approx(0.1)
    # 2 vs 2, a = {1, 4}, b = {2, 3}: every split is at least as extreme (rank sums 3..7, centre 5)
    # splits with |rank sum - 5| >= 0: all 6
    assert compare_values([1, 4], [2, 3]).p_value == pytest.approx(1.0)
    # a = {1, 2, 4}, b = {3, 5, 6}: rank sum 7, centre 10.5; |R - 10.5| >= 3.5 for R in {6, 7, 14, 15},
    # which occur 1 + 1 + 1 + 1 times out of 20
    assert compare_values([1, 2, 4], [3, 5, 6]).p_value == pytest.approx(0.2)


def test_large_groups_get_a_sampled_p_value_that_never_reaches_zero():
    result = compare_values(range(40), range(100, 140), permutations=999)
    assert result.p_value == pytest.approx(1 / 1000)


def test_interval_is_a_bootstrap_interval_around_the_effect_size():
    # completely separated groups stay separated in every resample
    separated = compare_values(range(20), range(100, 120))
    assert (separated.ci_low, separated.ci_high) == (-1, -1)
    # overlapping groups: the interval spans the estimate and stays inside [-1, 1]
    overlap = compare_values(range(0, 30), range(10, 40))
    assert -1 <= overlap.ci_low < overlap.effect_size < overlap.ci_high <= 1
    assert overlap.ci_high - overlap.ci_low > 0.2


def test_same_seed_gives_the_same_result():
    first = compare_values(range(0, 30), range(10, 40), rng=np.random.default_rng(5))
    second = compare_values(range(0, 30), range(10, 40), rng=np.random.default_rng(5))
    assert first == second


FAST = {"permutations": 999, "bootstrap": 500}


@pytest.mark.parametrize("seed", range(5))
def test_planted_difference_has_the_largest_effect_size(tmp_path, seed):
    planted = [synthetic.PlantedDifference("androsterone", 0.6, group="PSSD")]
    groups = {"PSSD": 100, "healthy control": 100}
    data = lab_dataset(tmp_path, groups=groups, planted=planted, seed=seed)
    table = compare_groups(data, "PSSD", "healthy control", **FAST)
    largest = table.iloc[0]
    assert largest["marker_id"] == "androsterone"
    assert largest["effect_size"] < 0  # PSSD lower
    assert largest["ci_high"] < 0
    assert largest["p_value"] < 0.01


def test_without_a_planted_difference_intervals_mostly_contain_zero(tmp_path):
    data = lab_dataset(tmp_path, null=True, seed=1)
    table = compare_groups(data, "PSSD", "healthy control", **FAST)
    contains_zero = (table["ci_low"] <= 0) & (table["ci_high"] >= 0)
    assert contains_zero.mean() >= 0.85  # 95% intervals; about 2 of 52 should miss by chance
    assert table["effect_size"].abs().median() < 0.2


def test_every_row_compares_within_one_sex_and_names_both_group_sizes(tmp_path):
    data = lab_dataset(tmp_path, groups={"PSSD": 40, "healthy control": 20}, female_share=0.25)
    table = compare_groups(data, "PSSD", "healthy control", **FAST)
    assert len(table) == 2 * len(synthetic.MARKERS)
    sizes = table.groupby("sex")[["n_a", "n_b"]].agg(set)
    assert sizes.loc["female"].tolist() == [{10}, {5}]
    assert sizes.loc["male"].tolist() == [{30}, {15}]
    assert set(table["group_a"]) == {"PSSD"} and set(table["group_b"]) == {"healthy control"}


def test_planted_difference_in_one_sex_shows_only_in_that_sex(tmp_path):
    planted = [synthetic.PlantedDifference("androsterone", 0.4, group="PSSD", sex="male")]
    data = lab_dataset(tmp_path, planted=planted, seed=2)
    table = compare_groups(data, "PSSD", "healthy control", **FAST).set_index(["marker_id", "sex"])
    assert table.at[("androsterone", "male"), "ci_high"] < 0
    assert (
        table.at[("androsterone", "female"), "ci_low"] <= 0 <= table.at[("androsterone", "female"), "ci_high"]
    )


def test_same_seed_gives_the_same_table(tmp_path):
    data = lab_dataset(tmp_path)
    first = compare_groups(data, "PSSD", "healthy control", seed=4, **FAST)
    second = compare_groups(data, "PSSD", "healthy control", seed=4, **FAST)
    assert first.equals(second)


def test_groups_must_be_known_different_and_present(tmp_path):
    data = lab_dataset(tmp_path)
    with pytest.raises(ValueError, match="Unknown group"):
        compare_groups(data, "PSSD", "controls")
    with pytest.raises(ValueError, match="two different groups"):
        compare_groups(data, "PSSD", "PSSD")
    with pytest.raises(ValueError, match="No loaded report is in the group 'PFS'"):
        compare_groups(data, "PSSD", "PFS")


def test_empty_or_missing_values_are_refused():
    with pytest.raises(ValueError, match="at least one value"):
        compare_values([], [1, 2])
    with pytest.raises(ValueError, match="missing"):
        compare_values([1, float("nan")], [1, 2])
