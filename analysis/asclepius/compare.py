"""Compare markers between two groups of lab reports, within sex (spec A1, S1, S3).

`compare_values` compares one marker's values in two groups: a two-sided permutation test on the
rank sum (Mann-Whitney), Cliff's delta as the effect size and a percentile bootstrap interval.
`compare_groups` runs it for every marker in each sex and returns the findings table.

Cliff's delta is P(a > b) - P(a < b) for a random report from each group: -1 when every value in
group a is below every value in group b, 0 for no difference, 1 for the reverse. Romano et al.
(2006) call |delta| of 0.147, 0.33 and 0.474 small, medium and large.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from asclepius import dataset
from asclepius.dataset import LabDataset
from asclepius.metadata import GROUPS

Value = float | str  # a number, or a limit such as "<5"
PERMUTATIONS = 9999
BOOTSTRAP = 2000
CONFIDENCE = 0.95
SEXES = tuple(sorted(dataset.SEXES))  # ordered, for seeding
KIND = "group comparison"  # the findings table's `comparison` column; see asclepius.refrange
COLUMNS = (
    "comparison",
    "marker_id",
    "marker_name",
    "sex",
    "group_a",
    "group_b",
    "n_a",
    "n_b",
    "n_censored",
    "age_a",
    "age_b",
    "median_a",
    "median_b",
    "effect_size",
    "ci_low",
    "ci_high",
    "p_value",
)


@dataclass(frozen=True)
class Comparison:
    """Group a against group b. `effect_size` is Cliff's delta, `ci_low`..`ci_high` its interval."""

    n_a: int
    n_b: int
    effect_size: float
    ci_low: float
    ci_high: float
    p_value: float


def compare_values(
    a: Iterable[Value],
    b: Iterable[Value],
    *,
    permutations: int = PERMUTATIONS,
    bootstrap: int = BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> Comparison:
    """Compare values `a` with values `b`. Neither may be empty or contain NaN.

    A value is a number, or a limit as a lab prints it: "<5" (below 5) or ">5" (above 5). Two
    values count as higher or lower only when that is certain; "<5" against 4 is a tie.
    """
    values_a, values_b = list(a), list(b)
    if not values_a or not values_b:
        raise ValueError("Both groups need at least one value.")
    above = _above(values_a + values_b)
    signs = above.astype(np.int64) - above.T  # +1: row value certainly higher, -1: lower, 0: undecided
    n_a = len(values_a)
    between = signs[:n_a, n_a:]
    rng = rng if rng is not None else np.random.default_rng(0)
    low, high = _bootstrap_interval(between, bootstrap, rng)
    return Comparison(
        n_a=n_a,
        n_b=len(values_b),
        effect_size=float(between.mean()),
        ci_low=low,
        ci_high=high,
        p_value=_permutation_p(signs.sum(axis=1), n_a, permutations, rng),
    )


def _above(values: list[Value]) -> np.ndarray:
    """`result[i, j]` is True when value i is certainly higher than value j.

    Each value is a range from `low` to `high`; a limit's end at the printed number is open ("<5"
    never equals 5), and its other end is infinite.
    """
    low = np.empty(len(values))
    high = np.empty(len(values))
    low_open = np.zeros(len(values), dtype=bool)
    high_open = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values):
        comparator, number = parse_value(value)
        low[i], high[i] = number, number
        if comparator == "<":
            low[i], high_open[i] = -np.inf, True
        elif comparator == ">":
            high[i], low_open[i] = np.inf, True
    touching = (low[:, None] == high[None, :]) & (low_open[:, None] | high_open[None, :])
    return (low[:, None] > high[None, :]) | touching


def parse_value(value: Value) -> tuple[str, float]:
    comparator, text = "", value
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ("<", ">"):
            comparator, text = text[0], text[1:]
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{value!r} is not a number, or a limit like <5 or >5.") from None
    if math.isnan(number):
        raise ValueError("Values must not be missing (NaN).")
    return comparator, number


def compare_groups(
    data: LabDataset,
    group_a: str,
    group_b: str,
    *,
    permutations: int = PERMUTATIONS,
    bootstrap: int = BOOTSTRAP,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare every marker between `group_a` and `group_b`, separately in each sex.

    One row per marker and sex where both groups have at least one value, with the columns in
    `COLUMNS`, largest effect first. Reports with no sex are left out, because comparisons are made
    within sex. A value given as "<x" or ">x" is compared as a range (see `compare_values`);
    `n_censored` counts them, and the medians read them as x. `age_a` and `age_b` are each group's
    median age, so an age difference shows; nothing is adjusted for age. The same `seed` gives the
    same table.
    Group-size limits (spec S4) and correction for multiple tests (S2) are not applied here.
    """
    for group in (group_a, group_b):
        if group not in GROUPS:
            raise ValueError(f"Unknown group {group!r}; use one of {', '.join(GROUPS)}.")
    if group_a == group_b:
        raise ValueError("Compare two different groups.")
    records = data.records.set_index("record")
    for group in (group_a, group_b):
        if not (records["group"] == group).any():
            raise ValueError(f"No loaded report is in the group {group!r}.")

    values = data.markers[data.markers["value"].notna()]  # the loader allows one row per report and marker
    report = records[["group", "sex", "age_years"]].rename(columns={"sex": "report_sex", "age_years": "age"})
    values = values.join(report, on="record")
    values = values[values["group"].isin([group_a, group_b]) & values["report_sex"].isin(SEXES)]

    rows = []
    names = data.markers.drop_duplicates("marker_id").set_index("marker_id")["marker_name"]
    for (marker, sex), cell in values.groupby(["marker_id", "report_sex"], sort=True):
        in_a, in_b = cell["group"] == group_a, cell["group"] == group_b
        if not in_a.any() or not in_b.any():
            continue
        limited = cell["comparator"].notna()
        as_printed = (
            cell["value"].astype(object).where(~limited, cell["comparator"] + cell["value"].map(repr))
        )
        # seeded per marker and sex, so a row does not change when other markers are added
        rng = np.random.default_rng([seed, zlib.crc32(str(marker).encode()), SEXES.index(str(sex))])
        result = compare_values(
            as_printed[in_a], as_printed[in_b], permutations=permutations, bootstrap=bootstrap, rng=rng
        )
        rows.append(
            {
                "comparison": KIND,
                "marker_id": marker,
                "marker_name": names[marker],
                "sex": sex,
                "group_a": group_a,
                "group_b": group_b,
                "n_a": result.n_a,
                "n_b": result.n_b,
                "n_censored": int(limited.sum()),
                "age_a": cell.loc[in_a, "age"].astype("float64").median(),
                "age_b": cell.loc[in_b, "age"].astype("float64").median(),
                "median_a": float(cell.loc[in_a, "value"].median()),
                "median_b": float(cell.loc[in_b, "value"].median()),
                "effect_size": result.effect_size,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "p_value": result.p_value,
            }
        )
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    order = table["effect_size"].abs().sort_values(ascending=False, kind="stable").index
    return table.loc[order].reset_index(drop=True)


def _permutation_p(scores: np.ndarray, n_a: int, permutations: int, rng: np.random.Generator) -> float:
    """Two-sided permutation p-value of Gehan's statistic, exact when there are few enough splits.

    `scores[i]` is how many values value i is certainly above, minus how many it is certainly
    below. The statistic is the sum of group a's scores; it is 0 on average under no difference.
    With no limits among the values a score is 2 * rank - (n + 1), so this is the rank-sum test.
    """
    n = len(scores)
    observed = abs(int(scores[:n_a].sum()))
    if math.comb(n, n_a) <= permutations:
        sums = np.array([scores[list(split)].sum() for split in combinations(range(n), n_a)])
        return float(np.mean(np.abs(sums) >= observed))
    extreme = 0
    for start in range(0, permutations, 1000):
        count = min(1000, permutations - start)
        shuffled = rng.permuted(np.tile(scores, (count, 1)), axis=1)
        extreme += int(np.sum(np.abs(shuffled[:, :n_a].sum(axis=1)) >= observed))
    return (extreme + 1) / (permutations + 1)


def _bootstrap_interval(signs: np.ndarray, resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap interval of Cliff's delta, resampling each group separately.

    A resample is a count of how often each value was drawn, so its delta is
    `counts_a @ signs @ counts_b / (n_a * n_b)` without building the resampled groups.
    """
    n_a, n_b = signs.shape
    counts_a = rng.multinomial(n_a, np.full(n_a, 1 / n_a), size=resamples)
    counts_b = rng.multinomial(n_b, np.full(n_b, 1 / n_b), size=resamples)
    deltas = np.einsum("ri,ij,rj->r", counts_a, signs, counts_b) / (n_a * n_b)
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(deltas, [tail, 100 - tail])
    return float(low), float(high)
