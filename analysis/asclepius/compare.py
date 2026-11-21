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

from asclepius.dataset import LabDataset
from asclepius.metadata import GROUPS

PERMUTATIONS = 9999
BOOTSTRAP = 2000
CONFIDENCE = 0.95
SEXES = ("female", "male")
COLUMNS = (
    "marker_id",
    "marker_name",
    "sex",
    "group_a",
    "group_b",
    "n_a",
    "n_b",
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
    a: Iterable[float],
    b: Iterable[float],
    *,
    permutations: int = PERMUTATIONS,
    bootstrap: int = BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> Comparison:
    """Compare values `a` with values `b`. Neither may be empty or contain NaN."""
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if not len(x) or not len(y):
        raise ValueError("Both groups need at least one value.")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Values must not be missing (NaN).")
    rng = rng if rng is not None else np.random.default_rng(0)
    signs = np.sign(x[:, None] - y[None, :])
    low, high = _bootstrap_interval(signs, bootstrap, rng)
    return Comparison(
        n_a=len(x),
        n_b=len(y),
        effect_size=float(signs.mean()),
        ci_low=low,
        ci_high=high,
        p_value=_permutation_p(x, y, permutations, rng),
    )


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
    within sex. A value given as "<x" or ">x" counts as x. The same `seed` gives the same table.
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

    values = data.markers[data.markers["value"].notna()]
    values = values.drop_duplicates(["record", "marker_id"])  # one value per report and marker
    values = values.join(records[["group", "sex"]].rename(columns={"sex": "report_sex"}), on="record")
    values = values[values["group"].isin([group_a, group_b]) & values["report_sex"].isin(SEXES)]

    rows = []
    names = data.markers.drop_duplicates("marker_id").set_index("marker_id")["marker_name"]
    for (marker, sex), cell in values.groupby(["marker_id", "report_sex"], sort=True):
        a = cell.loc[cell["group"] == group_a, "value"].to_numpy()
        b = cell.loc[cell["group"] == group_b, "value"].to_numpy()
        if not len(a) or not len(b):
            continue
        # seeded per marker and sex, so a row does not change when other markers are added
        rng = np.random.default_rng([seed, zlib.crc32(str(marker).encode()), SEXES.index(str(sex))])
        result = compare_values(a, b, permutations=permutations, bootstrap=bootstrap, rng=rng)
        rows.append(
            {
                "marker_id": marker,
                "marker_name": names[marker],
                "sex": sex,
                "group_a": group_a,
                "group_b": group_b,
                "n_a": result.n_a,
                "n_b": result.n_b,
                "median_a": float(np.median(a)),
                "median_b": float(np.median(b)),
                "effect_size": result.effect_size,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "p_value": result.p_value,
            }
        )
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    order = table["effect_size"].abs().sort_values(ascending=False, kind="stable").index
    return table.loc[order].reset_index(drop=True)


def _permutation_p(x: np.ndarray, y: np.ndarray, permutations: int, rng: np.random.Generator) -> float:
    """Two-sided p-value of the rank sum of `x`, exact when there are few enough splits."""
    n_a, n = len(x), len(x) + len(y)
    ranks = pd.Series(np.concatenate([x, y])).rank(method="average").to_numpy()
    centre = n_a * (n + 1) / 2
    observed = abs(ranks[:n_a].sum() - centre) - 1e-9  # tolerance for float sums of half ranks
    if math.comb(n, n_a) <= permutations:
        sums = np.array([ranks[list(split)].sum() for split in combinations(range(n), n_a)])
        return float(np.mean(np.abs(sums - centre) >= observed))
    extreme = 0
    for start in range(0, permutations, 1000):
        count = min(1000, permutations - start)
        shuffled = rng.permuted(np.tile(ranks, (count, 1)), axis=1)
        extreme += int(np.sum(np.abs(shuffled[:, :n_a].sum(axis=1) - centre) >= observed))
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
