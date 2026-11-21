"""One analysis run: every test, one false discovery rate correction, strength labels (spec S2, S4, S5, S8, S10).

`run_analysis` compares raw markers and enzyme-activity ratios between two groups of a lab-report
dataset, stacks them into one findings table and corrects them together, so a run of 68 tests is
corrected as 68 tests, never per call. `label_findings` does the correction and labelling for any
findings table; self-report analyses (M3-12, M3-14) use it with `self_report=True`.

The thresholds live in `Settings`, which is recorded with every run. They were chosen without a
clinician and are marked provisional until the review in M3-11.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from asclepius.compare import BOOTSTRAP, PERMUTATIONS, compare_groups
from asclepius.dataset import LabDataset
from asclepius.ratios import STATUS, compare_ratios
from asclepius.refrange import combine_findings
from asclepius.selfreport import LABEL as SELF_REPORT_LABEL

STRONG, SUGGESTIVE, NOISE = "strong", "suggestive", "noise-level"
NOTHING_RELIABLE = "nothing reliable yet"
PROVISIONAL = STATUS  # "provisional, needs expert review", until M3-11
TIER_A = "holds_in_tier_a"  # set by the Tier A re-run (M3-07); missing means not shown to hold


@dataclass(frozen=True)
class Settings:
    """Every threshold a run uses, in one place (spec S4, S5). `record()` is what a run stores.

    A finding is strong when its corrected p-value (q) is at most `strong_q`, its effect size
    (Cliff's delta) is at least `medium_effect` either way and it holds in Tier A records;
    suggestive when q is at most `suggestive_q`; noise-level otherwise. 0.33 is Romano et al.'s (2006) "medium" for Cliff's delta.
    """

    min_group_size: int = 10
    correction: str = "Benjamini-Hochberg false discovery rate, over every test in the run"
    strong_q: float = 0.05
    suggestive_q: float = 0.10
    medium_effect: float = 0.33
    permutations: int = PERMUTATIONS
    bootstrap: int = BOOTSTRAP
    seed: int = 0
    status: str = field(default=PROVISIONAL)

    def __post_init__(self) -> None:
        if self.min_group_size < 1:
            raise ValueError("min_group_size must be at least 1.")
        if not 0 < self.strong_q <= self.suggestive_q <= 1:
            raise ValueError("Need 0 < strong_q <= suggestive_q <= 1.")
        if not 0 <= self.medium_effect <= 1:
            raise ValueError("medium_effect is a Cliff's delta, between 0 and 1.")

    def record(self) -> dict[str, object]:
        return asdict(self)


class GroupTooSmall(ValueError):
    """A comparison was refused because a group has fewer reports than `Settings.min_group_size`."""


@dataclass(frozen=True)
class RunResult:
    """`findings`: every test, with `q_value` and `label`. `refused`: rows not tested because a
    group had too few values in that sex, with the `reason`. `skipped`: ratios that could not be
    made, as in `compare_ratios`. `statement`: "nothing reliable yet" when no finding is strong
    or suggestive (spec S8), else how many are. `settings`: `Settings.record()`."""

    findings: pd.DataFrame
    refused: pd.DataFrame
    skipped: pd.DataFrame
    statement: str
    settings: dict[str, object]


def run_analysis(data: LabDataset, group_a: str, group_b: str, settings: Settings | None = None) -> RunResult:
    """Compare every raw marker and enzyme-activity ratio between two groups, within sex, then
    correct all of them together and label each (spec S2, S4, S5, S8).

    Refuses, with `GroupTooSmall`, when either group has fewer than `settings.min_group_size`
    reports. A marker or ratio whose values in one sex fall short of that in either group is not
    tested: it goes to `refused` and does not count in the correction.
    """
    settings = settings or Settings()
    for group in (group_a, group_b):
        n = int((data.records["group"] == group).sum())
        if 0 < n < settings.min_group_size:
            raise GroupTooSmall(
                f"Refused: the group {group!r} has {n} reports, and a comparison needs at least "
                f"{settings.min_group_size} in each group (spec S4, {settings.status})."
            )
    options = {"permutations": settings.permutations, "bootstrap": settings.bootstrap, "seed": settings.seed}
    markers = compare_groups(data, group_a, group_b, **options)
    ratios = compare_ratios(data, group_a, group_b, **options)
    table = combine_findings(markers, ratios.rows)
    small = (table["n_a"] < settings.min_group_size) | (table["n_b"] < settings.min_group_size)
    refused = table.loc[small, ["comparison", "marker_id", "marker_name", "sex", "group_a", "group_b"]]
    refused = refused.assign(
        n_a=table.loc[small, "n_a"],
        n_b=table.loc[small, "n_b"],
        reason=f"fewer than {settings.min_group_size} values in a group in this sex (spec S4)",
    ).reset_index(drop=True)
    findings = label_findings(table[~small].reset_index(drop=True), settings)
    return RunResult(
        findings=findings,
        refused=refused,
        skipped=ratios.skipped,
        statement=statement(findings),
        settings=settings.record(),
    )


def label_findings(table: pd.DataFrame, settings: Settings, *, self_report: bool = False) -> pd.DataFrame:
    """Add `q_value` (Benjamini-Hochberg over every row with a p-value, together) and `label`.

    Pass the whole run's table: correcting parts of it separately would understate the number of
    tests. Rows with no p-value (reference-range rows) are descriptions, not tests: they get no
    q-value and no label.

    Strong needs all three conditions of spec S5: q at most `strong_q`, an effect of at least
    `medium_effect`, and `holds_in_tier_a` True. That column comes from the Tier A re-run (M3-07);
    a row without it holds only in the full dataset, so it is at most suggestive. With
    `self_report`, no row can be strong (spec S10), and every row carries the caveat
    "self-report, no controls".
    """
    tested = _column(table, "p_value").notna()
    p = table.loc[tested, "p_value"].astype("float64")
    q = pd.Series(np.nan, index=table.index)
    q[tested] = benjamini_hochberg(p.to_numpy())
    effect = _column(table, "effect_size").abs()
    in_tier_a = _column(table, TIER_A).eq(True)
    strong = tested & (q <= settings.strong_q) & (effect >= settings.medium_effect) & in_tier_a
    strong &= not self_report
    suggestive = tested & ~strong & (q <= settings.suggestive_q)
    label = pd.Series(None, index=table.index, dtype=object)
    label[tested] = NOISE
    label[suggestive] = SUGGESTIVE
    label[strong] = STRONG
    out = table.assign(q_value=q, label=label)
    if self_report:
        out = out.assign(caveat=SELF_REPORT_LABEL)
    return out


def _column(table: pd.DataFrame, name: str) -> pd.Series:
    """A column of `table`, or all missing when it has none."""
    return table[name] if name in table.columns else pd.Series(np.nan, index=table.index)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), in the order given."""
    m = len(p)
    if m == 0:
        return np.array([], dtype="float64")
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(adjusted, 1.0)
    return q


def statement(findings: pd.DataFrame) -> str:
    """ "nothing reliable yet" when no finding is strong or suggestive (spec S8), else the counts."""
    counts = findings["label"].value_counts()
    strong, suggestive = int(counts.get(STRONG, 0)), int(counts.get(SUGGESTIVE, 0))
    if strong == suggestive == 0:
        return NOTHING_RELIABLE
    return f"{strong} strong and {suggestive} suggestive findings ({PROVISIONAL} thresholds)"
