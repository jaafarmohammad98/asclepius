"""One analysis run: every test, one false discovery rate correction, a Tier A re-run of every
finding, strength labels and a run record (spec S2, S4 to S8, S10).

`run_analysis` compares raw markers and enzyme-activity ratios between two groups of a lab-report
dataset, stacks them into one findings table and corrects them together, so a run of 68 tests is
corrected as 68 tests, never per call. `label_findings` does the correction and labelling for any
findings table; self-report analyses (M3-12, M3-14) use it with `self_report=True`.

The thresholds live in `Settings`, which is recorded with every run. They were chosen without a
clinician and are marked provisional until the review in M3-11.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from asclepius.cluster import count_text
from asclepius.compare import BOOTSTRAP, PERMUTATIONS, compare_groups
from asclepius.dataset import LabDataset
from asclepius.ratios import STATUS, compare_ratios
from asclepius.record import run_record
from asclepius.refrange import combine_findings
from asclepius.selfreport import LABEL as SELF_REPORT_LABEL

if TYPE_CHECKING:  # subgroups imports this module for Settings and the Tier A verdict
    from asclepius.subgroups import SubgroupLinks

STRONG, SUGGESTIVE, NOISE = "strong", "suggestive", "noise-level"
NOTHING_RELIABLE = "nothing reliable yet"
PROVISIONAL = STATUS  # "provisional, needs expert review", until M3-11
TIER_A_NAME = "A"  # metadata.TIERS[0]: confirmed by a clinician
TIER_A = "holds_in_tier_a"  # set by the Tier A re-run; missing means not shown to hold
HOLDS, DOES_NOT_HOLD, TOO_SMALL = "holds", "does not hold", "Tier A too small to test"
KEY = ["comparison", "marker_id", "sex"]  # one findings row
TIER_A_COLUMNS = {  # Tier A re-run column -> the full-dataset column it repeats
    "tier_a_n_a": "n_a",
    "tier_a_n_b": "n_b",
    "tier_a_effect_size": "effect_size",
    "tier_a_ci_low": "ci_low",
    "tier_a_ci_high": "ci_high",
    "tier_a_p_value": "p_value",
}


@dataclass(frozen=True)
class Settings:
    """Every threshold a run uses, in one place (spec S4, S5). `record()` is what a run stores.

    A finding is strong when its corrected p-value (q) is at most `strong_q`, its effect size
    (Cliff's delta) is at least `medium_effect` either way and it holds in Tier A records;
    suggestive when q is at most `suggestive_q`; noise-level otherwise. 0.33 is Romano et al.'s (2006) "medium" for Cliff's delta.

    A finding holds in Tier A when the Tier A re-run has at least `min_group_size` values in each
    group, an effect the same way of at least `medium_effect`, and a p-value of at most `tier_a_p`.
    That p-value is not corrected again: it checks findings the corrected run already found.
    """

    min_group_size: int = 10
    correction: str = "Benjamini-Hochberg false discovery rate, over every test in the run"
    strong_q: float = 0.05
    suggestive_q: float = 0.10
    medium_effect: float = 0.33
    tier_a_p: float = 0.05
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
        if not 0 < self.tier_a_p <= 1:
            raise ValueError("tier_a_p must be above 0 and at most 1.")

    def record(self) -> dict[str, object]:
        return asdict(self)


class GroupTooSmall(ValueError):
    """A comparison was refused because a group has fewer reports than `Settings.min_group_size`."""


@dataclass(frozen=True)
class RunResult:
    """`findings`: every test, with `q_value` and `label`. `refused`: rows not tested because a
    group had too few values in that sex, with the `reason`. `skipped`: ratios that could not be
    made, as in `compare_ratios`. `statement`: "nothing reliable yet" when no finding is strong
    or suggestive (spec S8), else how many are. `settings`: `Settings.record()`. `record`: the
    run record (`asclepius.record.run_record`: file hashes, tool version, settings, date)."""

    findings: pd.DataFrame
    refused: pd.DataFrame
    skipped: pd.DataFrame
    statement: str
    settings: dict[str, object]
    record: dict[str, object]


def run_analysis(
    data: LabDataset,
    group_a: str,
    group_b: str,
    settings: Settings | None = None,
    *,
    subgroups: SubgroupLinks | None = None,
) -> RunResult:
    """Compare every raw marker and enzyme-activity ratio between two groups, within sex, then
    correct all of them together and label each (spec S2, S4, S5, S8).

    Refuses, with `GroupTooSmall`, when either group has fewer than `settings.min_group_size`
    reports. A marker or ratio whose values in one sex fall short of that in either group is not
    tested: it goes to `refused` and does not count in the correction.

    `subgroups`, from `asclepius.subgroups.link_subgroups` on the same data and settings, adds the
    subgroup findings to the table, corrected and labelled with everything else; its refused tests
    join `refused`, the statement ends with the clusters' statement (or that there was nothing to
    link), and the record gains `subgroups`, the clustering's statement and settings.
    """
    settings = settings or Settings()
    if subgroups is not None:
        if subgroups.fingerprint != data.fingerprint:
            raise ValueError("These subgroup links were made from a different dataset.")
        if subgroups.settings != settings.record():
            raise ValueError(
                "Link subgroups with the same settings as the run, so one correction covers both."
            )
    for group in (group_a, group_b):
        n = int((data.records["group"] == group).sum())
        if 0 < n < settings.min_group_size:
            raise GroupTooSmall(
                f"Refused: the group {group!r} has {count_text(n)} reports, and a comparison needs at least "
                f"{settings.min_group_size} in each group (spec S4, {settings.status})."
            )
    table, skipped = group_tests(data, group_a, group_b, settings)
    tested, refused = refuse_small(table, settings)
    tested = tier_a_rerun(tested, data, group_a, group_b, settings)
    record = run_record(data, (group_a, group_b), settings.record())
    if subgroups is not None:
        tested = combine_findings(tested, subgroups.rows)
        refused = combine_findings(refused, subgroups.refused)
        record["subgroups"] = subgroups.clusters
    findings = label_findings(tested, settings)
    return RunResult(
        findings=findings,
        refused=refused,
        skipped=skipped,
        statement=statement(findings) + ("" if subgroups is None else f"; subgroups: {subgroups.statement}"),
        settings=settings.record(),
        record=record,
    )


def group_tests(
    data: LabDataset, group_a: str, group_b: str, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every marker and ratio compared, stacked into one table, and the ratios' `skipped`."""
    options = {"permutations": settings.permutations, "bootstrap": settings.bootstrap, "seed": settings.seed}
    markers = compare_groups(data, group_a, group_b, **options)
    ratios = compare_ratios(data, group_a, group_b, **options)
    return combine_findings(markers, ratios.rows), ratios.skipped


def refuse_small(table: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `group_tests` rows into those tested and those refused because a group has fewer than
    `min_group_size` values in that sex (spec S4), with the `reason`."""
    small = (table["n_a"] < settings.min_group_size) | (table["n_b"] < settings.min_group_size)
    refused = table.loc[small, ["comparison", "marker_id", "marker_name", "sex", "group_a", "group_b"]]
    refused = refused.assign(
        n_a=table.loc[small, "n_a"],
        n_b=table.loc[small, "n_b"],
        reason=f"fewer than {settings.min_group_size} values in a group in this sex (spec S4)",
    ).reset_index(drop=True)
    return table[~small].reset_index(drop=True), refused


def tier_a_rerun(
    findings: pd.DataFrame, data: LabDataset, group_a: str, group_b: str, settings: Settings
) -> pd.DataFrame:
    """Re-run every finding on Tier A records only (spec S6) and add the result beside it.

    Adds `tier_a_n_a`, `tier_a_n_b` (Tier A values per group), `tier_a_effect_size`,
    `tier_a_ci_low`, `tier_a_ci_high`, `tier_a_p_value` (uncorrected), `tier_a_result` (`holds`,
    `does not hold`, or `Tier A too small to test`) and `holds_in_tier_a`. When a group has fewer
    than `min_group_size` Tier A values, the comparison is refused as in the full run (spec S4):
    only the sizes are shown.
    """
    records = data.records[data.records["tier"] == TIER_A_NAME]
    tier_a = replace(
        data, records=records, markers=data.markers[data.markers["record"].isin(records["record"])]
    )
    if (records["group"] == group_a).any() and (records["group"] == group_b).any():
        rerun, _ = group_tests(tier_a, group_a, group_b, settings)
    else:  # a group has no Tier A report: nothing to compare
        rerun = pd.DataFrame(columns=[*KEY, *TIER_A_COLUMNS.values()])
    rerun = rerun[[*KEY, *TIER_A_COLUMNS.values()]].rename(columns={v: k for k, v in TIER_A_COLUMNS.items()})
    out = findings.merge(rerun, on=KEY, how="left", validate="one_to_one")
    return judge_tier_a(out, settings)


def judge_tier_a(out: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add `tier_a_result` and `holds_in_tier_a` to rows that have the `TIER_A_COLUMNS` (a missing
    size counts as 0), and hide the Tier A statistics of a re-run too small to test (spec S4).
    Rows of one group against a fixed reference (a reference range) have no `tier_a_n_b`."""
    out = out.copy()
    sizes = [c for c in ("tier_a_n_a", "tier_a_n_b") if c in out.columns]
    for column in sizes:
        out[column] = out[column].fillna(0).astype("int64")
    too_small = (out[sizes] < settings.min_group_size).any(axis=1)
    stats = ["tier_a_effect_size", "tier_a_ci_low", "tier_a_ci_high", "tier_a_p_value"]
    out[stats] = out[stats].astype("float64").mask(too_small)
    same_way = np.sign(out["tier_a_effect_size"]) == np.sign(out["effect_size"])
    holds = (
        ~too_small
        & same_way
        & (out["tier_a_effect_size"].abs() >= settings.medium_effect)
        & (out["tier_a_p_value"] <= settings.tier_a_p)
    )
    result = (
        pd.Series(DOES_NOT_HOLD, index=out.index, dtype=object).mask(holds, HOLDS).mask(too_small, TOO_SMALL)
    )
    return out.assign(tier_a_result=result, **{TIER_A: holds})


def label_findings(table: pd.DataFrame, settings: Settings, *, self_report: bool = False) -> pd.DataFrame:
    """Add `q_value` (Benjamini-Hochberg over every row with a p-value, together) and `label`.

    Pass the whole run's table: correcting parts of it separately would understate the number of
    tests. Rows with no p-value (reference-range rows) are descriptions, not tests: they get no
    q-value and no label.

    Strong needs all three conditions of spec S5: q at most `strong_q`, an effect of at least
    `medium_effect`, and `holds_in_tier_a` True. That column comes from `tier_a_rerun`;
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
