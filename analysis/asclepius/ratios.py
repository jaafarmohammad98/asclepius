"""Enzyme-activity ratios between metabolites (spec A2).

The leading hypothesis is about enzyme activity, which shows up as a ratio of a product to its
sibling or precursor rather than as either marker alone. `RATIOS` is the built-in list, as data:
the README table is generated from it (`documentation_table`). The list was chosen without a
clinician, so it is marked "provisional, needs expert review" until M3-11.

`compute_ratios` turns a `LabDataset` into one of the same shape whose markers are the ratios, so
`compare_ratios` runs them through the same comparison as raw markers (`asclepius.compare`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asclepius.compare import BOOTSTRAP, PERMUTATIONS, compare_groups
from asclepius.dataset import LabDataset

STATUS = "provisional, needs expert review"
KIND = "enzyme-activity ratio"  # the findings table's `comparison` column, and each ratio's `category`
MISSING, ZERO, BOTH_LIMITS = "component missing", "component zero", "both components limits"


@dataclass(frozen=True)
class Ratio:
    """`numerator` / `denominator` (marker ids), a proxy for `proxy_for`."""

    name: str
    numerator: str
    denominator: str
    proxy_for: str

    @property
    def id(self) -> str:
        return f"{self.numerator}/{self.denominator}"


_5AR = "5-alpha vs 5-beta reductase"
RATIOS = (
    Ratio("Androsterone / etiocholanolone", "androsterone", "etiocholanolone", f"{_5AR} (androgens)"),
    Ratio(
        "5a-androstanediol / 5b-androstanediol",
        "5a_androstanediol",
        "5b_androstanediol",
        f"{_5AR} (androgens)",
    ),
    Ratio("a-THF / b-THF", "a_tetrahydrocortisol_a_thf", "b_tetrahydrocortisol_b_thf", f"{_5AR} (cortisol)"),
    Ratio("a-pregnanediol / b-pregnanediol", "a_pregnanediol", "b_pregnanediol", f"{_5AR} (progesterone)"),
    Ratio(
        "Free cortisol / free cortisone", "24hr_free_cortisol", "24hr_free_cortisone", "11-beta-HSD balance"
    ),
    Ratio("2-methoxy-E1 / 2-OH-E1", "2_methoxy_e1", "2_oh_e1", "Methylation (COMT) activity"),
    Ratio("2-OH-E1 / 16-OH-E1", "2_oh_e1", "16_oh_e1", "Estrogen metabolism pathway preference"),
    Ratio("2-OH-E1 / 4-OH-E1", "2_oh_e1", "4_oh_e1", "Estrogen metabolism pathway preference"),
)

# The ratio's limit, from each component's comparator ("" for an exact value). "<10" over 2 is
# below 5; 10 over "<2" is above 5. Two limits the same way say nothing, so they are not listed.
_LIMIT = {
    ("", ""): "",
    ("<", ""): "<",
    (">", ""): ">",
    ("", "<"): ">",
    ("", ">"): "<",
    ("<", ">"): "<",
    (">", "<"): ">",
}


@dataclass(frozen=True)
class RatioValues:
    """`data`: a `LabDataset` whose markers are the ratios (one row per report and ratio computed),
    with the original `records`, so anything that takes a `LabDataset` works on it. `skipped`: per
    ratio and reason (`component missing`, `component zero`, `both components limits`), how many
    reports had no value for it."""

    data: LabDataset
    skipped: pd.DataFrame


@dataclass(frozen=True)
class RatioFindings:
    """`rows`: findings-table rows, one per ratio and sex. `skipped`: as in `RatioValues`."""

    rows: pd.DataFrame
    skipped: pd.DataFrame


def compute_ratios(data: LabDataset) -> RatioValues:
    """Every ratio in `RATIOS` for every report in `data`, skipping and counting those that cannot be made.

    A ratio is skipped for a report when a component is missing or blank, when either is zero, or
    when both are limits pointing the same way ("<10" over "<2" could be anything). One limit gives
    a limit on the ratio. Ratios have no reference range; `unit` is "ratio".
    """
    records = data.records["record"]
    markers = data.markers  # the loader allows one row per report and marker
    value = markers.pivot(index="record", columns="marker_id", values="value").reindex(records)
    comparator = (
        markers.assign(comparator=markers["comparator"].fillna(""))
        .pivot(index="record", columns="marker_id", values="comparator")
        .reindex(records)
    )
    rows, skipped = [], []
    for ratio in RATIOS:
        numerator, denominator = (_column(value, m) for m in (ratio.numerator, ratio.denominator))
        limit = pd.Series(
            list(
                zip(_column(comparator, ratio.numerator), _column(comparator, ratio.denominator), strict=True)
            ),
            index=value.index,
        ).map(_LIMIT)
        reason = _skip_reasons(numerator, denominator, limit)
        for why, count in reason.value_counts(sort=False).items():
            skipped.append(
                {"ratio_id": ratio.id, "ratio_name": ratio.name, "reason": why, "n_reports": count}
            )
        made = reason.isna()
        rows.append(
            pd.DataFrame(
                {
                    "record": value.index[made],
                    "marker_id": ratio.id,
                    "marker_name": ratio.name,
                    "category": KIND,
                    "comparator": limit[made].replace("", None).to_numpy(),
                    "value": (numerator[made] / denominator[made]).to_numpy(dtype="float64"),
                    "unit": "ratio",
                    "ref_low": np.nan,
                    "ref_high": np.nan,
                    "status": STATUS,
                }
            )
        )
    reports = data.records.set_index("record")
    report_columns = [c for c in data.markers.columns if c in reports.columns and c != "record"]
    ratio_markers = pd.concat(rows, ignore_index=True).join(reports[report_columns], on="record")
    skipped_table = pd.DataFrame(skipped, columns=["ratio_id", "ratio_name", "reason", "n_reports"])
    ratio_data = LabDataset(
        markers=ratio_markers,
        fingerprint=data.fingerprint,
        records=data.records,
        unmatched=data.unmatched,
        kind=data.kind,
    )
    return RatioValues(data=ratio_data, skipped=skipped_table)


def _skip_reasons(numerator: pd.Series, denominator: pd.Series, limit: pd.Series) -> pd.Series:
    """Why each report has no value for a ratio, or missing when it has one. A missing component
    outranks a zero, which outranks two limits the same way (`limit` missing)."""
    reason = pd.Series(None, index=numerator.index, dtype=object)
    reason = reason.mask(limit.isna(), BOTH_LIMITS)
    reason = reason.mask((numerator == 0) | (denominator == 0), ZERO)
    return reason.mask(numerator.isna() | denominator.isna(), MISSING)


def _column(table: pd.DataFrame, marker: str) -> pd.Series:
    """One marker's column of a record-by-marker table; all missing when no report has it."""
    return table[marker] if marker in table.columns else pd.Series(np.nan, index=table.index)


def compare_ratios(
    data: LabDataset,
    group_a: str,
    group_b: str,
    *,
    permutations: int = PERMUTATIONS,
    bootstrap: int = BOOTSTRAP,
    seed: int = 0,
) -> RatioFindings:
    """Compare every ratio between two groups, within sex, exactly as `compare_groups` compares markers.

    `rows` has `compare_groups`'s columns, with `comparison` set to "enzyme-activity ratio", plus
    `status`, "provisional, needs expert review" until M3-11. `skipped` is `compute_ratios`'s, over
    every loaded report.
    """
    computed = compute_ratios(data)
    table = compare_groups(
        computed.data, group_a, group_b, permutations=permutations, bootstrap=bootstrap, seed=seed
    )
    return RatioFindings(rows=table.assign(comparison=KIND, status=STATUS), skipped=computed.skipped)


def documentation_table() -> str:
    """The ratio list as a Markdown table, for the README."""
    lines = [
        f"Ratio list: {STATUS}.",
        "",
        "| Ratio | Numerator | Denominator | Proxy for |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r.name} | `{r.numerator}` | `{r.denominator}` | {r.proxy_for} |" for r in RATIOS]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(documentation_table(), end="")
