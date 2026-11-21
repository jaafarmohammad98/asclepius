"""Place values within their own report's reference range (spec A1, A8).

Without controls, the only comparison available is the range printed on the report. A value is
`below`, `within` or `above` that range, or `unclear` when it is a limit ("<x", ">x") that could
fall on either side of a range end. A range end that is not given has no side beyond it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from asclepius.compare import Value, parse_value
from asclepius.dataset import LabDataset
from asclepius.selfreport import SelfReportDataset

CONFIDENCE_Z = 1.959963984540054  # two-sided 95%
LAB = "reference range"
SELF_REPORT = "reference range, self-report"
SIDES = ("below", "within", "above")
COLUMNS = (
    "comparison",
    "marker_id",
    "marker_name",
    "sex",
    "group_a",
    "n_a",
    "n_unclear",
    "age_a",
    "median_position",
    *(c for side in SIDES for c in (f"share_{side}", f"{side}_low", f"{side}_high")),
)


def zone(value: Value, low: float | None, high: float | None) -> str:
    """Where `value` falls against the range `low`..`high` (ends included); either end may be None."""
    comparator, number = parse_value(value)
    low = -math.inf if low is None or math.isnan(low) else low
    high = math.inf if high is None or math.isnan(high) else high
    # the value as a range: "<x" is below x (x itself excluded), ">x" above it
    if comparator == "<":
        v_low, v_high = -math.inf, number
    elif comparator == ">":
        v_low, v_high = number, math.inf
    else:
        v_low, v_high = number, number
    if v_high < low or (comparator == "<" and v_high == low):
        return "below"
    if v_low > high or (comparator == ">" and v_low == high):
        return "above"
    if v_low >= low and v_high <= high:
        return "within"
    return "unclear"


@dataclass(frozen=True)
class RangeFindings:
    """`patients`: one row per report (or author) and marker with a range, giving its `zone` and
    `position` (0 at the range's low end, 1 at its high end; missing for a limit or a range with one
    end). It is working data for later analyses, one row per person, so never shown or exported
    (spec S11). `rows`: findings-table rows, one per marker, sex and group. `skipped`: markers
    with values but no range, and how many values were left out for that."""

    patients: pd.DataFrame
    rows: pd.DataFrame
    skipped: pd.DataFrame


def lab_range_findings(data: LabDataset) -> RangeFindings:
    """Place every lab-report value in its report's own reference range."""
    values = data.markers[data.markers["value"].notna()]
    report = data.records.set_index("record")[["group", "sex", "age_years"]]
    values = values.drop(columns=["sex", "age_years"]).join(report, on="record")
    has_range = values["ref_low"].notna() | values["ref_high"].notna()
    in_range = values[has_range]
    said = in_range["status"].fillna("").str.split().str[0].map({"Below": "below", "Above": "above"})
    patients = _place(in_range, printed=_as_printed(in_range), said=said)
    columns = ["record", "group", "sex", "age_years", "marker_id", "marker_name", "value", "comparator"]
    patients = patients[columns + ["ref_low", "ref_high", "zone", "position"]].reset_index(drop=True)
    skipped = _skipped(values[~has_range])
    rows = _group_rows(patients.rename(columns={"age_years": "age"}), LAB)
    return RangeFindings(patients=patients, rows=rows, skipped=skipped)


def wilson_interval(count: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval of the share `count / n`; it stays inside 0..1 for small n."""
    if n == 0:
        return math.nan, math.nan
    share = count / n
    z2 = CONFIDENCE_Z**2
    centre = (share + z2 / (2 * n)) / (1 + z2 / n)
    half = CONFIDENCE_Z * math.sqrt(share * (1 - share) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _group_rows(patients: pd.DataFrame, comparison: str) -> pd.DataFrame:
    """One findings row per marker, sex and group. Shares are of the values placed with certainty."""
    rows = []
    keys = ["marker_id", "sex", "group"]
    for (marker, sex, group), cell in patients.groupby(keys, sort=True, dropna=False):
        placed = cell[cell["zone"] != "unclear"]
        n = len(placed)
        row = {
            "comparison": comparison,
            "marker_id": marker,
            "marker_name": cell["marker_name"].iloc[0],
            "sex": sex,
            "group_a": group,
            "n_a": n,
            "n_unclear": len(cell) - n,
            "age_a": cell["age"].astype("float64").median(),
            "median_position": placed["position"].median(),
        }
        for side in SIDES:
            count = int((placed["zone"] == side).sum())
            row[f"share_{side}"] = count / n if n else math.nan
            row[f"{side}_low"], row[f"{side}_high"] = wilson_interval(count, n)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(COLUMNS))


def _as_printed(values: pd.DataFrame) -> pd.Series:
    """Each value as the report printed it: a number, or a limit such as "<5"."""
    limited = values["comparator"].notna()
    return values["value"].astype(object).where(~limited, values["comparator"] + values["value"].map(repr))


def _place(values: pd.DataFrame, printed: pd.Series, said: pd.Series) -> pd.DataFrame:
    """Add each value's `zone` and `position` within its own range.

    `said` is the source's own verdict, `below`, `above` or missing (a lab's status, an author's
    flag). It decides only for a value printed exactly at a range end, which may have been rounded.
    """
    zones = pd.Series(
        [zone(v, lo, hi) for v, lo, hi in zip(printed, values["ref_low"], values["ref_high"], strict=True)],
        index=values.index,
    )
    at_end = (values["value"] == values["ref_low"]) | (values["value"] == values["ref_high"])
    zones = zones.where(~(at_end & said.notna()), said)
    exact = printed.map(lambda v: not isinstance(v, str))
    position = (values["value"] - values["ref_low"]) / (values["ref_high"] - values["ref_low"])
    return values.assign(
        zone=zones, position=position.where(exact & (values["ref_high"] > values["ref_low"]))
    )


def _skipped(values: pd.DataFrame) -> pd.DataFrame:
    return values.groupby(["marker_id", "marker_name"], sort=True).size().reset_index(name="n_values")


def combine_findings(*tables: pd.DataFrame) -> pd.DataFrame:
    """One findings table from group comparisons and reference-range rows, told apart by the
    `comparison` column. A column one kind does not have is left empty in its rows."""
    columns = list(dict.fromkeys(c for table in tables for c in table.columns))
    present = [t for t in tables if not t.empty]
    if not present:
        return pd.DataFrame(columns=columns)
    return pd.concat(present, ignore_index=True)[columns]


def self_report_range_findings(data: SelfReportDataset) -> RangeFindings:
    """Place typed-in lab values (`labs.csv`) in the range typed with them (spec A8), per author.

    A test is named as typed, lower-cased with spaces tidied, so "Total  Testosterone" and "total
    testosterone" are one marker; units may differ, since each value is placed in its own range.
    A value typed exactly at a range end follows the author's flag (high, low) when there is one.
    An author counts once per test: when their values fall in different zones, as `unclear`.
    Groups are the authors' conditions. Rows are labelled "reference range, self-report" and are
    never pooled with lab reports.
    """
    labs = data.labs.assign(marker_id=data.labs["test"].str.lower().str.split().str.join(" "))
    has_range = labs["ref_low"].notna() | labs["ref_high"].notna()
    labs = labs.assign(marker_name=labs["marker_id"])
    skipped = _skipped(labs[~has_range])
    in_range = labs[has_range]
    said = in_range["flag"].map({"low": "below", "high": "above"})
    placed = _place(in_range, printed=in_range["value"], said=said)
    per_author = (
        placed.groupby(["author", "marker_id"], sort=True)
        .agg(
            n_values=("value", "size"),
            zone=("zone", lambda z: z.iloc[0] if z.nunique() == 1 else "unclear"),
            position=("position", "median"),
        )
        .reset_index()
    )
    authors = data.authors.set_index("author")[["condition", "sex", "age"]].rename(
        columns={"condition": "group"}
    )
    patients = per_author.join(authors, on="author")
    rows = _group_rows(patients.assign(marker_name=patients["marker_id"]), SELF_REPORT)
    return RangeFindings(patients=patients, rows=rows, skipped=skipped)
