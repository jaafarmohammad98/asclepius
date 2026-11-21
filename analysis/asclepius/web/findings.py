"""The findings view: what the page shows of a run, one section per step (spec S3, S5, S8, S11).

Every finding carries its sample sizes, its label and its Tier A result, and a plot of the groups:

- `distribution`: per side, its size and the 10th, 25th, 50th, 75th and 90th percentiles of its
  values. Percentiles only, never single values (spec S11); a side with fewer than 5 values has none.
- `shares`: per side, its size and a share, named by `caption`: of a cluster, the share that took
  a drug (subgroup links); of a drug group, the share that reports a symptom (drug profile).
- `zones`: per group, how many values fall below, within and above the reference range, or are
  unclear. Counts only, with every count from 1 to 4 hidden, and the next smallest too when the
  hidden ones would add up to 1 to 4 beside the total shown (spec S11). 0 is no one, so it shows.

Reference-range rows describe, they do not test, so their label is `DESCRIPTION`. Self-report
sections carry "self-report, no controls" (spec S10). Every value that is not a number is None,
so the view is valid JSON.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd

from asclepius.cluster import MIN_SHOWN, ClusterResult
from asclepius.dataset import LabDataset
from asclepius.profile import PROFILE_SYMPTOM
from asclepius.ratios import KIND as RATIO
from asclepius.ratios import compute_ratios
from asclepius.refrange import SIDES
from asclepius.run import NOISE, NOTHING_RELIABLE, PROVISIONAL, STRONG, SUGGESTIVE, statement
from asclepius.selfreport import LABEL as SELF_REPORT_LABEL
from asclepius.subgroups import SUBGROUP_DRUG, SUBGROUP_KINDS, SUBGROUP_SCORE, SubgroupLinks
from asclepius.syndromes import REFERENCE_RANGE

DESCRIPTION = "description, not a test"
FEW = f"fewer than {MIN_SHOWN}"  # in place of a count too small to show (spec S11)
# Percentiles of fewer values come close to giving the values back (at 5, the five percentiles
# are the five values), so a side is plotted from S4's smallest group size up (spec S11).
MIN_PLOTTED = 10
TIER_A_STATS = ("tier_a_effect_size", "tier_a_ci_low", "tier_a_ci_high")
ZONES = (*SIDES, "unclear")
PERCENTILES = {"q10": 10, "q25": 25, "median": 50, "q75": 75, "q90": 90}
RANK = {STRONG: 0, SUGGESTIVE: 1, NOISE: 2}  # strongest first
# the statistics shown, as floats (the CSV export reads them back as such)
STATISTICS = (
    "median_a",
    "median_b",
    "share_a",
    "share_b",
    "share_above",
    "share_below",
    "median_position",
    "effect_size",
    "ci_low",
    "ci_high",
    "odds_ratio",
    "odds_ratio_low",
    "odds_ratio_high",
    "p_value",
    "q_value",
    *TIER_A_STATS,
)
COUNTS = ("n_a", "n_b", "n_censored", "n_unclear", "tier_a_n_a", "tier_a_n_b")
SHOWN_COLUMNS = (
    "comparison",
    "measure",
    "marker_id",
    "marker_name",
    "sex",
    "group_a",
    "group_b",
    *COUNTS[:4],
    "median_a",
    "median_b",
    "share_a",
    "share_b",
    "share_above",
    "share_below",
    "median_position",
    "effect_size",
    "ci_low",
    "ci_high",
    "odds_ratio",
    "odds_ratio_low",
    "odds_ratio_high",
    "p_value",
    "q_value",
    "label",
    "tier_a_result",
    *COUNTS[4:],
    "tier_a_effect_size",
    "tier_a_ci_low",
    "tier_a_ci_high",
    "caveat",
    "status",
)

# a section's kind: what its step does
SUBGROUPS, COMPARISON, SYNDROMES, RANGES = "subgroups", "comparison", "syndromes", "ranges"
SELF_REPORT_RANGES, PROFILE, SYMPTOM_SUBGROUPS, INTERVENTIONS = (
    "self-report ranges",
    "drug and symptom profile",
    "symptom subgroups",
    "intervention outcomes",
)
SELF_REPORT_KINDS = (SELF_REPORT_RANGES, PROFILE, SYMPTOM_SUBGROUPS, INTERVENTIONS)


class Values:
    """The values behind a lab-report finding, for its plot: raw markers, and the ratios made once."""

    def __init__(self, data: LabDataset) -> None:
        self.data = data
        self._long: dict[bool, pd.DataFrame] = {}

    def of(self, ratio: bool) -> pd.DataFrame:
        """One row per report and marker (or ratio) with a value: `marker_id`, `value`, `group`, `sex`."""
        if ratio not in self._long:
            markers = compute_ratios(self.data).data.markers if ratio else self.data.markers
            markers = markers[markers["value"].notna()][["record", "marker_id", "value"]]
            report = self.data.records.set_index("record")[["group", "sex"]]
            self._long[ratio] = markers.join(report, on="record")
        return self._long[ratio]

    def side(self, ratio: bool, marker: str, sex: str, group: str) -> pd.Series:
        """The values of `marker` (a ratio's id when `ratio`) in one group and sex."""
        long = self.of(ratio)
        chosen = (long["marker_id"] == marker) & (long["group"] == group) & (long["sex"] == sex)
        return long.loc[chosen, "value"].astype("float64")


def distribution(group: str, values: pd.Series) -> dict[str, object]:
    """A side of a `distribution` plot: its size and percentiles, none below `MIN_PLOTTED` values
    (spec S11). A limit ("<5") counts as its number, as the medians do."""
    n = len(values)
    side: dict[str, object] = {"group": group, "n": shown(n)}
    if n < MIN_PLOTTED:
        return side | dict.fromkeys(PERCENTILES)
    points = np.percentile(values.to_numpy(), list(PERCENTILES.values()))
    return side | {name: float(p) for name, p in zip(PERCENTILES, points, strict=True)}


def shown(count: object) -> int | None:
    """`count`, or None when it is missing or below 5 (spec S11)."""
    if count is None or pd.isna(count):  # type: ignore[arg-type]
        return None
    return int(count) if int(count) >= MIN_SHOWN else None  # type: ignore[call-overload]


def shown_parts(parts: dict[str, int]) -> dict[str, int | None]:
    """`parts` of a total that is shown, each from 1 to 4 hidden, and then the next smallest too until
    the hidden ones add up to 0 or at least 5, so the total cannot give one back (spec S11)."""
    hidden = {k: 0 < n < MIN_SHOWN for k, n in parts.items()}  # 0 is no one, so it is shown
    for key in sorted(parts, key=parts.__getitem__):
        if hidden[key] or parts[key] == 0:  # hiding a 0 would hide no one
            continue
        if not 0 < sum(n for k, n in parts.items() if hidden[k]) < MIN_SHOWN:
            break
        hidden[key] = True
    return {k: None if hidden[k] else n for k, n in parts.items()}


def plain(value: object) -> object:
    """A JSON-ready value: None for anything missing, Python numbers for numpy ones."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    return value if isinstance(value, str) else str(value)


def row_view(row: pd.Series) -> dict[str, object]:
    """One findings row as the page shows it, with every count below 5 hidden."""
    out = {c: plain(row[c]) for c in SHOWN_COLUMNS if c in row.index}
    for c in COUNTS:
        if c in out:
            out[c] = shown(out[c])
    return out


def finding_view(
    row: pd.Series, values: Values | None, plot: dict[str, object] | None = None
) -> dict[str, object]:
    """One tested finding as the page shows it, with its plot (`plot`, or the lab-report one made
    from `values`). Also hides what would give back a count below 5 from the numbers shown (S11):

    - a syndrome against the reference ranges whose values within the range are 1 to 4: the size,
      the shares above and below, or the size and the effect (share above minus share below)
      would give that count back, so the shares, the effect and the plot are hidden;
    - a drug link whose shares are hidden: on yes/no values the effect is the difference of the
      shares, which with both sizes can give a count of takers back, so it is hidden too. Its Tier A
      effect always is, since the Tier A shares are not checked.
    """
    out = row_view(row)
    if row.get("group_b") == REFERENCE_RANGE and _within(row) in range(1, MIN_SHOWN):
        hidden = ("share_above", "share_below", "effect_size", "ci_low", "ci_high", *TIER_A_STATS)
        out |= dict.fromkeys(hidden)
        row = row.drop(["share_above", "share_below"])
    if row.get("comparison") == SUBGROUP_DRUG:
        out |= dict.fromkeys(TIER_A_STATS)
        if out.get("share_a") is None or out.get("share_b") is None:
            out |= dict.fromkeys(("effect_size", "ci_low", "ci_high"))
    if plot is None:
        plot = drug_shares(row) if row.get("comparison") == SUBGROUP_DRUG else lab_plot(row, values)
    return out | {"plot": plot}


def _within(row: pd.Series) -> int | None:
    """How many of a syndrome's values fall within the reference range, when its shares are known."""
    if pd.isna(row.get("share_above")) or pd.isna(row.get("share_below")) or pd.isna(row.get("n_a")):
        return None
    n = int(row["n_a"])
    return n - round(row["share_above"] * n) - round(row["share_below"] * n)


def ordered(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Strongest first, then the largest effect; rows with no label (descriptions) last."""

    def key(row: dict[str, object]) -> tuple[int, float]:
        effect = row.get("effect_size")
        return RANK.get(str(row.get("label")), len(RANK)), -abs(effect) if isinstance(effect, float) else 0.0

    return sorted(rows, key=key)


def tested_statement(table: pd.DataFrame) -> str:
    """ "nothing reliable yet" when no row of `table` is strong or suggestive (spec S8)."""
    return statement(table) if "label" in table.columns else NOTHING_RELIABLE


def refused_view(table: pd.DataFrame) -> list[dict[str, object]]:
    """Tests not made, with their sizes (below 5 hidden) and the reason."""
    columns = ["comparison", "marker_name", "sex", "group_a", "group_b", "n_a", "n_b", "reason"]
    rows = []
    for _, row in table.iterrows():
        rows.append({c: shown(row.get(c)) if c in ("n_a", "n_b") else plain(row.get(c)) for c in columns})
    return rows


# ---------------------------------------------------------------------------------- lab reports


def comparison_section(result: Any, values: Values | None) -> dict[str, object]:
    """The two groups compared: raw markers and ratios (the subgroup rows have their own section)."""
    table = result.findings[~result.findings["comparison"].isin(SUBGROUP_KINDS)]
    rows = [finding_view(row, values) for _, row in table.iterrows()]
    refused = result.refused[~result.refused["comparison"].isin(SUBGROUP_KINDS)]
    said = tested_statement(table)
    return {
        "statement": said,
        "nothing_reliable": said == NOTHING_RELIABLE,
        "labels": [f"thresholds {PROVISIONAL}"],
        "findings": ordered(rows),
        "refused": refused_view(refused),
    }


def lab_plot(row: pd.Series, values: Values | None) -> dict[str, object]:
    """The plot of a lab-report finding: the two groups' distributions, or for a syndrome against
    the reference ranges, where its values fall."""
    if row.get("group_b") == REFERENCE_RANGE:
        return shift_zones(row)
    if values is None:
        raise ValueError("A group comparison's plot needs the dataset's values.")
    ratio = RATIO in (row.get("comparison"), row.get("measure"))
    sides = [
        distribution(row[g], values.side(ratio, row["marker_id"], row["sex"], row[g]))
        for g in ("group_a", "group_b")
    ]
    return {"type": "distribution", "sides": sides}


def shift_zones(row: pd.Series) -> dict[str, object]:
    """A syndrome against the reference ranges: how many values fall below, within and above. All
    are hidden when the shares are (`compare_syndromes` hides them when above or below is 1 to 4,
    `finding_view` when within is), since the effect and the size would give any one back."""
    n = int(row["n_a"]) if pd.notna(row.get("n_a")) else 0
    counts: dict[str, int | None] = dict.fromkeys(SIDES)  # `n` counts the values placed, so no unclear
    if pd.notna(row.get("share_above")) and pd.notna(row.get("share_below")):
        above, below = round(row["share_above"] * n), round(row["share_below"] * n)
        counts = shown_parts({"below": below, "within": n - above - below, "above": above})
    return {"type": "zones", "sides": [{"group": row["group_a"], "n": shown(n), **counts}]}


def range_section(found: Any, *, self_report: bool = False) -> dict[str, object]:
    """Values placed in their own reference range: descriptions, one row per marker, sex and group,
    with counts per zone instead of shares (a share and the size would give a hidden count back)."""
    rows = []
    for _, row in found.rows.iterrows():
        n_placed = int(row["n_a"])
        n_unclear = int(row["n_unclear"]) if pd.notna(row["n_unclear"]) else 0
        counts = {side: round(row[f"share_{side}"] * n_placed) if n_placed else 0 for side in SIDES}
        counts["unclear"] = n_unclear
        n = n_placed + n_unclear
        zones = shown_parts(counts) if n >= MIN_SHOWN else dict.fromkeys(counts)
        view = {
            "comparison": plain(row["comparison"]),
            "marker_id": plain(row["marker_id"]),
            "marker_name": plain(row["marker_name"]),
            "sex": plain(row["sex"]),
            "group_a": plain(row["group_a"]),
            "n_a": shown(n),
            "median_position": plain(row["median_position"]) if n_placed >= MIN_SHOWN else None,
            "label": DESCRIPTION,
            "zones": zones,
        }
        if self_report:
            view["caveat"] = SELF_REPORT_LABEL
        view["plot"] = {"type": "zones", "sides": [{"group": view["group_a"], "n": view["n_a"], **zones}]}
        rows.append(view)
    rows.sort(key=lambda r: (str(r["marker_name"]), str(r["sex"]), str(r["group_a"])))
    placed = shown(len(found.patients))
    what = "author and test pairs" if self_report else "values"
    return {
        "statement": f"{placed if placed is not None else FEW} {what} placed in their own range",
        "nothing_reliable": None,  # descriptions: nothing here is tested
        "labels": [SELF_REPORT_LABEL] if self_report else [],
        "findings": rows,
        "refused": [],
        "note": (
            "Positions within each report's own reference range describe the values; they are "
            "not tests, so they carry no strength label."
        ),
    }


def subgroup_section(found: Any, comparison: Any, data: LabDataset) -> dict[str, object]:
    """Clusters, and whether they differ by symptom score or drug taken. The links are corrected and
    labelled with the comparison, so they are labelled only once it has run."""
    clusters: ClusterResult = found.clusters
    links: SubgroupLinks = found.links
    if comparison is not None:
        table = comparison.findings[comparison.findings["comparison"].isin(SUBGROUP_KINDS)]
        refused = comparison.refused[comparison.refused["comparison"].isin(SUBGROUP_KINDS)]
    else:  # unlabelled links are not shown (spec S5): only how many there were
        table, refused = links.rows.iloc[0:0], links.refused.iloc[0:0]
    rows = [
        finding_view(
            row, None, plot=score_plot(row, clusters, data) if row["comparison"] == SUBGROUP_SCORE else None
        )
        for _, row in table.iterrows()
    ]
    nothing: bool | None
    if clusters.k is None:
        said, nothing = clusters.statement, True
    else:
        said = tested_statement(table) if comparison is not None else f"{len(links.rows)} links not labelled"
        nothing = said == NOTHING_RELIABLE if comparison is not None else None
    out: dict[str, object] = {
        "statement": said,
        "nothing_reliable": nothing,
        "labels": [f"thresholds {PROVISIONAL}"],
        "clusters": {
            "statement": clusters.statement,
            "reason": clusters.reason,
            "sizes": cluster_sizes(clusters.sizes),
        },
        "findings": ordered(rows),
        "refused": refused_view(refused),
    }
    if comparison is None:
        out["note"] = (
            "The links to symptoms and drugs are labelled in one correction with the group "
            "comparison, which did not finish, so they are not shown."
        )
    return out


def cluster_sizes(sizes: pd.DataFrame) -> list[dict[str, object]]:
    """Patients per cluster and sex. The links show each cluster's size, so when one sex of a
    cluster is hidden (1 to 4), the cluster's others would give it back: they are hidden too."""
    out = []
    for cluster, cells in sizes.groupby("cluster", sort=True):
        counts = [shown(n) for n in cells["n_patients"]]
        if None in counts:
            counts = [None] * len(counts)
        out += [
            {"cluster": plain(cluster), "sex": plain(sex), "n": n}
            for sex, n in zip(cells["sex"], counts, strict=True)
        ]
    return out


def drug_shares(row: pd.Series) -> dict[str, object]:
    """A drug link's plot: the share of each side that took it (hidden as `subgroups` hides it)."""
    return {
        "type": "shares",
        "caption": "share who took the drug",
        "sides": [
            {"group": row["group_a"], "n": shown(row["n_a"]), "share": plain(row["share_a"])},
            {"group": row["group_b"], "n": shown(row["n_b"]), "share": plain(row["share_b"])},
        ],
    }


def score_plot(row: pd.Series, clusters: ClusterResult, data: LabDataset) -> dict[str, object]:
    """A symptom-score link's plot: the score's distribution in the cluster and in the other side
    (the next cluster, or all the others), as `link_subgroups` names them."""
    sides = [row["group_a"], row["group_b"]]
    scores = clusters.assignments.join(data.records.set_index("record")[row["marker_id"]], on="record")
    number = int(str(row["group_a"]).removeprefix("cluster "))
    in_a = scores["cluster"] == number
    if row["group_b"] == "other clusters":
        in_b = ~in_a
    else:
        in_b = scores["cluster"] == int(str(row["group_b"]).removeprefix("cluster "))
    score = scores[row["marker_id"]].astype("float64")
    return {
        "type": "distribution",
        "sides": [distribution(sides[0], score[in_a].dropna()), distribution(sides[1], score[in_b].dropna())],
    }


# ---------------------------------------------------------------------------------- self-reports


def profile_section(result: Any) -> dict[str, object]:
    """Symptoms, onset and trajectory compared between drug classes and single drugs. The rows hide
    their shares, odds ratio and effect when a cell is below 5 (`asclepius.profile`)."""
    rows = [finding_view(row, None, plot=outcome_shares(row)) for _, row in result.findings.iterrows()]
    said = tested_statement(result.findings)
    return {
        "statement": result.statement,
        "nothing_reliable": said == NOTHING_RELIABLE,
        "labels": [SELF_REPORT_LABEL, f"thresholds {PROVISIONAL}"],
        "note": (
            f"{result.note} Effect: the share of the first group with the outcome minus the share "
            "of the other, from -1 to 1, with the odds ratio beside it. Fisher's exact test."
        ),
        "groups": [
            {c: shown(v) if c == "authors" else plain(v) for c, v in r.items()}
            for _, r in result.summary.iterrows()
        ],
        "findings": ordered(rows),
        "refused": refused_view(result.refused),
    }


def outcome_shares(row: pd.Series) -> dict[str, object]:
    """A drug profile row's plot: the share of each side with the outcome (hidden with a small cell)."""
    caption = (
        "share who report it" if row["comparison"] == PROFILE_SYMPTOM else f"share: {row['marker_name']}"
    )
    return {
        "type": "shares",
        "caption": caption,
        "sides": [
            {"group": row["group_a"], "n": shown(row["n_a"]), "share": plain(row["share_a"])},
            {"group": row["group_b"], "n": shown(row["n_b"]), "share": plain(row["share_b"])},
        ],
    }


def syndrome_section(result: Any, values: Values | None) -> dict[str, object]:
    """PSSD, PFS and PAS against the reference and each other, and the pattern per marker."""
    rows = [finding_view(row, values) for _, row in result.findings.iterrows()]
    said = tested_statement(result.findings)
    summary = [{c: plain(v) for c, v in r.items()} for _, r in result.summary.iterrows()]
    syndromes = [c.removeprefix("direction_") for c in result.summary.columns if c.startswith("direction_")]
    return {
        "statement": result.statement,
        "nothing_reliable": said == NOTHING_RELIABLE,
        "labels": [f"thresholds {PROVISIONAL}"],
        "reference": result.record.get("reference"),
        "syndromes": syndromes,
        "summary": summary,
        "findings": ordered(rows),
        "refused": refused_view(result.refused),
    }


# ---------------------------------------------------------------------------------- the view


def section(
    kind: str, name: str, state: str, message: str, body: dict[str, object] | None
) -> dict[str, object]:
    """A step's section: its state and message, its labels, and `body` once it has finished."""
    out: dict[str, object] = {
        "kind": kind,
        "title": name,
        "state": state,
        "message": message,
        "labels": [SELF_REPORT_LABEL] if kind in SELF_REPORT_KINDS else [],
        "findings": [],
    }
    return out | (body or {})


# a finished step's section body, by kind: (its result, every result by kind, the data, its values)
BUILDERS: dict[str, Callable[[Any, dict[str, Any], Any, Values | None], dict[str, object]]] = {
    COMPARISON: lambda result, _, __, values: comparison_section(result, values),
    SUBGROUPS: lambda result, results, data, _: subgroup_section(result, results.get(COMPARISON), data),
    SYNDROMES: lambda result, _, __, values: syndrome_section(result, values),
    RANGES: lambda result, *_: range_section(result),
    SELF_REPORT_RANGES: lambda result, *_: range_section(result, self_report=True),
    PROFILE: lambda result, *_: profile_section(result),
}


def findings_view(data: Any, steps: Iterable[Any], results: dict[str, Any]) -> dict[str, object]:
    """Every step's section, in run order, and whether nothing reliable was found in any (spec S8).
    `steps` have a `name`, `kind`, `state` and `message`; `results` maps a step's name to what it
    returned."""
    steps = list(steps)
    by_kind = {s.kind: results[s.name] for s in steps if s.name in results}
    values = Values(data) if isinstance(data, LabDataset) else None
    sections = []
    for step in steps:
        body = None
        if step.kind in by_kind:
            body = BUILDERS[step.kind](by_kind[step.kind], by_kind, data, values)
        sections.append(section(step.kind, step.name, step.state, step.message, body))
    tested = [s for s in sections if s.get("nothing_reliable") is not None and s["state"] == "done"]
    nothing = bool(tested) and all(s["nothing_reliable"] for s in tested)
    out: dict[str, object] = {
        "kind": getattr(data, "kind", None),
        "sections": sections,
        "nothing_reliable": nothing,
        "statement": NOTHING_RELIABLE if nothing else None,
    }
    if data is not None and getattr(data, "label", None):
        out["label"] = data.label
    return out
