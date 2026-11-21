"""Intervention outcomes as ranked questions (spec A7, S3, S4, S9 to S11).

Patients have tried many things and reported the results. Ranked honestly, that tells researchers
which interventions are worth a proper study; it does not tell anyone what works.
`rank_interventions` counts, per intervention, the authors reporting each outcome:

- The loader merges posts per author (spec S9): an author who reported the same intervention
  twice counts once, with their latest clear outcome (`asclepius.selfreport`).
- `unclear` outcomes are left out of the shares and counted on their own.
- The share improved (major plus partial improvement) and the share worse, of the authors with a
  clear outcome, each with a 95% Wilson interval (spec S3). An intervention with fewer than
  `min_group_size` (10) authors with a clear outcome is refused (spec S4).
- Interventions are ranked by the lower bound of the share improved. Separately, `safety` lists
  those whose share worse has a lower bound of at least `worse_signal` (provisional, M3-11).
- Every table carries "reported outcome, not evidence that it works" and "self-report, no
  controls"; nothing is worded as advice (overview rule 5).

No count below 5 is shown (spec S11), and 0, which is no one, is. A share with its interval gives
its counts back, so a share is shown only when every count it implies is 0 or at least 5, and the
outcome counts are then hidden where those shown shares leave them unknown, as
`asclepius.cluster.shown_parts` hides parts of a total. A row whose share improved is hidden is not
ranked. `n_authors` (with unclear) and `unclear` are shown only when `unclear` is 0 or 5 or more.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from asclepius.cluster import MIN_SHOWN, count_text, shown_parts
from asclepius.record import run_record
from asclepius.refrange import wilson_interval
from asclepius.run import RunResult, Settings
from asclepius.selfreport import LABEL as SELF_REPORT_LABEL
from asclepius.selfreport import SelfReportDataset

ANALYSIS = "intervention outcomes"  # the run record's `analysis`
COMPARISON = "intervention outcome"  # the findings table's `comparison` column
REPORTED = "reported outcome, not evidence that it works"
CAVEAT = f"{REPORTED}; {SELF_REPORT_LABEL}"
WORSE_SIGNAL = 0.2  # provisional, for review in M3-11
IMPROVED = ("major_improvement", "partial_improvement")
CLEAR = (*IMPROVED, "no_effect", "worse")
COLUMNS = (
    "comparison",
    "marker_id",
    "marker_name",
    "rank",
    "n_authors",
    "n_clear",
    *CLEAR,
    "unclear",
    "share_improved",
    "improved_low",
    "improved_high",
    "share_worse",
    "worse_low",
    "worse_high",
    "safety_signal",
    "caveat",
)
REFUSED_COLUMNS = ("comparison", "marker_id", "marker_name", "n_clear", "reason", "caveat")


@dataclass(frozen=True)
class InterventionResult(RunResult):
    """What `rank_interventions` found.

    `findings`: one row per intervention with enough authors (`COLUMNS`), ranked: `rank` 1 has the
    highest lower bound of the share improved, and rows whose share improved is hidden have none
    and come last. `safety`: the rows whose share worse has a lower bound of at least
    `worse_signal`, highest first. `refused`: interventions with too few authors, with the
    `reason`. `skipped`: empty; there for `asclepius.record.export_run`. `statement`: how many were
    ranked and flagged. Every table has `caveat`. `settings`, `record`: as for `run_analysis`
    (spec S7).
    """

    safety: pd.DataFrame


def rank_interventions(
    data: SelfReportDataset, settings: Settings | None = None, *, worse_signal: float = WORSE_SIGNAL
) -> InterventionResult:
    """Rank interventions by the share of authors reporting improvement, and flag those with a high
    share reporting worse (spec A7). The same data and settings give the same result."""
    settings = settings or Settings()
    if not 0 <= worse_signal <= 1:
        raise ValueError("worse_signal must be between 0 and 1.")
    counts = pd.crosstab(data.interventions["intervention"], data.interventions["outcome"].astype(object))
    counts = counts.reindex(columns=[*CLEAR, "unclear"], fill_value=0)
    rows: list[dict[str, object]] = []
    refused: list[dict[str, object]] = []
    for intervention, c in counts.iterrows():
        head: dict[str, object] = {
            "comparison": COMPARISON,
            "marker_id": f"intervention:{intervention}",
            "marker_name": str(intervention).replace("_", " "),
        }
        clear = int(c[list(CLEAR)].sum())
        if clear < settings.min_group_size:
            refused.append(
                head
                | {
                    "n_clear": _count(clear),
                    "reason": f"fewer than {settings.min_group_size} authors with a clear outcome (spec S4)",
                }
            )
            continue
        rows.append(head | _row({k: int(c[k]) for k in (*CLEAR, "unclear")}, worse_signal))

    findings = pd.DataFrame(rows, columns=[c for c in COLUMNS if c not in ("rank", "caveat")])
    ranked = findings["improved_low"].notna()
    order = findings.assign(_ranked=ranked).sort_values(
        ["_ranked", "improved_low", "share_improved", "marker_name"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    findings = order.drop(columns="_ranked").reset_index(drop=True)
    findings.insert(
        3,
        "rank",
        pd.array(
            [i + 1 if r else None for i, r in enumerate(findings["improved_low"].notna())], dtype="Int64"
        ),
    )
    findings["caveat"] = CAVEAT
    findings = findings.astype({c: "Int64" for c in ("n_authors", "n_clear", *CLEAR, "unclear")})
    safety = (
        findings[findings["safety_signal"]]
        .sort_values(["worse_low", "marker_name"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    refused_table = pd.DataFrame(refused, columns=[c for c in REFUSED_COLUMNS if c != "caveat"])
    refused_table = refused_table.astype({"n_clear": "Int64"}).assign(caveat=CAVEAT)

    recorded = settings.record() | {"worse_signal": worse_signal}
    n_ranked = int(findings["rank"].notna().sum())
    statement = (
        f"{count_text(n_ranked)} interventions ranked by the share of authors reporting improvement, "
        f"{count_text(len(safety))} with a high share reporting worse; {REPORTED}"
    )
    return InterventionResult(
        findings=findings,
        refused=refused_table,
        skipped=pd.DataFrame(columns=["level", "group", "reason"]),
        safety=safety,
        statement=statement,
        settings=recorded,
        record=run_record(data, [], recorded) | {"analysis": ANALYSIS, "note": REPORTED},
    )


def _row(c: dict[str, int], worse_signal: float) -> dict[str, object]:
    """One intervention's counts, shares and intervals, hidden as the module docstring says."""
    improved = c["major_improvement"] + c["partial_improvement"]
    clear = improved + c["no_effect"] + c["worse"]
    show_improved = _ok(improved) and _ok(clear - improved)
    show_worse = _ok(c["worse"]) and _ok(clear - c["worse"])
    if show_improved and show_worse and not _ok(c["no_effect"]):
        show_worse = False  # both shares would give no_effect back; the ranking keeps its share
    # groups of outcomes whose total is known, from n_clear and the shares shown
    if show_improved and show_worse:
        groups = [list(IMPROVED), ["no_effect"], ["worse"]]
    elif show_improved:
        groups = [list(IMPROVED), ["no_effect", "worse"]]
    elif show_worse:
        groups = [[*IMPROVED, "no_effect"], ["worse"]]
    else:
        groups = [list(CLEAR)]
    shown: dict[str, int | None] = {}
    for group in groups:
        shown |= shown_parts({k: c[k] for k in group})
    out: dict[str, object] = {"n_clear": clear, **shown}
    unclear_ok = _ok(c["unclear"])
    out["unclear"] = c["unclear"] if unclear_ok else None
    out["n_authors"] = clear + c["unclear"] if unclear_ok else None
    nan = float("nan")
    low, high = wilson_interval(improved, clear)
    out |= {"share_improved": improved / clear, "improved_low": low, "improved_high": high}
    if not show_improved:
        out |= {"share_improved": nan, "improved_low": nan, "improved_high": nan}
    low, high = wilson_interval(c["worse"], clear)
    out |= {"share_worse": c["worse"] / clear, "worse_low": low, "worse_high": high}
    if not show_worse:
        out |= {"share_worse": nan, "worse_low": nan, "worse_high": nan}
    out["safety_signal"] = bool(show_worse and low >= worse_signal)
    return out


def _ok(count: int) -> bool:
    """A count that may be shown or given back: 0 (no one) or at least 5 (spec S11)."""
    return count == 0 or count >= MIN_SHOWN


def _count(count: int) -> int | None:
    return count if _ok(count) else None
