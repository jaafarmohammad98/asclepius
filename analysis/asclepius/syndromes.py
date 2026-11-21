"""Cross-syndrome comparison: do PSSD, PFS and PAS shift the same markers the same way? (spec A4)

The idea under test is that the three syndromes share a mechanism. If they do, a marker or ratio
that is off in one should be off the same way in the others. `compare_syndromes` makes two kinds of
test on every raw marker and enzyme-activity ratio, within sex:

- **Each syndrome against the reference**: a control group (by default the healthy controls), with
  the same test as any group comparison. This gives each syndrome's direction.
- **Each pair of syndromes against each other**, the same way. This says where they differ.

Every test is corrected together, in one false discovery rate correction, re-run on Tier A records
and labelled exactly as in `run_analysis` (spec S2, S4 to S6). `summary` then reads them per marker
and sex: which syndromes are shifted which way, which pairs differ, and the pattern that makes.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from asclepius.cluster import MIN_SHOWN
from asclepius.compare import CONFIDENCE, SEXES
from asclepius.dataset import LabDataset
from asclepius.metadata import GROUPS
from asclepius.ratios import KIND as RATIO
from asclepius.record import run_record
from asclepius.refrange import combine_findings, lab_range_findings
from asclepius.run import (
    NOISE,
    STRONG,
    SUGGESTIVE,
    TIER_A_NAME,
    GroupTooSmall,
    RunResult,
    Settings,
    group_tests,
    judge_tier_a,
    label_findings,
    refuse_small,
    statement,
    tier_a_rerun,
)

SYNDROMES = ("PSSD", "PFS", "PAS")
CONTROLS = tuple(g for g in GROUPS if g not in SYNDROMES)
REFERENCE_RANGE = "reference range"  # the reference when there are no controls
ANALYSIS = "cross-syndrome"  # the run record's `analysis`
VS_REFERENCE = "cross-syndrome: vs reference"  # the findings table's `comparison` column
BETWEEN = "cross-syndrome: between syndromes"
RAW = "raw marker"  # the `measure` column: a raw marker or an enzyme-activity ratio
HIGHER, LOWER, NO_SHIFT, NOT_TESTED = "higher", "lower", "no clear shift", "not tested"
SHARED = "shared"
SAME_WAY_DIFFERENT_SIZE = "same direction, different size"
DIFFERS = "differs"
ONE_ONLY = "one syndrome only"
_RANK = {NOISE: 0, SUGGESTIVE: 1, STRONG: 2}
_FOUND = (STRONG, SUGGESTIVE)


@dataclass(frozen=True)
class SyndromeResult(RunResult):
    """A `RunResult` (findings, refused, skipped, statement, settings, record) plus `summary`: one
    row per measure, marker and sex, reading the findings across syndromes.

    `summary` columns: `measure`, `marker_id`, `marker_name`, `sex`; per syndrome `n_<s>` (its
    values; hidden below 5, spec S11), `effect_<s>` (Cliff's delta against the reference) and
    `direction_<s>` (`higher`, `lower`, `no clear shift`, `not tested`); `n_reference`; `differ`,
    the pairs that differ (`PFS lower than PSSD`); `pattern`; `shared_by`; and `label`, how strong
    the pattern is: the weakest label among the findings that make it.
    """

    summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def compare_syndromes(
    data: LabDataset, reference: str = "healthy control", settings: Settings | None = None
) -> SyndromeResult:
    """Compare PSSD, PFS and PAS with `reference` and with each other, on every marker and ratio.

    The syndromes compared are those with reports in `data`. One with fewer than
    `settings.min_group_size` reports is left out and listed in `refused` (spec S4); fewer than two
    left is refused with `GroupTooSmall`, as is a reference group that small. Within a sex, a test
    where either side has too few values is refused as in `run_analysis`.
    """
    settings = settings or Settings()
    if reference not in (*CONTROLS, REFERENCE_RANGE):
        raise ValueError(
            f"The reference must be a control group ({', '.join(CONTROLS)}) or {REFERENCE_RANGE!r}."
        )
    sizes = data.records["group"].value_counts()
    if reference != REFERENCE_RANGE:
        n_reference = int(sizes.get(reference, 0))
        if n_reference == 0:
            raise ValueError(
                f"No loaded report is in the reference group {reference!r}; "
                f"with no controls, use reference={REFERENCE_RANGE!r}."
            )
        if n_reference < settings.min_group_size:
            raise GroupTooSmall(_too_small(reference, n_reference, settings))
    present = [s for s in SYNDROMES if sizes.get(s, 0) > 0]
    syndromes = [s for s in present if sizes[s] >= settings.min_group_size]
    reasons = {s: _too_small(s, int(sizes[s]), settings) for s in present if s not in syndromes}
    left_out = [
        {"comparison": ANALYSIS, "group_a": s, "n_a": _shown(int(sizes[s])), "reason": reason}
        for s, reason in reasons.items()
    ]
    if len(syndromes) < 2:
        raise GroupTooSmall(
            "Refused: a cross-syndrome comparison needs at least two of "
            f"{', '.join(SYNDROMES)} with {settings.min_group_size} or more reports each"
            + (f" ({'; '.join(reasons.values())})." if reasons else ".")
        )

    pairs = [] if reference == REFERENCE_RANGE else [(s, reference, VS_REFERENCE) for s in syndromes]
    pairs += [(a, b, BETWEEN) for a, b in combinations(syndromes, 2)]
    tested, refused, skipped = [], [pd.DataFrame(left_out)], pd.DataFrame()
    if reference == REFERENCE_RANGE:
        ok, small = range_tests(data, syndromes, settings)
        tested.append(ok)
        refused.append(small)
    for group_a, group_b, kind in pairs:
        table, skipped = group_tests(data, group_a, group_b, settings)
        ok, small = refuse_small(table, settings)
        ok = tier_a_rerun(ok, data, group_a, group_b, settings)
        tested.append(_as_kind(ok, kind))
        refused.append(_as_kind(small, kind))
    findings = label_findings(combine_findings(*tested), settings)
    refused_table = combine_findings(*refused)
    summary = summarise(findings, refused_table, syndromes)
    record = run_record(data, [*syndromes, reference], settings.record())
    record |= {"analysis": ANALYSIS, "reference": reference}
    return SyndromeResult(
        findings=findings,
        refused=refused_table,
        skipped=skipped,
        statement=_statement(findings, summary),
        settings=settings.record(),
        record=record,
        summary=summary,
    )


def range_tests(
    data: LabDataset, syndromes: list[str], settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each syndrome against the reference ranges, per raw marker and sex, re-run on Tier A.

    Of the values placed with certainty (`asclepius.refrange`), the effect is the share above the
    range minus the share below it: 0 when as many fall on each side, as in healthy people, whose
    range is about the middle 95%. The p-value is an exact two-sided sign test of above against
    below; the interval a percentile bootstrap. Fewer than `min_group_size` values is refused
    (spec S4). Returns (tested rows, refused rows).
    """
    patients = lab_range_findings(data).patients
    patients = patients[patients["group"].isin(syndromes) & patients["sex"].isin(SEXES)]
    tiers = data.records.set_index("record")["tier"]
    patients = patients.assign(tier=patients["record"].map(tiers))
    rows, refused = [], []
    for (marker, sex, group), cell in patients.groupby(["marker_id", "sex", "group"], sort=True):
        placed = cell[cell["zone"] != "unclear"]
        head = {
            "comparison": VS_REFERENCE,
            "measure": RAW,
            "marker_id": marker,
            "marker_name": cell["marker_name"].iloc[0],
            "sex": sex,
            "group_a": group,
            "group_b": REFERENCE_RANGE,
            "n_a": len(placed),
            "n_unclear": len(cell) - len(placed),
        }
        if len(placed) < settings.min_group_size:
            reason = f"fewer than {settings.min_group_size} values placed in a range in this sex (spec S4)"
            refused.append(head | {"reason": reason})
            continue
        # seeded per marker, sex and group, as group comparisons are
        seed = [
            settings.seed,
            zlib.crc32(str(marker).encode()),
            SEXES.index(str(sex)),
            SYNDROMES.index(group),
        ]
        full = range_shift(placed["zone"], settings, np.random.default_rng(seed))
        tier_a = placed[placed["tier"] == TIER_A_NAME]["zone"]
        rerun = range_shift(tier_a, settings, np.random.default_rng(seed)) if len(tier_a) else {"n": 0}
        above, below = int((placed["zone"] == "above").sum()), int((placed["zone"] == "below").sum())
        shares = {"share_above": above / len(placed), "share_below": below / len(placed)}
        if min(above, below) < MIN_SHOWN:  # with the effect, one share gives the other back (S11)
            shares = {"share_above": float("nan"), "share_below": float("nan")}
        rows.append(
            head
            | shares
            | {k: full[k] for k in ("effect_size", "ci_low", "ci_high", "p_value")}
            | {"tier_a_n_a": rerun["n"]}
            | {
                f"tier_a_{k}": rerun.get(k, float("nan"))
                for k in ("effect_size", "ci_low", "ci_high", "p_value")
            }
        )
    tested = judge_tier_a(pd.DataFrame(rows), settings) if rows else pd.DataFrame(columns=list(rows))
    return tested, pd.DataFrame(refused)


def range_shift(zones: pd.Series, settings: Settings, rng: np.random.Generator) -> dict[str, float]:
    """`n`, `effect_size` (share above minus share below), its bootstrap `ci_low`..`ci_high` and
    the exact two-sided sign test's `p_value` for one group's zones (`below`, `within`, `above`)."""
    n = len(zones)
    above, below = int((zones == "above").sum()), int((zones == "below").sum())
    outside = above + below
    tail = sum(math.comb(outside, k) for k in range(min(above, below) + 1)) / 2**outside
    draws = rng.multinomial(n, [above / n, below / n, (n - outside) / n], size=settings.bootstrap)
    effects = (draws[:, 0] - draws[:, 1]) / n
    low, high = np.percentile(effects, [(1 - CONFIDENCE) / 2 * 100, (1 + CONFIDENCE) / 2 * 100])
    return {
        "n": n,
        "effect_size": (above - below) / n,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": min(1.0, 2 * tail),
    }


def _as_kind(table: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Mark rows of `group_tests` as `kind`, keeping whether each is a raw marker or a ratio."""
    measure = table["comparison"].where(table["comparison"] == RATIO, RAW)
    return table.assign(comparison=kind).assign(measure=measure)


def summarise(findings: pd.DataFrame, refused: pd.DataFrame, syndromes: list[str]) -> pd.DataFrame:
    """One row per measure, marker and sex: each syndrome's direction against the reference, the
    pairs that differ, and the pattern (see `SyndromeResult`)."""
    keys = ["measure", "marker_id", "marker_name", "sex"]
    rows = []
    every = combine_findings(findings, refused[refused["comparison"] != ANALYSIS])
    for (measure, marker, name, sex), cell in every.groupby(keys, sort=True):
        row: dict[str, object] = {"measure": measure, "marker_id": marker, "marker_name": name, "sex": sex}
        labels: dict[str, str] = {}
        up: list[str] = []
        down: list[str] = []
        vs = cell[cell["comparison"] == VS_REFERENCE].set_index("group_a")
        for s in syndromes:
            if s in vs.index:
                test = vs.loc[s]
                row[f"n_{s}"] = _shown(int(test["n_a"]))
                tested = pd.notna(test.get("p_value"))
                row[f"effect_{s}"] = test["effect_size"] if tested else float("nan")
                direction = NO_SHIFT if tested else NOT_TESTED
                if tested and test["label"] in _FOUND:
                    direction = HIGHER if test["effect_size"] > 0 else LOWER
                    labels[s] = test["label"]
                    (up if direction == HIGHER else down).append(s)
            else:
                row[f"n_{s}"], row[f"effect_{s}"], direction = None, float("nan"), NOT_TESTED
            row[f"direction_{s}"] = direction
        n_reference = vs["n_b"].max() if "n_b" in vs.columns else float("nan")
        row["n_reference"] = None if pd.isna(n_reference) else _shown(int(n_reference))

        differ, differ_labels = [], {}
        between = cell[(cell["comparison"] == BETWEEN) & cell["label"].isin(_FOUND)]
        for _, test in between.iterrows():
            a, b = test["group_a"], test["group_b"]
            differ.append(f"{a} {'higher' if test['effect_size'] > 0 else 'lower'} than {b}")
            differ_labels[(a, b)] = test["label"]
        row["differ"] = "; ".join(differ)

        sharers = up if len(up) >= 2 and not down else down if len(down) >= 2 and not up else []
        pattern, label = _pattern(row, syndromes, up, down, sharers, labels, differ_labels)
        row |= {"pattern": pattern, "shared_by": ", ".join(sharers), "label": label}
        rows.append(row)
    columns = [*keys]
    columns += [f"{c}_{s}" for c in ("n", "effect", "direction") for s in syndromes]
    columns += ["n_reference", "differ", "pattern", "shared_by", "label"]
    return pd.DataFrame(rows, columns=columns)


def _pattern(
    row: dict[str, object],
    syndromes: list[str],
    up: list[str],
    down: list[str],
    sharers: list[str],
    labels: dict[str, str],
    differ_labels: dict[tuple[str, str], str],
) -> tuple[str, str | None]:
    """The pattern and its label: the weakest label among the findings that make it."""
    if up and down:  # shifted opposite ways
        return DIFFERS, _strongest(
            [_strongest([labels[s] for s in up]), _strongest([labels[s] for s in down])], weakest=True
        )
    if sharers:
        among = [lab for (a, b), lab in differ_labels.items() if a in sharers and b in sharers]
        shared = _strongest([labels[s] for s in sharers], weakest=True)
        return (SAME_WAY_DIFFERENT_SIZE, shared) if among else (SHARED, shared)
    if differ_labels:
        return DIFFERS, _strongest(list(differ_labels.values()))
    if up or down:
        return ONE_ONLY, labels[(up or down)[0]]
    if all(row[f"direction_{s}"] == NOT_TESTED for s in syndromes):
        return NOT_TESTED, None
    return NO_SHIFT, None


def _strongest(labels: list[str], *, weakest: bool = False) -> str:
    return (min if weakest else max)(labels, key=_RANK.__getitem__)


def _statement(findings: pd.DataFrame, summary: pd.DataFrame) -> str:
    """The run's statement (spec S8), then how many marker-sex rows are shared or differ."""
    said = statement(findings)
    counts = summary["pattern"].value_counts()
    parts = [f"{int(counts[p])} {p}" for p in (SHARED, SAME_WAY_DIFFERENT_SIZE, DIFFERS) if p in counts]
    return said + (f"; across syndromes: {', '.join(parts)}" if parts else "")


def _too_small(group: str, n: int, settings: Settings) -> str:
    return (
        f"the group {group!r} has {n if n >= MIN_SHOWN else f'fewer than {MIN_SHOWN}'} reports, "
        f"and a comparison needs at least {settings.min_group_size} (spec S4, {settings.status})"
    )


def _shown(count: int) -> int | None:
    return count if count >= MIN_SHOWN else None
