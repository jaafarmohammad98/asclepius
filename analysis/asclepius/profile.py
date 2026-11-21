"""Drug and symptom profile of self-reports (spec A6, S1 to S3, S9 to S11).

Different drugs may leave different symptom patterns; if they do, that points at mechanisms worth
testing. `profile_drugs` compares, for each symptom, how often authors report it between drug
classes and between the most common single drugs, and does the same for onset and trajectory:

- Classes (`DRUG_CLASSES`, a documented table): each class against the authors in the other
  classes. An author with drugs in several classes counts in each.
- Single drugs: each of the `top_drugs` most common named drugs against the authors who took
  another causal drug. A drug that alone makes up its class here is left out: its class row is the
  same test.
- Every row is a yes/no outcome per author: reports the symptom or not; onset during use (of those
  whose onset is during use or after stopping); each trajectory (of those with a clear one).
  Fisher's exact test (spec S1), the difference in shares with Newcombe's interval as the effect
  size and the odds ratio with Woolf's interval (spec S3), and each side's number of authors.
- Every row of the run is corrected together and labelled by `asclepius.run.label_findings` with
  `self_report=True`: "self-report, no controls", never strong (spec S2, S5, S10).
- No count below 5 is shown (spec S11): sizes below 5 are hidden, and a row with a cell below 5
  (reporters or non-reporters on a side) shows only its sizes and label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import pandas as pd

from asclepius.cluster import MIN_SHOWN
from asclepius.record import run_record
from asclepius.refrange import wilson_interval
from asclepius.run import NOTHING_RELIABLE, RunResult, Settings, label_findings, statement
from asclepius.selfreport import SelfReportDataset

ANALYSIS = "drug and symptom profile"  # the run record's `analysis`
PROFILE_SYMPTOM = "drug profile: symptom"  # the findings table's `comparison` column
PROFILE_ONSET = "drug profile: onset"
PROFILE_TRAJECTORY = "drug profile: trajectory"
CLASS, DRUG = "class", "drug"  # the `level` column
OTHER_CLASSES, OTHER_DRUGS = "other drug classes", "other causal drugs"
TRAJECTORIES = ("recovered", "improving", "stable", "worsening")
ONSET_DURING, ONSET_AFTER = "during_use", "after_stopping"
TOP_DRUGS = 8
NOTE = "An author with drugs in several classes counts in each; see the summary for how many."
COLUMNS = (
    "comparison",
    "marker_id",
    "marker_name",
    "sex",
    "level",
    "group_a",
    "group_b",
    "n_a",
    "n_b",
    "reported_a",
    "reported_b",
    "share_a",
    "share_b",
    "effect_size",
    "ci_low",
    "ci_high",
    "odds_ratio",
    "odds_ratio_low",
    "odds_ratio_high",
    "p_value",
)
# what a row with a cell below 5 hides: each gives the cells back, with the sizes (the p-value
# with the total of reporters another row shows)
CELL_COLUMNS = (
    "p_value",
    "q_value",
    "reported_a",
    "reported_b",
    "share_a",
    "share_b",
    "effect_size",
    "ci_low",
    "ci_high",
    "odds_ratio",
    "odds_ratio_low",
    "odds_ratio_high",
)
REFUSED_COLUMNS = (
    "comparison",
    "marker_id",
    "marker_name",
    "level",
    "group_a",
    "group_b",
    "n_a",
    "n_b",
    "reason",
)
SUMMARY_COLUMNS = ("level", "group", "authors", "tested")

CONFIDENCE = 0.95
_Z = NormalDist().inv_cdf(0.5 + CONFIDENCE / 2)

CLASSES = ("SSRI", "SNRI", "tricyclic", "other antidepressant", "finasteride", "isotretinoin")
# causal drug, as the harvester names it -> its class. A drug not listed is in no class: it is
# left out of the class comparisons. An unspecified antidepressant could be any class, so it is
# not listed. Misspellings and brand names are listed where the generic is clear.
_LISTED = {
    "SSRI": (
        "unspecified_ssri", "sertraline", "lustral", "escitalopram", "fluoxetine", "paroxetine",
        "paxetine", "citalopram", "citralopram", "fluvoxamine", "dumirox", "dapoxetine",
    ),
    "SNRI": (
        "unspecified_snri", "venlafaxine", "desvenlafaxine", "prestiq", "duloxetine", "milnacipran",
        "levomilnacipran",
    ),
    "tricyclic": (
        "tca", "tricyclic_antidepressant", "unspecified_tricyclic", "unspecified_tricyclic_antidepressant",
        "clomipramine", "amitriptyline", "laroxyl", "nortriptyline", "imipramine", "desipramine",
        "doxepin", "dosulepin", "melitracen",
    ),
    "other antidepressant": (
        "bupropion", "vortioxetine", "vilazodone", "viibryd", "mirtazapine", "mirtzapine", "trazodone",
        "nefazodone", "mianserin", "agomelatine", "moclobemide", "phenelzine", "tianeptine",
    ),
    "finasteride": ("finasteride",),
    "isotretinoin": ("isotretinoin",),
}  # fmt: skip
DRUG_CLASSES = {drug: cls for cls, drugs in _LISTED.items() for drug in drugs}
# names of a class, not a drug, besides `unspecified_...`: not compared as single drugs
UNSPECIFIED = frozenset({"tca", "tricyclic_antidepressant"})


def author_profiles(data: SelfReportDataset) -> pd.DataFrame:
    """One row per author (index `author`): `drugs` and `classes` (sorted tuples; an author with
    drugs in several classes has each), `onset`, `overall_trajectory`, and one True/False column per
    symptom reported by anyone. Working data: one row per person, so never shown (spec S11)."""
    authors = data.authors.set_index("author")[["onset", "overall_trajectory"]]
    drugs = data.causal_drugs.groupby("author")["drug"].agg(lambda d: tuple(sorted(set(d))))
    drugs = drugs.reindex(authors.index).map(lambda d: d if isinstance(d, tuple) else ())
    classes = drugs.map(lambda d: tuple(c for c in CLASSES if c in {DRUG_CLASSES.get(x) for x in d}))
    reported = pd.crosstab(data.symptoms["author"], data.symptoms["symptom"]).gt(0)
    reported = reported.reindex(authors.index, fill_value=False).astype(bool)
    reported.columns.name = None
    return pd.concat([authors.assign(drugs=drugs, classes=classes), reported], axis=1)


@dataclass(frozen=True)
class ProfileResult(RunResult):
    """What `profile_drugs` found.

    `findings`: one row per outcome and group (`COLUMNS`), with `q_value`, `label` and `caveat`.
    `refused`: groups or outcomes not tested, with the `reason`. `summary`: authors per class and
    drug, and how many have drugs in several classes, in no class, or none recorded (below 5
    hidden). `skipped`: empty; there for `asclepius.record.export_run`. `note`: that an author can
    count in several classes. `statement`: "nothing reliable yet" when no row is suggestive (spec
    S8). `settings`, `record`: as for `run_analysis` (spec S7).
    """

    summary: pd.DataFrame
    note: str


def profile_drugs(
    data: SelfReportDataset, settings: Settings | None = None, *, top_drugs: int = TOP_DRUGS
) -> ProfileResult:
    """Compare symptoms, onset and trajectory between drug classes and the `top_drugs` most common
    single drugs (spec A6). A group, or a side of one outcome, with fewer than
    `settings.min_group_size` authors is refused (spec S4), as is a symptom fewer than that many
    authors report. The same data and settings give the same result."""
    settings = settings or Settings()
    if top_drugs < 0:
        raise ValueError("top_drugs cannot be negative.")
    profiles = author_profiles(data)
    symptoms = [c for c in profiles.columns if c not in ("onset", "overall_trajectory", "drugs", "classes")]
    has_class = profiles[profiles["classes"].map(bool)]
    has_drug = profiles[profiles["drugs"].map(bool)]
    minimum = settings.min_group_size

    groups: list[tuple[str, str, str, pd.DataFrame, pd.Series]] = []  # level, name, rest, authors, in a
    for cls in CLASSES:
        groups.append(
            (CLASS, cls, OTHER_CLASSES, has_class, has_class["classes"].map(lambda c, x=cls: x in c))
        )
    for drug in _single_drugs(has_drug, top_drugs):
        groups.append((DRUG, drug, OTHER_DRUGS, has_drug, has_drug["drugs"].map(lambda d, x=drug: x in d)))

    outcomes: list[tuple[str, str, str, pd.Series]] = []  # comparison, marker_id, name, yes/no or NA
    refused: list[dict[str, object]] = []
    for symptom in symptoms:
        reporters = int(profiles[symptom].sum())
        if reporters < minimum:
            refused.append(
                {
                    "comparison": PROFILE_SYMPTOM,
                    "marker_id": f"symptom:{symptom}",
                    "marker_name": _name(symptom),
                    "n_a": _shown(reporters),
                    "reason": f"fewer than {minimum} authors report it (spec S4)",
                }
            )
            continue
        outcomes.append(
            (PROFILE_SYMPTOM, f"symptom:{symptom}", _name(symptom), profiles[symptom].astype(float))
        )
    onset = profiles["onset"].astype(object)
    clear = onset.isin([ONSET_DURING, ONSET_AFTER])
    outcomes.append(
        (
            PROFILE_ONSET,
            f"onset:{ONSET_DURING}",
            "onset during use",
            (onset == ONSET_DURING).astype(float).where(clear),
        )
    )
    trajectory = profiles["overall_trajectory"].astype(object)
    clear = trajectory.isin(TRAJECTORIES)
    for value in TRAJECTORIES:
        outcomes.append(
            (
                PROFILE_TRAJECTORY,
                f"trajectory:{value}",
                value,
                (trajectory == value).astype(float).where(clear),
            )
        )

    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for level, name, rest, authors, in_a in groups:
        size, others = int(in_a.sum()), int((~in_a).sum())
        tested = min(size, others) >= minimum
        summary.append({"level": level, "group": name, "authors": _shown(size), "tested": tested})
        head = {"level": level, "group_a": name, "group_b": rest}
        if not tested:
            if size:
                refused.append(
                    head
                    | {
                        "n_a": _shown(size),
                        "n_b": _shown(others),
                        "reason": f"fewer than {minimum} authors in a group (spec S4)",
                    }
                )
            continue
        for comparison, marker_id, marker_name, values in outcomes:
            values = values.reindex(authors.index)
            a, b = values[in_a].dropna(), values[~in_a].dropna()
            what = head | {"comparison": comparison, "marker_id": marker_id, "marker_name": marker_name}
            if min(len(a), len(b)) < minimum:
                refused.append(
                    what
                    | {
                        "n_a": _shown(len(a)),
                        "n_b": _shown(len(b)),
                        "reason": f"fewer than {minimum} authors with a clear value on a side (spec S4)",
                    }
                )
                continue
            rows.append(what | {"sex": "all"} | _test(int(a.sum()), len(a), int(b.sum()), len(b)))

    multi = int((profiles["classes"].map(len) > 1).sum())
    summary += [
        {"level": CLASS, "group": "in several classes", "authors": _shown(multi)},
        {"level": CLASS, "group": "in no class", "authors": _shown(len(has_drug) - len(has_class))},
        {"level": DRUG, "group": "no causal drug recorded", "authors": _shown(len(profiles) - len(has_drug))},
    ]

    table = pd.DataFrame(rows, columns=list(COLUMNS))
    findings = _hide_small_cells(label_findings(table, settings, self_report=True))
    tested_groups = [str(g) for g in dict.fromkeys(findings["group_a"])]
    recorded = settings.record() | {"top_drugs": top_drugs}
    record = run_record(data, tested_groups, recorded) | {"analysis": ANALYSIS, "note": NOTE}
    return ProfileResult(
        findings=findings,
        refused=pd.DataFrame(refused, columns=list(REFUSED_COLUMNS)).astype({"n_a": "Int64", "n_b": "Int64"}),
        summary=pd.DataFrame(summary, columns=list(SUMMARY_COLUMNS)).astype({"authors": "Int64"}),
        skipped=pd.DataFrame(columns=["level", "group", "reason"]),
        note=NOTE,
        statement=statement(findings) if len(findings) else NOTHING_RELIABLE,
        settings=recorded,
        record=record,
    )


def _single_drugs(has_drug: pd.DataFrame, top: int) -> list[str]:
    """The `top` most common named drugs (not `unspecified_...`), leaving out a drug taken by
    exactly the authors of its class, whose class row is the same test. Ties go alphabetically."""
    counts = has_drug["drugs"].explode().value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    chosen: list[str] = []
    for drug in counts.index:
        if len(chosen) == top:
            break
        if str(drug).startswith("unspecified_") or drug in UNSPECIFIED:
            continue
        cls = DRUG_CLASSES.get(str(drug))
        if cls is not None:
            took = set(has_drug.index[has_drug["drugs"].map(lambda d, x=drug: x in d)])
            in_class = set(has_drug.index[has_drug["classes"].map(lambda c, x=cls: x in c)])
            if took == in_class:
                continue
        chosen.append(str(drug))
    return chosen


def _test(reported_a: int, n_a: int, reported_b: int, n_b: int) -> dict[str, object]:
    """One yes/no outcome compared between two sides of authors."""
    share_a, share_b = reported_a / n_a, reported_b / n_b
    low, high = newcombe_interval(reported_a, n_a, reported_b, n_b)
    ratio, ratio_low, ratio_high = odds_ratio(reported_a, n_a - reported_a, reported_b, n_b - reported_b)
    return {
        "n_a": n_a,
        "n_b": n_b,
        "reported_a": reported_a,
        "reported_b": reported_b,
        "share_a": share_a,
        "share_b": share_b,
        "effect_size": share_a - share_b,
        "ci_low": low,
        "ci_high": high,
        "odds_ratio": ratio,
        "odds_ratio_low": ratio_low,
        "odds_ratio_high": ratio_high,
        "p_value": fisher_exact(reported_a, n_a - reported_a, reported_b, n_b - reported_b),
    }


def _hide_small_cells(findings: pd.DataFrame) -> pd.DataFrame:
    """Hide what would give back a cell below 5: on a side, reporters or non-reporters, and any
    size below 5, which a `min_group_size` under 5 lets through (spec S11). The label stays: it
    comes from the run's one correction."""
    out = findings.astype({"n_a": "Int64", "n_b": "Int64", "reported_a": "Int64", "reported_b": "Int64"})
    cells = pd.concat(
        [
            out["reported_a"],
            out["n_a"] - out["reported_a"],
            out["reported_b"],
            out["n_b"] - out["reported_b"],
        ],
        axis=1,
    )
    small = (cells < MIN_SHOWN).any(axis=1).to_numpy(dtype=bool)
    for column in CELL_COLUMNS:
        out[column] = out[column].mask(small)
    for column in ("n_a", "n_b"):
        out[column] = out[column].mask(out[column] < MIN_SHOWN)
    return out


def newcombe_interval(x_a: int, n_a: int, x_b: int, n_b: int) -> tuple[float, float]:
    """95% interval of the difference of shares x_a / n_a - x_b / n_b, from each share's Wilson
    interval (Newcombe 1998, method 10). It stays within -1..1 and works for shares of 0 or 1."""
    p_a, p_b = x_a / n_a, x_b / n_b
    low_a, high_a = wilson_interval(x_a, n_a)
    low_b, high_b = wilson_interval(x_b, n_b)
    d = p_a - p_b
    return (
        d - math.sqrt((p_a - low_a) ** 2 + (high_b - p_b) ** 2),
        d + math.sqrt((high_a - p_a) ** 2 + (p_b - low_b) ** 2),
    )


def _name(symptom: str) -> str:
    return symptom.replace("_", " ")


def _shown(count: int) -> int | None:
    return count if count >= MIN_SHOWN else None


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided p-value of Fisher's exact test on the table [[a, b], [c, d]]: the probability,
    with the margins fixed, of a table no more likely than this one (spec S1)."""
    row, col, n = a + b, a + c, a + b + c + d

    def log_p(x: int) -> float:
        return _log_choose(row, x) + _log_choose(n - row, col - x) - _log_choose(n, col)

    observed = log_p(a)
    tables = range(max(0, col - (n - row)), min(row, col) + 1)
    total = sum(math.exp(log_p(x)) for x in tables if log_p(x) <= observed + 1e-7)
    return min(1.0, total)


def odds_ratio(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Odds ratio (a / b) / (c / d) of [[a, b], [c, d]] and its 95% Woolf (log) interval. When a
    cell is 0, 0.5 is added to every cell (Haldane-Anscombe), so the ratio stays finite."""
    if 0 in (a, b, c, d):
        a, b, c, d = (x + 0.5 for x in (a, b, c, d))  # type: ignore[assignment]
    log_ratio = math.log(a * d / (b * c))
    half = _Z * math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return math.exp(log_ratio), math.exp(log_ratio - half), math.exp(log_ratio + half)


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
