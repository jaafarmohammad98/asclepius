"""Synthetic datasets where the true answer is known, for testing the statistics (spec S8).

Two kinds, matching the two loaders:

- `write_lab_reports` writes an extractor CSV file (`reports.csv`), a `metadata.csv` naming each
  record's group and tier (`asclepius.metadata`), and optionally its drug and questionnaire
  symptom scores, and `truth.json` with the settings and the planted differences.
- `write_self_reports` writes the harvester's five tables and `truth.json` with the planted
  drug-symptom associations and intervention outcome shares.

Nothing here is real data. The same settings and seed always give byte-identical files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from os import PathLike
from pathlib import Path

import numpy as np

from asclepius import dataset, metadata, selfreport

# ---------------------------------------------------------------- lab reports


@dataclass(frozen=True)
class Marker:
    id: str
    name: str
    category: str
    unit: str
    male: tuple[float, float]  # reference range (low, high)
    female: tuple[float, float]


_ANDROGENS = "Androgens and Metabolites"
_ESTROGENS = "Estrogens and Metabolites"
_CORTISOL = "Cortisol Metabolites and DHEA-S"
_FREE = "Daily Free Cortisol and Cortisone"
_PROGESTERONE = "Progesterone Metabolites"

# Ranges as printed on the public DUTCH sample reports (tests/fixtures/dutch_samples.csv). Includes
# every marker the enzyme-activity ratios (spec A2) need.
MARKERS = (
    Marker("testosterone", "Testosterone", _ANDROGENS, "ng/mg", (25, 115), (2.3, 14)),
    Marker("epi_testosterone", "Epi-Testosterone", _ANDROGENS, "ng/mg", (25, 115), (2.3, 14)),
    Marker("5a_dht", "5a-DHT", _ANDROGENS, "ng/mg", (5, 25), (0, 6.6)),
    Marker("androsterone", "Androsterone", _ANDROGENS, "ng/mg", (500, 3000), (200, 1650)),
    Marker("etiocholanolone", "Etiocholanolone", _ANDROGENS, "ng/mg", (400, 1500), (200, 1000)),
    Marker("5a_androstanediol", "5a-Androstanediol", _ANDROGENS, "ng/mg", (30, 250), (12, 30)),
    Marker("5b_androstanediol", "5b-Androstanediol", _ANDROGENS, "ng/mg", (40, 250), (20, 75)),
    Marker("dhea_s", "DHEA-S", _ANDROGENS, "ng/mg", (30, 1500), (20, 750)),
    Marker("estrone_e1", "Estrone(E1)", _ESTROGENS, "ng/mg", (4, 16), (12, 26)),
    Marker("estradiol_e2", "Estradiol(E2)", _ESTROGENS, "ng/mg", (0.5, 2.2), (1.8, 4.5)),
    Marker("estriol_e3", "Estriol(E3)", _ESTROGENS, "ng/mg", (2, 8), (5, 18)),
    Marker("2_oh_e1", "2-OH-E1", _ESTROGENS, "ng/mg", (0, 5.9), (5.1, 13.1)),
    Marker("4_oh_e1", "4-OH-E1", _ESTROGENS, "ng/mg", (0, 0.8), (0, 1.8)),
    Marker("16_oh_e1", "16-OH-E1", _ESTROGENS, "ng/mg", (0, 1.2), (0.7, 2.6)),
    Marker("2_methoxy_e1", "2-Methoxy-E1", _ESTROGENS, "ng/mg", (0, 2.8), (2.5, 6.5)),
    Marker("a_pregnanediol", "a-Pregnanediol", _PROGESTERONE, "ng/mg", (20, 130), (200, 740)),
    Marker("b_pregnanediol", "b-Pregnanediol", _PROGESTERONE, "ng/mg", (75, 400), (600, 2000)),
    Marker(
        "a_tetrahydrocortisol_a_thf",
        "a-Tetrahydrocortisol (a-THF)",
        _CORTISOL,
        "ng/mg",
        (175, 700),
        (75, 370),
    ),
    Marker(
        "b_tetrahydrocortisol_b_thf",
        "b-Tetrahydrocortisol (b-THF)",
        _CORTISOL,
        "ng/mg",
        (1750, 4000),
        (1050, 2500),
    ),
    Marker(
        "b_tetrahydrocortisone_b_the",
        "b-Tetrahydrocortisone (b-THE)",
        _CORTISOL,
        "ng/mg",
        (2350, 5800),
        (1550, 3800),
    ),
    Marker("24hr_free_cortisol", "24hr Free Cortisol", _FREE, "ng/mg", (100, 310), (80, 230)),
    Marker("24hr_free_cortisone", "24hr Free Cortisone", _FREE, "ng/mg", (250, 500), (220, 450)),
    Marker("melatonin_waking", "Melatonin (Waking)", "Melatonin", "ng/mg", (10, 85), (10, 85)),
    Marker("homovanillate_hva", "Homovanillate (HVA)", "Dopamine Metabolite", "ug/mg", (4.8, 19), (4.5, 13)),
    Marker(
        "vanilmandelate_vma",
        "Vanilmandelate (VMA)",
        "Norepinephrine/Epinephrine Metabolite",
        "ug/mg",
        (2.8, 8),
        (2.7, 6.4),
    ),
    Marker(
        "5_hydroxyindoleacetate_5hiaa",
        "5-Hydroxyindoleacetate (5-HIAA)",
        "Serotonin Metabolite",
        "ug/mg",
        (3, 7.5),
        (3, 7.5),
    ),
)
MARKER_IDS = tuple(m.id for m in MARKERS)
LAB_GROUPS = metadata.GROUPS
LAB_COLUMNS = dataset.COLUMNS  # the extractor CSV format


@dataclass(frozen=True)
class PlantedDifference:
    """`group`'s values of `marker` are `fold` times those of everyone else (0.6 = 40% lower).

    `sex` limits it to one sex; None plants it in both.
    """

    marker: str
    fold: float
    group: str = "PSSD"
    sex: str | None = None


@dataclass(frozen=True)
class PlantedScore:
    """`group`'s symptom score `score` is `shift` points higher than everyone else's (before it is
    rounded and kept within 0 to 10)."""

    score: str
    shift: float
    group: str = "PSSD"


SCORE_MEAN, SCORE_SD = 4.0, 2.0  # a symptom score on a 0 to 10 scale, when nothing is planted


@dataclass(frozen=True)
class LabOutput:
    reports: Path  # extractor CSV
    metadata: Path  # record_id, group, tier
    truth: Path  # settings and planted differences


def write_lab_reports(
    out_dir: str | PathLike[str],
    *,
    groups: Mapping[str, int] | None = None,
    planted: Sequence[PlantedDifference] = (),
    null: bool = False,
    female_share: float = 0.3,
    tier_a_share: float = 0.25,
    scores: Sequence[str] = (),
    planted_scores: Sequence[PlantedScore] = (),
    drugs: Mapping[str, Sequence[str]] | None = None,
    seed: int = 0,
) -> LabOutput:
    """Write `groups` (name -> number of patients) of synthetic DUTCH reports.

    Values are log-normal per sex, with the report's reference range at about the 2.5th and 97.5th
    percentiles of a healthy person. `null` mode plants nothing: no group differs from another.
    About `tier_a_share` of each group is tier A, the rest tier B.

    `scores` names questionnaire symptom scores, written to the metadata as `score_<name>`: whole
    numbers from 0 to 10, normal with mean 4 and sd 2 before rounding, plus any `planted_scores`.
    `drugs` maps a group to the drugs its patients took, each patient one of them at random.
    Neither changes the reports themselves.
    """
    # 40 per group, so that even at 30% women each sex has at least 10 (spec S4)
    groups = dict(groups if groups is not None else {"PSSD": 40, "healthy control": 40})
    drugs = {group: list(names) for group, names in (drugs or {}).items()}
    _check_lab_settings(groups, planted, null, female_share)
    _check_questionnaire_settings(groups, scores, planted_scores, drugs, null)
    if not 0 <= tier_a_share <= 1:
        raise ValueError("tier_a_share must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    tier_rng = np.random.default_rng([seed, 1])  # separate, so tiers never change the values
    meta_rng = np.random.default_rng([seed, 2])  # and scores and drugs neither
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = []
    assigned: list[list[object]] = []
    number = 0
    for group, size in groups.items():
        females = round(size * female_share)
        for i in range(size):
            number += 1
            record_id = f"r{number:03d}"
            sex = "female" if i < females else "male"
            assigned.append([record_id, group, "A" if tier_rng.random() < tier_a_share else "B"])
            if group in drugs:
                assigned[-1].append(drugs[group][int(meta_rng.integers(len(drugs[group])))])
            elif drugs:
                assigned[-1].append("")
            for score in scores:
                shift = sum(p.shift for p in planted_scores if p.score == score and p.group == group)
                assigned[-1].append(int(np.clip(round(meta_rng.normal(SCORE_MEAN, SCORE_SD) + shift), 0, 10)))
            age = int(rng.integers(20, 56))
            month = int(rng.integers(0, 46))  # 2023-01 .. 2026-10
            collected = f"{2023 + month // 12}-{month % 12 + 1:02d}"
            reported = f"{2023 + (month + 1) // 12}-{(month + 1) % 12 + 1:02d}"
            for marker in MARKERS:
                low, high = marker.male if sex == "male" else marker.female
                value = _healthy_value(rng, low, high)
                for p in planted:
                    if p.marker == marker.id and p.group == group and p.sex in (None, sex):
                        value *= p.fold
                value = _round(value)
                status = "Below range" if value < low else "Above range" if value > high else "Within range"
                rows.append(
                    [record_id, sex, age, collected, reported, marker.id, marker.name, marker.category]
                    + ["", value, marker.unit, _num(low), _num(high), status]
                )

    _write_csv(out / "reports.csv", LAB_COLUMNS, rows)
    columns = ["record_id", "group", "tier", *(["drug"] if drugs else []), *(f"score_{s}" for s in scores)]
    _write_csv(out / "metadata.csv", columns, assigned)
    truth = {
        "kind": "lab-report",
        "generator": f"asclepius.synthetic {version('asclepius')}",
        "seed": seed,
        "null": not planted,
        "groups": groups,
        "female_share": female_share,
        "tier_a_share": tier_a_share,
        "planted": [asdict(p) for p in planted],
    }
    if scores or drugs:
        truth |= {
            "scores": list(scores),
            "planted_scores": [asdict(p) for p in planted_scores],
            "drugs": drugs,
        }
    _write_json(out / "truth.json", truth)
    return LabOutput(reports=out / "reports.csv", metadata=out / "metadata.csv", truth=out / "truth.json")


def _check_lab_settings(
    groups: Mapping[str, int], planted: Sequence[PlantedDifference], null: bool, female_share: float
) -> None:
    for name, size in groups.items():
        if name not in LAB_GROUPS:
            raise ValueError(f"Unknown group {name!r}; use one of {', '.join(LAB_GROUPS)}.")
        if size < 1:
            raise ValueError(f"Group {name!r} needs at least one patient.")
    if not 0 <= female_share <= 1:
        raise ValueError("female_share must be between 0 and 1.")
    if null and planted:
        raise ValueError("Null mode plants no differences; leave `planted` empty.")
    for p in planted:
        if p.marker not in MARKER_IDS:
            raise ValueError(f"Unknown marker {p.marker!r}.")
        if p.group not in groups:
            raise ValueError(f"Planted difference in {p.group!r}, which is not one of the groups.")
        if not (p.fold > 0 and math.isfinite(p.fold)):
            raise ValueError(f"Planted fold for {p.marker} must be a positive number.")
        if p.sex not in (None, "male", "female"):
            raise ValueError(f"Planted sex must be male, female or None, not {p.sex!r}.")


def _check_questionnaire_settings(
    groups: Mapping[str, int],
    scores: Sequence[str],
    planted_scores: Sequence[PlantedScore],
    drugs: Mapping[str, Sequence[str]],
    null: bool,
) -> None:
    for score in scores:
        if not metadata.SCORE.match(f"score_{score}"):
            raise ValueError(f"{score!r}: score names are letters, digits and underscores.")
    if null and planted_scores:
        raise ValueError("Null mode plants no differences; leave `planted_scores` empty.")
    for p in planted_scores:
        if p.score not in scores:
            raise ValueError(f"Planted score {p.score!r} is not one of the scores.")
        if p.group not in groups:
            raise ValueError(f"Planted score in {p.group!r}, which is not one of the groups.")
        if not math.isfinite(p.shift):
            raise ValueError(f"Planted shift for {p.score} must be a number.")
    for group, names in drugs.items():
        if group not in groups:
            raise ValueError(f"Drugs for {group!r}, which is not one of the groups.")
        if not names or not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError(f"Drugs for {group!r} must be a list of drug names.")


def _healthy_value(rng: np.random.Generator, low: float, high: float) -> float:
    low = low if low > 0 else high / 20  # a range starting at 0 means "below detection is normal"
    mu = (math.log(low) + math.log(high)) / 2
    sigma = (math.log(high) - math.log(low)) / (2 * 1.96)
    return float(math.exp(rng.normal(mu, sigma)))


def _round(x: float) -> float:
    """Three significant figures, as reports print them."""
    return float(f"{x:.3g}")


def _num(x: float) -> str:
    return f"{x:g}"


# ---------------------------------------------------------------- self-reports

# condition -> (share of authors, causal drugs). Names as the harvester writes them.
CONDITIONS = {
    "PSSD": (0.8, ("sertraline", "escitalopram", "fluoxetine", "paroxetine", "venlafaxine", "citalopram")),
    "PFS": (0.12, ("finasteride",)),
    "PSA": (0.08, ("isotretinoin",)),
}
DRUGS = tuple(drug for _, drugs in CONDITIONS.values() for drug in drugs)
# usual daily doses in mg, one picked per author
DOSES = {
    "sertraline": (50, 100, 150),
    "escitalopram": (10, 20),
    "fluoxetine": (20, 40),
    "paroxetine": (20, 40),
    "venlafaxine": (75, 150),
    "citalopram": (20, 40),
    "finasteride": (1, 5),
    "isotretinoin": (20, 40, 60),
}
# symptom -> share of authors reporting it, when nothing is planted
SYMPTOMS = {
    "low_libido": 0.55,
    "anhedonia": 0.4,
    "emotional_blunting": 0.4,
    "genital_numbness": 0.35,
    "erectile_dysfunction": 0.3,
    "brain_fog": 0.25,
    "anorgasmia": 0.2,
    "depression": 0.2,
    "fatigue": 0.15,
    "insomnia": 0.12,
}
INTERVENTIONS = (
    "bupropion",
    "testosterone_replacement",
    "tadalafil",
    "buspirone",
    "sildenafil",
    "exercise",
    "cyproheptadine",
    "vitamin_d",
    "probiotics",
    "psychotherapy",
)
# share of the authors who tried an intervention reporting improvement (major or partial) and worse,
# when nothing is planted; the rest report no effect
DEFAULT_IMPROVED = 0.25
DEFAULT_WORSE = 0.15
MAJOR_SHARE = 1 / 3  # of those improved, the share reporting major improvement
SELF_REPORT_COLUMNS = {name: table.columns for name, table in selfreport.TABLES.items()}


@dataclass(frozen=True)
class PlantedAssociation:
    """Authors whose causal drug is `drug` report `symptom` with `odds_ratio` times the usual odds.

    That is the odds ratio against everyone else as long as no other drug has an association
    planted on the same symptom."""

    drug: str
    symptom: str
    odds_ratio: float


@dataclass(frozen=True)
class PlantedOutcome:
    """Of the authors who tried `intervention`, `improved` report improvement (a third of them
    major) and `worse` report it made them worse; the rest report no effect."""

    intervention: str
    improved: float
    worse: float


@dataclass(frozen=True)
class SelfReportOutput:
    export: Path  # folder with the five tables, like harvester/data/export/
    truth: Path


@dataclass
class _Post:
    case_id: str
    year: int
    cells: dict[str, object] = field(default_factory=dict)
    drugs: list[list[object]] = field(default_factory=list)
    symptoms: list[list[object]] = field(default_factory=list)
    interventions: list[list[object]] = field(default_factory=list)
    labs: list[list[object]] = field(default_factory=list)


def write_self_reports(
    out_dir: str | PathLike[str],
    *,
    authors: int = 300,
    associations: Sequence[PlantedAssociation] = (),
    outcomes: Sequence[PlantedOutcome] = (),
    null: bool = False,
    repeat_share: float = 0.3,
    deleted_posts: int = 10,
    unverified_posts: int = 10,
    seed: int = 0,
) -> SelfReportOutput:
    """Write a synthetic harvester export of `authors` people.

    `repeat_share` of authors post 2 to 4 times; their symptoms and interventions are spread over
    the posts, so the truth only shows once the loader merges them per author. `deleted_posts`
    extra posts have no author (deleted accounts), which the loader must leave out, and
    `unverified_posts` extra posts, each from an author of its own, fail the evidence check, so
    the loader leaves them out by default. `null` mode plants nothing: no drug differs in symptoms and every
    intervention has the same outcome shares.
    """
    _check_self_report_settings(
        authors, associations, outcomes, null, repeat_share, deleted_posts + unverified_posts
    )
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conditions = list(CONDITIONS)
    weights = np.array([CONDITIONS[c][0] for c in conditions])
    shares = {o.intervention: (o.improved, o.worse) for o in outcomes}

    posts: list[_Post] = []
    keys: set[str] = set()
    for _ in range(authors):
        key = _author_key(rng, keys)
        condition = conditions[int(rng.choice(len(conditions), p=weights))]
        drugs = CONDITIONS[condition][1]
        drug = drugs[int(rng.integers(len(drugs)))]
        count = int(rng.integers(2, 5)) if rng.random() < repeat_share else 1
        years = sorted(int(y) for y in rng.choice(np.arange(2015, 2027), size=count, replace=False))
        mine = [_Post(case_id="", year=y) for y in years]
        sex = "male" if rng.random() < 0.75 else "female"
        for i, post in enumerate(mine):
            post.cells = {
                "author_key": key,
                "kind": "submission" if i == 0 else "comment",
                "condition": condition,
                "sex": sex if i == 0 or rng.random() < 0.5 else "",
                "onset": "after_stopping" if i == 0 else "",
                "evidence_ok": "True",
            }
            post.drugs.append([drug, float(rng.choice(DOSES[drug])) if i == 0 else "", ""])
        latest = mine[-1]
        latest.cells["age"] = int(rng.integers(18, 51))
        latest.cells["months_since_stopping"] = int(rng.integers(1, 120))
        latest.cells["overall_trajectory"] = str(
            rng.choice(["recovered", "improving", "stable", "worsening"])
        )

        for symptom, base in SYMPTOMS.items():
            odds = base / (1 - base)
            for a in associations:
                if a.drug == drug and a.symptom == symptom:
                    odds *= a.odds_ratio
            if rng.random() < odds / (1 + odds):
                severity = str(rng.choice(["", "mild", "moderate", "severe"]))
                mine[int(rng.integers(count))].symptoms.append([symptom, severity])

        tried = rng.choice(len(INTERVENTIONS), size=int(rng.integers(0, 4)), replace=False)
        for index in sorted(int(i) for i in tried):
            intervention = INTERVENTIONS[index]
            improved, worse = shares.get(intervention, (DEFAULT_IMPROVED, DEFAULT_WORSE))
            u = rng.random()
            if u < improved:
                outcome = "major_improvement" if u < improved * MAJOR_SHARE else "partial_improvement"
            elif u < improved + worse:
                outcome = "worse"
            else:
                outcome = "no_effect"
            weeks = int(rng.integers(1, 27)) if rng.random() < 0.5 else ""
            mine[int(rng.integers(count))].interventions.append([intervention, outcome, weeks])

        if rng.random() < 0.1 and sex == "male":
            value = _round(_healthy_value(rng, 264, 916))
            flag = "low" if value < 264 else "high" if value > 916 else "normal"
            latest.labs.append(["total_testosterone", value, "ng/dL", 264.0, 916.0, flag, "blood"])
        posts.extend(mine)

    for i in range(deleted_posts + unverified_posts):
        post = _Post(case_id="", year=int(rng.integers(2015, 2027)))
        deleted = i < deleted_posts
        post.cells = {"kind": "comment", "condition": "PSSD", "evidence_ok": "True" if deleted else "False"}
        if not deleted:
            post.cells["author_key"] = _author_key(rng, keys)
        post.drugs.append([DRUGS[int(rng.integers(len(DRUGS)))], "", ""])
        post.symptoms.append(["anhedonia", ""])
        posts.insert(int(rng.integers(len(posts) + 1)), post)

    _write_self_report_tables(out, posts)
    truth = {
        "kind": "self-report",
        "generator": f"asclepius.synthetic {version('asclepius')}",
        "seed": seed,
        "null": not associations and not outcomes,
        "authors": authors,
        "repeat_share": repeat_share,
        "deleted_posts": deleted_posts,
        "unverified_posts": unverified_posts,
        "symptom_shares": SYMPTOMS,
        "default_outcome_shares": {"improved": DEFAULT_IMPROVED, "worse": DEFAULT_WORSE},
        "associations": [asdict(a) for a in associations],
        "outcomes": [asdict(o) for o in outcomes],
    }
    _write_json(out / "truth.json", truth)
    return SelfReportOutput(export=out, truth=out / "truth.json")


def _check_self_report_settings(
    authors: int,
    associations: Sequence[PlantedAssociation],
    outcomes: Sequence[PlantedOutcome],
    null: bool,
    repeat_share: float,
    extra_posts: int,
) -> None:
    if authors < 1:
        raise ValueError("Write at least one author.")
    if extra_posts < 0:
        raise ValueError("deleted_posts and unverified_posts cannot be negative.")
    if not 0 <= repeat_share <= 1:
        raise ValueError("repeat_share must be between 0 and 1.")
    if null and (associations or outcomes):
        raise ValueError("Null mode plants nothing; leave `associations` and `outcomes` empty.")
    for a in associations:
        if a.drug not in DRUGS:
            raise ValueError(f"Unknown drug {a.drug!r}; use one of {', '.join(DRUGS)}.")
        if a.symptom not in SYMPTOMS:
            raise ValueError(f"Unknown symptom {a.symptom!r}; use one of {', '.join(SYMPTOMS)}.")
        if not (a.odds_ratio > 0 and math.isfinite(a.odds_ratio)):
            raise ValueError(f"Odds ratio for {a.drug} and {a.symptom} must be a positive number.")
    for o in outcomes:
        if o.intervention not in INTERVENTIONS:
            raise ValueError(
                f"Unknown intervention {o.intervention!r}; use one of {', '.join(INTERVENTIONS)}."
            )
        if not (o.improved >= 0 and o.worse >= 0 and o.improved + o.worse <= 1):
            raise ValueError(
                f"Outcome shares for {o.intervention} must be between 0 and 1 and add to at most 1."
            )


def _author_key(rng: np.random.Generator, taken: set[str]) -> str:
    """16 hex characters, like the harvester's salted hash."""
    while True:
        key = rng.bytes(8).hex()
        if key not in taken:
            taken.add(key)
            return key


def _write_self_report_tables(out: Path, posts: list[_Post]) -> None:
    tables: dict[str, list[list[object]]] = {name: [] for name in SELF_REPORT_COLUMNS}
    case_columns = SELF_REPORT_COLUMNS["cases"]
    for number, post in enumerate(posts, start=1):
        case_id = f"syn{number:05d}"
        cells = {**post.cells, "case_id": case_id, "year": post.year}
        tables["cases"].append([cells.get(c, "") for c in case_columns])
        for name, rows in (
            ("causal_drugs", post.drugs),
            ("symptoms", post.symptoms),
            ("interventions", post.interventions),
            ("labs", post.labs),
        ):
            tables[name].extend([case_id, *row] for row in rows)
    for name, columns in SELF_REPORT_COLUMNS.items():
        _write_csv(out / f"{name}.csv", columns, tables[name])


# ---------------------------------------------------------------- writing


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(
            ["" if v is None else _num(v) if isinstance(v, float) else v for v in row] for row in rows
        )


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m asclepius.synthetic", description="Write a synthetic dataset with a known answer."
    )
    kinds = parser.add_subparsers(dest="kind", required=True)

    lab = kinds.add_parser("lab", help="extractor CSV with metadata.csv and truth.json")
    lab.add_argument("out_dir")
    lab.add_argument("--group", action="append", metavar="NAME=N", help='e.g. "PSSD=40" (repeatable)')
    lab.add_argument("--plant", action="append", default=[], metavar="MARKER:FOLD[:GROUP[:SEX]]")
    lab.add_argument("--female-share", type=float, default=0.3)
    lab.add_argument("--tier-a-share", type=float, default=0.25)

    self_report = kinds.add_parser("selfreport", help="the harvester's five tables with truth.json")
    self_report.add_argument("out_dir")
    self_report.add_argument("--authors", type=int, default=300)
    self_report.add_argument("--associate", action="append", default=[], metavar="DRUG:SYMPTOM:ODDS_RATIO")
    self_report.add_argument("--outcome", action="append", default=[], metavar="INTERVENTION:IMPROVED:WORSE")
    self_report.add_argument("--repeat-share", type=float, default=0.3)
    self_report.add_argument("--deleted-posts", type=int, default=10)
    self_report.add_argument("--unverified-posts", type=int, default=10)

    for sub in (lab, self_report):
        sub.add_argument("--null", action="store_true", help="plant nothing")
        sub.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        _run(args)
    except ValueError as err:  # includes a malformed --group, --plant, --associate or --outcome
        parser.error(str(err))


def _fields(item: str, low: int, high: int, form: str) -> list[str]:
    parts = item.split(":")
    if not low <= len(parts) <= high:
        raise ValueError(f"{item!r} should be written as {form}.")
    return parts


def _run(args: argparse.Namespace) -> None:
    if args.kind == "lab":
        groups = None
        if args.group:
            groups = {}
            for item in args.group:
                name, _, size = item.rpartition("=")
                if not name or not size.isdigit():
                    raise ValueError(f"{item!r} should be written as NAME=N.")
                groups[name] = int(size)
        planted = []
        for item in args.plant:
            marker, fold, *rest = _fields(item, 2, 4, "MARKER:FOLD[:GROUP[:SEX]]")
            group = rest[0] if rest else "PSSD"
            sex = rest[1] if len(rest) > 1 else None
            planted.append(PlantedDifference(marker, float(fold), group, sex))
        lab_out = write_lab_reports(
            args.out_dir,
            groups=groups,
            planted=planted,
            null=args.null,
            female_share=args.female_share,
            tier_a_share=args.tier_a_share,
            seed=args.seed,
        )
        print(f"Wrote {lab_out.reports}, {lab_out.metadata} and {lab_out.truth}")
    else:
        associations = [
            PlantedAssociation(drug, symptom, float(ratio))
            for drug, symptom, ratio in (_fields(i, 3, 3, "DRUG:SYMPTOM:ODDS_RATIO") for i in args.associate)
        ]
        outcomes = [
            PlantedOutcome(name, float(improved), float(worse))
            for name, improved, worse in (
                _fields(i, 3, 3, "INTERVENTION:IMPROVED:WORSE") for i in args.outcome
            )
        ]
        sr_out = write_self_reports(
            args.out_dir,
            authors=args.authors,
            associations=associations,
            outcomes=outcomes,
            null=args.null,
            repeat_share=args.repeat_share,
            deleted_posts=args.deleted_posts,
            unverified_posts=args.unverified_posts,
            seed=args.seed,
        )
        print(f"Wrote the five tables and {sr_out.truth.name} to {sr_out.export}")


if __name__ == "__main__":
    main()
