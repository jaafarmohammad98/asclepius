"""Load the r/PSSD harvester export (`harvester/data/export/`) as a self-report dataset.

The export is five CSV tables linked by `case_id`. A case is one post or comment; the unit of
analysis is the author (spec S9), so every table is merged to one row per author per item.
Case ids and author keys are dropped after merging and never appear in a message: authors are
numbered 1, 2, 3... and the numbers say nothing about who they are.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import pandas as pd

from asclepius.dataset import DatasetError

LABEL = "self-report, no controls"

# harvester condition -> the plan's group (PSA, post-Accutane, is the plan's PAS)
CONDITION_GROUPS = {
    "PSSD": "PSSD",
    "PFS": "PFS",
    "PSA": "PAS",
    "multiple": "multiple",
    "unclear": "unclear",
    "none": "none",
}
NAMED_GROUPS = frozenset({"PSSD", "PFS", "PAS"})
SEVERITIES = ("mild", "moderate", "severe")
OUTCOMES = ("major_improvement", "partial_improvement", "no_effect", "worse", "unclear")


@dataclass(frozen=True)
class _Table:
    columns: tuple[str, ...]
    required: tuple[str, ...] = ()
    allowed: dict[str, frozenset[str]] = field(default_factory=dict)
    numbers: tuple[str, ...] = ()
    years: tuple[str, ...] = ()


TABLES = {
    "cases": _Table(
        columns=(
            "case_id",
            "author_key",
            "year",
            "kind",
            "condition",
            "sex",
            "age",
            "onset",
            "months_since_stopping",
            "overall_trajectory",
            "evidence_ok",
        ),
        required=("case_id", "evidence_ok"),
        allowed={
            "kind": frozenset({"submission", "comment"}),
            "condition": frozenset(CONDITION_GROUPS),
            "sex": frozenset({"male", "female", "other"}),
            "onset": frozenset({"during_use", "after_stopping", "unclear"}),
            "overall_trajectory": frozenset({"recovered", "improving", "stable", "worsening", "unclear"}),
            "evidence_ok": frozenset({"True", "False"}),
        },
        numbers=("age", "months_since_stopping"),
        years=("year",),
    ),
    "causal_drugs": _Table(
        columns=("case_id", "drug", "dose_mg", "duration_months"),
        required=("case_id", "drug"),
        numbers=("dose_mg", "duration_months"),
    ),
    "symptoms": _Table(
        columns=("case_id", "symptom", "severity"),
        required=("case_id", "symptom"),
        allowed={"severity": frozenset(SEVERITIES)},
    ),
    "interventions": _Table(
        columns=("case_id", "intervention", "outcome", "duration_weeks"),
        required=("case_id", "intervention", "outcome"),
        allowed={"outcome": frozenset(OUTCOMES)},
        numbers=("duration_weeks",),
    ),
    "labs": _Table(
        columns=("case_id", "test", "value", "unit", "ref_low", "ref_high", "flag", "panel"),
        required=("case_id", "test", "value"),
        numbers=("value", "ref_low", "ref_high"),
    ),
}


@dataclass(frozen=True)
class SelfReportDataset:
    """Public self-reports merged per author. Never pooled with a `LabDataset`.

    `authors` has one row per author; the other tables have one row per author per drug, symptom,
    intervention or lab value, keyed by the `author` number. `excluded` counts the posts left out:
    `unverified` (`evidence_ok` false, unless asked for) and `no_author` (deleted accounts, which
    cannot be merged, so one person could be counted many times). `fingerprint` (a SHA-256 over
    the five files) and `settings` record what a result was computed from (spec S7).
    """

    authors: pd.DataFrame
    causal_drugs: pd.DataFrame
    symptoms: pd.DataFrame
    interventions: pd.DataFrame
    labs: pd.DataFrame
    excluded: dict[str, int]
    kind: str = "self-report"
    fingerprint: str = ""
    settings: dict[str, bool] = field(default_factory=dict)
    label: str = LABEL


def load_self_reports(
    export_dir: str | PathLike[str], *, include_unverified: bool = False
) -> SelfReportDataset:
    export_dir = Path(export_dir)
    tables = {name: _read_table(export_dir / f"{name}.csv", spec) for name, spec in TABLES.items()}
    _check_links(tables)
    digest = hashlib.sha256()
    for name in TABLES:  # fixed order, so the same export always gives the same fingerprint
        digest.update(hashlib.sha256((export_dir / f"{name}.csv").read_bytes()).digest())

    cases = tables["cases"]
    cases["condition"] = cases["condition"].fillna("unclear")  # the harvester blanks values it can't map
    no_author = cases["author_key"].isna()
    unverified = (cases["evidence_ok"] == "False") & ~no_author
    keep = ~no_author if include_unverified else ~no_author & ~unverified
    excluded = {
        "no_author": int(no_author.sum()),
        "unverified": 0 if include_unverified else int(unverified.sum()),
    }
    cases = cases[keep]
    numbers = {key: n for n, key in enumerate(sorted(cases["author_key"].unique()), start=1)}
    cases = cases.assign(author=cases["author_key"].map(numbers)).drop(columns="author_key")
    posted = cases.set_index("case_id")[["author", "year"]]

    def per_author(name: str) -> pd.DataFrame:
        return tables[name].join(posted, on="case_id", how="inner").drop(columns="case_id")

    authors = _merge_authors(cases)

    drugs = per_author("causal_drugs")
    causal_drugs = drugs.groupby(["author", "drug"], as_index=False)[["dose_mg", "duration_months"]].max()

    symptoms = per_author("symptoms")
    symptoms["severity"] = pd.Categorical(symptoms["severity"], categories=SEVERITIES, ordered=True)
    symptoms = symptoms.groupby(["author", "symptom"], as_index=False, observed=True)["severity"].max()

    tried = per_author("interventions")
    keys = ["author", "intervention"]
    interventions = (
        pd.DataFrame(
            {
                "outcome": _latest(tried, keys, "outcome", unclear="unclear"),
                "duration_weeks": tried.groupby(keys)["duration_weeks"].max(),
            }
        )
        .reset_index()
        .astype({"outcome": "string"})
    )

    labs = (
        per_author("labs")
        .drop(columns="year")
        .drop_duplicates()
        .sort_values(["author", "test"], ignore_index=True)
    )
    return SelfReportDataset(
        authors=authors,
        causal_drugs=causal_drugs,
        symptoms=symptoms,
        interventions=interventions,
        labs=labs,
        excluded=excluded,
        fingerprint=digest.hexdigest(),
        settings={"include_unverified": include_unverified},
    )


# ---------------------------------------------------------------- merging one author's posts


def _merge_authors(cases: pd.DataFrame) -> pd.DataFrame:
    by_author = cases.groupby("author")
    authors = pd.DataFrame(
        {
            "n_cases": by_author.size(),
            "first_year": by_author["year"].min(),
            "last_year": by_author["year"].max(),
            "condition": by_author["condition"].agg(_merge_condition),
            "sex": _agreed(cases, "sex"),
            "age": _latest(cases, ["author"], "age"),
            "onset": _agreed(cases, "onset", unclear="unclear"),
            "months_since_stopping": by_author["months_since_stopping"].max(),
            "overall_trajectory": _latest(cases, ["author"], "overall_trajectory", unclear="unclear"),
        }
    ).reset_index()
    text = ["condition", "sex", "onset", "overall_trajectory"]
    return authors.astype(dict.fromkeys(text, "string")).assign(
        age=pd.to_numeric(authors["age"]).astype("float64")
    )


def _merge_condition(values: pd.Series) -> str:
    """One group per author. Different named conditions across posts make the author `multiple`."""
    groups = {CONDITION_GROUPS[v] for v in values}
    named = groups & NAMED_GROUPS
    if len(named) > 1 or "multiple" in groups:
        return "multiple"
    if named:
        return named.pop()
    return "unclear" if "unclear" in groups else "none"


def _agreed(cases: pd.DataFrame, column: str, unclear: str | None = None) -> pd.Series:
    """Per author, the value every post that gives one agrees on. Disagreement gives `unclear`,
    or missing when the column has no such value."""
    authors = pd.Index(cases["author"].unique(), name="author")
    given = cases.dropna(subset=[column])
    clear = given[given[column] != unclear] if unclear else given
    values = clear.groupby("author")[column].agg(lambda v: v.iloc[0] if v.nunique() == 1 else pd.NA)
    result = values.reindex(authors).astype(object)
    if unclear:
        result[result.isna() & authors.isin(given["author"])] = unclear
    return result.where(result.notna(), pd.NA)


def _latest(rows: pd.DataFrame, keys: list[str], column: str, unclear: str | None = None) -> pd.Series:
    """Per key, the value from the most recent post that gives a clear one (things change).

    Posts from that same year that disagree give `unclear`; for a number, the largest.
    """
    index = (
        pd.MultiIndex.from_frame(rows[keys].drop_duplicates())
        if len(keys) > 1
        else (pd.Index(rows[keys[0]].unique(), name=keys[0]))
    )
    given = rows.dropna(subset=[column])
    clear = given[given[column] != unclear] if unclear else given
    year = clear["year"].fillna(0)  # a post with no year counts as the oldest
    newest = clear[year == year.groupby([clear[k] for k in keys]).transform("max")]
    grouped = newest.groupby(keys)[column]
    values = grouped.agg(lambda v: v.iloc[0] if v.nunique() == 1 else unclear) if unclear else grouped.max()
    result = values.reindex(index).astype(object)
    if unclear:
        had_any = given.set_index(keys).index.unique()
        result[result.isna() & index.isin(had_any)] = unclear
    return result.where(result.notna(), pd.NA)


# ---------------------------------------------------------------- reading and checking
# Messages name the file, line and column, and quote a bad value only from a column that is not
# an id: a case_id or author_key never appears in a message.

_IDS = frozenset({"case_id", "author_key"})


def _read_table(path: Path, spec: _Table) -> pd.DataFrame:
    name = path.name
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        raise DatasetError(f"{name} is missing from the export folder.") from None
    except pd.errors.EmptyDataError:
        raise DatasetError(f"{name}: the file is empty.") from None
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise DatasetError(f"{name}: not a readable CSV file.") from None
    missing = [c for c in spec.columns if c not in frame.columns]
    if missing:
        raise DatasetError(f"{name}: missing column(s) {', '.join(missing)}. Is this a harvester export?")

    text = frame[list(spec.columns)].apply(lambda col: col.str.strip())
    lines = text.index + 2  # the header is line 1

    def fail(row, column: str, problem: str):
        raise DatasetError(f"{name}, line {lines[row]}, column {column}: {problem}")

    def quote(row, column: str) -> str:
        return "this value" if column in _IDS else repr(text.at[row, column])

    for column in spec.required:
        empty = text.index[text[column] == ""]
        if len(empty):
            fail(empty[0], column, "is empty.")
    for column, allowed in spec.allowed.items():
        bad = text.index[(text[column] != "") & ~text[column].isin(allowed)]
        if len(bad):
            fail(bad[0], column, f"{quote(bad[0], column)} is not one of {', '.join(sorted(allowed))}.")

    out = text.replace("", pd.NA)
    for column in spec.numbers + spec.years:
        numbers = pd.to_numeric(out[column], errors="coerce")
        bad = text.index[
            (text[column] != "") & (numbers.isna() | numbers.isin([float("inf"), -float("inf")]))
        ]
        if len(bad):
            kind = "a year" if column in spec.years else "a number"
            fail(bad[0], column, f"{quote(bad[0], column)} is not {kind}.")
        out[column] = numbers.astype("float64")
    for column in spec.years:
        years = out[column]
        bad = text.index[years.notna() & ((years % 1 != 0) | (years < 2000) | (years > 2100))]
        if len(bad):
            fail(bad[0], column, f"{quote(bad[0], column)} is not a year.")
        out[column] = years.astype("Int64")

    duplicate = text.index[text["case_id"].duplicated()] if name == "cases.csv" else []
    if len(duplicate):
        raise DatasetError(
            f"{name}, line {lines[duplicate[0]]}: the same case_id appears on an earlier line."
        )
    return out


def _check_links(tables: dict[str, pd.DataFrame]) -> None:
    known = set(tables["cases"]["case_id"])
    for name, table in tables.items():
        if name == "cases":
            continue
        orphans = table.index[~table["case_id"].isin(known)]
        if len(orphans):
            line = orphans[0] + 2  # rows keep the index they were read with
            raise DatasetError(f"{name}.csv, line {line}: refers to a case not in cases.csv.")
