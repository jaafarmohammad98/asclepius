"""Group, tier and questionnaire data for lab reports, from a metadata file keyed by `record_id`.

The researcher (or the platform) supplies one CSV file with one row per report:

| Column | Required | Values |
|---|---|---|
| record_id | yes | `r001` when only one loaded file has that id, else `<file number>/r001` |
| group | yes (column) | one of `GROUPS`, any case; blank means `unassigned` |
| tier | no | A, B or C (overview, "Data trust"); blank or absent means B |
| drug | no | free text |
| months_since_stopping | no | a number, 0 or more |
| score_<name> | no, any number | a number (a questionnaire symptom score) |

Reports with no row go to the `unassigned` group. Rows naming a report that was not loaded are
left out and listed in `LabDataset.unmatched`. Two rows for the same report must agree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from asclepius.dataset import DatasetError

GROUPS = ("PSSD", "PFS", "PAS", "SSRI-exposed control", "healthy control")
UNASSIGNED = "unassigned"
TIERS = ("A", "B", "C")
DEFAULT_TIER = "B"
FIXED = ("record_id", "group", "tier", "drug", "months_since_stopping")
SCORE = re.compile(r"^score_[A-Za-z0-9_]+$")
_QUALIFIED = re.compile(r"^(\d+)/(.+)$")


def _group_key(name: str) -> str:
    return re.sub(r"\s+", " ", name).casefold()


_BY_KEY = {_group_key(g): g for g in GROUPS}


def default_records(markers: pd.DataFrame) -> pd.DataFrame:
    """One row per loaded report, every one `unassigned` and tier B."""
    by_report = markers.groupby("record", sort=False)
    records = by_report[["file", "record_id"]].first()
    # a report's sex and age are the same on every marker row; take the first one given
    records["sex"] = by_report["sex"].agg(lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA)
    records["age_years"] = by_report["age_years"].first().astype("Int64")
    records["group"] = UNASSIGNED
    records["tier"] = DEFAULT_TIER
    records["drug"] = pd.Series(pd.NA, index=records.index, dtype="object")
    records["months_since_stopping"] = float("nan")
    return records.reset_index()


def apply(records: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Join the metadata file at `path` to `records`. Returns the joined table and unmatched ids."""
    meta, scores = _read(path)
    name = path.name
    by_id: dict[str, list[str]] = {}
    for record, record_id in zip(records["record"], records["record_id"], strict=True):
        by_id.setdefault(record_id, []).append(record)
    known = set(records["record"])

    keys: list[str | None] = []
    unmatched: list[str] = []
    for row, written in meta["record_id"].items():
        qualified = _QUALIFIED.match(written)
        if qualified:
            key = f"{int(qualified.group(1))}/{qualified.group(2)}"
            found = key if key in known else None
        else:
            matches = by_id.get(written, [])
            if len(matches) > 1:
                raise DatasetError(
                    f"{name}, line {row}, column record_id: {written} is in more than one file; "
                    f"write it as {' or '.join(matches)}."
                )
            found = matches[0] if matches else None
        if found is None and written not in unmatched:
            unmatched.append(written)
        keys.append(found)
    meta["record"] = keys
    meta = meta[meta["record"].notna()]

    values = ["group", "tier", "drug", "months_since_stopping", *scores]
    _check_agree(meta, values, name)
    meta = meta.drop_duplicates("record").set_index("record")[values]

    out = records.set_index("record")
    for column in scores:
        out[column] = float("nan")
    out.update(meta[["months_since_stopping", *scores]])
    out.loc[meta.index, ["group", "tier", "drug"]] = meta[["group", "tier", "drug"]]
    return out.reset_index(), tuple(unmatched)


def _read(path: Path) -> tuple[pd.DataFrame, list[str]]:
    name = path.name
    try:
        # blank lines are kept as empty rows so the line numbers in messages stay right
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=False)
    except pd.errors.EmptyDataError:
        raise DatasetError(f"{name}: the file is empty.") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as err:
        raise DatasetError(f"{name}: not a readable CSV file ({err}).") from None

    columns = [c.strip() for c in frame.columns]
    frame.columns = columns
    missing = [c for c in ("record_id", "group") if c not in columns]
    unknown = [c for c in columns if c not in FIXED and not SCORE.match(c)]
    if missing:
        raise DatasetError(f"{name}: missing column(s) {', '.join(missing)}.")
    if unknown:
        raise DatasetError(
            f"{name}: unexpected column(s) {', '.join(unknown)}. Allowed: {', '.join(FIXED)}, "
            "and symptom scores named score_<name>."
        )
    scores = [c for c in columns if SCORE.match(c)]
    frame = frame.apply(lambda col: col.str.strip())
    frame.index = frame.index + 2  # line in the file, counting the header as line 1
    frame = frame[(frame != "").any(axis=1)]

    def fail(row: int, column: str, problem: str) -> None:
        raise DatasetError(f"{name}, line {row}, column {column}: {problem}")

    empty = frame.index[frame["record_id"] == ""]
    if len(empty):
        fail(empty[0], "record_id", "is empty.")

    out = pd.DataFrame(index=frame.index)
    out["record_id"] = frame["record_id"]
    group = frame["group"].map(lambda g: _BY_KEY.get(_group_key(g), pd.NA) if g else UNASSIGNED)
    bad = group.index[group.isna()]
    if len(bad):
        fail(
            bad[0],
            "group",
            f"{frame.at[bad[0], 'group']!r} is not a group. Use one of {', '.join(GROUPS)}, "
            "or leave it blank for unassigned.",
        )
    out["group"] = group

    tier = frame["tier"].str.upper() if "tier" in frame else pd.Series("", index=frame.index)
    bad = tier.index[(tier != "") & ~tier.isin(TIERS)]
    if len(bad):
        fail(bad[0], "tier", f"{frame.at[bad[0], 'tier']!r} is not one of {', '.join(TIERS)}.")
    out["tier"] = tier.replace("", DEFAULT_TIER)

    drug = frame["drug"] if "drug" in frame else pd.Series("", index=frame.index)
    out["drug"] = drug.replace("", pd.NA).astype("object")

    if "months_since_stopping" in frame:
        months = _numbers(frame["months_since_stopping"], "months_since_stopping", fail)
        bad = months.index[months < 0]
        if len(bad):
            fail(
                bad[0],
                "months_since_stopping",
                f"{frame.at[bad[0], 'months_since_stopping']!r} is not 0 or more.",
            )
        out["months_since_stopping"] = months
    else:
        out["months_since_stopping"] = float("nan")
    for column in scores:
        out[column] = _numbers(frame[column], column, fail)
    return out, scores


def _numbers(text: pd.Series, column: str, fail) -> pd.Series:
    numbers = pd.to_numeric(text.where(text != ""), errors="coerce").astype("float64")
    bad = text.index[(text != "") & ~numbers.map(lambda x: x == x and abs(x) != float("inf"))]
    if len(bad):
        fail(bad[0], column, f"{text.at[bad[0]]!r} is not a number.")
    return numbers


def _check_agree(meta: pd.DataFrame, values: list[str], name: str) -> None:
    for record, rows in meta[meta["record"].duplicated(keep=False)].groupby("record", sort=False):
        first = rows.iloc[0]
        for line, row in rows.iloc[1:].iterrows():
            for column in values:
                a, b = first[column], row[column]
                if (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and a == b):
                    continue
                written = set(rows["record_id"])
                shown = next(iter(written)) if len(written) == 1 else record
                raise DatasetError(
                    f"{name}: lines {rows.index[0]} and {line} give record {shown} different {column} "
                    "values. Keep one row per report."
                )
