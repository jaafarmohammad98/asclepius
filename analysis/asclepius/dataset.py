"""Load lab-report dataset files written by the extractor (`npm run extract -- --csv`).

A file is one row per marker per report ("long" format). Every file is checked before anything is
returned: a problem stops the load with a `DatasetError` naming the file, the line and the column.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import pandas as pd

COLUMNS = (
    "record_id",
    "sex",
    "age_years",
    "collection_month",
    "report_month",
    "marker_id",
    "marker_name",
    "category",
    "comparator",
    "value",
    "unit",
    "ref_low",
    "ref_high",
    "status",
)
REQUIRED = ("record_id", "marker_id", "marker_name", "value", "unit")
# The same pattern the extractor reads units with (extractor/src/parse.ts), e.g. ng/mg, µg/mg
UNIT = re.compile(r"^[a-zµμ]+/[a-zµμ]+$", re.IGNORECASE)
SEXES = frozenset({"male", "female"})
COMPARATORS = frozenset({"<", ">"})
MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class DatasetError(ValueError):
    """A dataset file cannot be used. The message says which file, line and column, and why."""


@dataclass(frozen=True)
class LabDataset:
    """Marker values from lab reports. Never pooled with a `SelfReportDataset`.

    `markers` has one row per marker per report. `record` identifies a report across all loaded
    files (`<file number>/<record_id>`, files numbered from 1 in load order), because each extractor
    file numbers its reports from r001. `fingerprint` is a SHA-256 over the input files' bytes
    (metadata file included), so a result can name the exact data it came from. `files` gives
    each input file's role (`lab reports`, `metadata`), name and SHA-256, in load order, for the
    run record (`asclepius.record`).

    `records` has one row per report: `record`, `file`, `record_id`, `sex`, `age_years`, and from
    the metadata file (`asclepius.metadata`) `group`, `tier`, `drug`, `months_since_stopping` and any
    `score_<name>` columns. Every report has a group (`unassigned` without a metadata row) and a
    tier (B unless the metadata says otherwise). `unmatched` lists metadata record ids that name no
    loaded report.
    """

    markers: pd.DataFrame
    fingerprint: str
    records: pd.DataFrame
    unmatched: tuple[str, ...] = ()
    kind: str = "lab-report"
    files: tuple[FileHash, ...] = ()


@dataclass(frozen=True)
class FileHash:
    """One input file of a dataset: what it is (`role`), its name and the SHA-256 of its bytes."""

    role: str
    name: str
    sha256: str


def load_lab_reports(
    paths: Iterable[str | PathLike[str]], metadata: str | PathLike[str] | None = None
) -> LabDataset:
    """Load extractor CSV files, and optionally the metadata file naming each report's group and tier."""
    from asclepius import metadata as meta  # metadata uses DatasetError from this module

    files = [Path(p) for p in paths]
    if not files:
        raise DatasetError("No dataset files were given.")
    digest = hashlib.sha256()
    frames = []
    hashes = []
    for number, path in enumerate(files, start=1):
        raw = path.read_bytes()
        digest.update(hashlib.sha256(raw).digest())
        hashes.append(FileHash("lab reports", path.name, hashlib.sha256(raw).hexdigest()))
        frame = _read_file(path)
        frame.insert(0, "file", number)
        frame.insert(0, "record", str(number) + "/" + frame["record_id"])
        frames.append(frame)
    markers = pd.concat(frames, ignore_index=True)
    _check_units_agree(markers, files)
    records = meta.default_records(markers)
    unmatched: tuple[str, ...] = ()
    if metadata is not None:
        meta_path = Path(metadata)
        if not meta_path.is_file():
            raise DatasetError(f"{meta_path.name}: the file does not exist.")
        meta_hash = hashlib.sha256(meta_path.read_bytes())
        digest.update(b"metadata")
        digest.update(meta_hash.digest())
        hashes.append(FileHash("metadata", meta_path.name, meta_hash.hexdigest()))
        records, unmatched = meta.apply(records, meta_path)
    return LabDataset(
        markers=markers,
        fingerprint=digest.hexdigest(),
        records=records,
        unmatched=unmatched,
        files=tuple(hashes),
    )


def _read_file(path: Path) -> pd.DataFrame:
    name = path.name
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=True)
    except pd.errors.EmptyDataError:
        raise DatasetError(f"{name}: the file is empty.") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as err:
        raise DatasetError(f"{name}: not a readable CSV file ({err}).") from None

    missing = [c for c in COLUMNS if c not in frame.columns]
    unknown = [c for c in frame.columns if c not in COLUMNS]
    if missing:
        raise DatasetError(f"{name}: missing column(s) {', '.join(missing)}. Is this an extractor CSV file?")
    if unknown:
        raise DatasetError(f"{name}: unexpected column(s) {', '.join(unknown)}.")
    if frame.empty:
        raise DatasetError(f"{name}: the file has a header but no rows.")

    frame = frame[list(COLUMNS)].apply(lambda col: col.str.strip())
    # line in the file, counting the header as line 1 (blank lines are only allowed at the end)
    lines = frame.index + 2

    def fail(row: int, column: str, problem: str) -> None:
        raise DatasetError(f"{name}, line {lines[row]}, column {column}: {problem}")

    for column in REQUIRED:
        empty = frame.index[frame[column] == ""]
        if len(empty):
            fail(empty[0], column, "is empty.")

    _check_allowed(frame, "sex", SEXES, fail)
    _check_allowed(frame, "comparator", COMPARATORS, fail)
    bad = frame.index[~frame["unit"].str.match(UNIT)]
    if len(bad):
        fail(bad[0], "unit", f"{frame.at[bad[0], 'unit']!r} is not a unit written like ng/mg.")
    for column in ("collection_month", "report_month"):
        bad = frame.index[(frame[column] != "") & ~frame[column].str.match(MONTH)]
        if len(bad):
            fail(bad[0], column, f"{frame.at[bad[0], column]!r} is not a month written as YYYY-MM.")

    out = frame.copy()
    for column in ("value", "ref_low", "ref_high"):
        out[column] = _numbers(frame, column, fail)
    age = _numbers(frame, "age_years", fail)
    bad = age.index[age.notna() & ((age % 1 != 0) | (age < 0) | (age > 130))]
    if len(bad):
        fail(bad[0], "age_years", f"{frame.at[bad[0], 'age_years']!r} is not an age in whole years.")
    out["age_years"] = age.astype("Int64")

    both = out["ref_low"].notna() & out["ref_high"].notna()
    bad = out.index[both & (out["ref_low"] > out["ref_high"])]
    if len(bad):
        fail(bad[0], "ref_low", "the reference range's low end is above its high end.")

    # every value is readable; now check the rows of each report agree
    _check_one_row_per_marker(frame, lines, fail)
    for column in ("sex", "age_years"):
        _check_one_per_report(frame, column, lines, fail)

    for column in ("sex", "collection_month", "report_month", "category", "comparator", "status"):
        out[column] = frame[column].replace("", pd.NA)
    return out[~frame.duplicated()]  # an identical repeated row counts once


def _check_one_row_per_marker(frame: pd.DataFrame, lines: pd.Index, fail) -> None:
    """A report may repeat a row exactly, but not give one marker two different rows."""
    seen: dict[tuple[str, str], int] = {}
    for row in frame.index:
        key = (frame.at[row, "record_id"], frame.at[row, "marker_id"])
        first = seen.setdefault(key, row)
        if first == row:
            continue
        differs = [c for c in COLUMNS if frame.at[row, c] != frame.at[first, c]]
        if differs:
            fail(row, differs[0], f"{key[0]} already has a row for {key[1]} (line {lines[first]}).")


def _check_one_per_report(frame: pd.DataFrame, column: str, lines: pd.Index, fail) -> None:
    """Every row of a report gives the same `column`, or leaves it blank."""
    given = frame[frame[column] != ""]
    first = given.groupby("record_id", sort=False)[column].transform("first")
    bad = given.index[given[column] != first]
    if len(bad):
        row = bad[0]
        record = frame.at[row, "record_id"]
        earlier = given.index[(given["record_id"] == record)][0]
        fail(row, column, f"{record} is {frame.at[earlier, column]} on line {lines[earlier]}.")


def _check_allowed(frame: pd.DataFrame, column: str, allowed: frozenset[str], fail) -> None:
    bad = frame.index[(frame[column] != "") & ~frame[column].isin(allowed)]
    if len(bad):
        fail(bad[0], column, f"{frame.at[bad[0], column]!r} is not one of {', '.join(sorted(allowed))}.")


def _numbers(frame: pd.DataFrame, column: str, fail) -> pd.Series:
    text = frame[column]
    numbers = pd.to_numeric(text.where(text != ""), errors="coerce")
    bad = text.index[(text != "") & (numbers.isna() | ~numbers.map(_finite))]
    if len(bad):
        fail(bad[0], column, f"{text.at[bad[0]]!r} is not a number.")
    return numbers.astype("float64")


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def _check_units_agree(markers: pd.DataFrame, paths: list[Path]) -> None:
    units = markers.groupby("marker_id")["unit"].unique()
    mixed = units[units.map(len) > 1]
    if mixed.empty:
        return
    marker = mixed.index[0]
    rows = markers[markers["marker_id"] == marker]
    found = "; ".join(
        f"{unit} in {', '.join(sorted({paths[n - 1].name for n in rows.loc[rows['unit'] == unit, 'file']}))}"
        for unit in sorted(mixed.iloc[0])
    )
    raise DatasetError(
        f"Marker {marker} is recorded in more than one unit ({found}). "
        "Convert the files to one unit before loading them together."
    )
