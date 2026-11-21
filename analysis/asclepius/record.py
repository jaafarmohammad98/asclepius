"""The run record: what a result was computed from, so it can be reproduced (spec S7).

`run_record` names the dataset (its kind, fingerprint and every input file's role and SHA-256;
not the file names, which could name a patient), the tool version and a hash of its code, the
groups compared, every setting and the date. `run_analysis` attaches it to its
result, and `export_run` writes it beside every exported file. `reproduce` runs a record again on
the same files and refuses when a file or the tool version differs, so its output is identical,
apart from the date.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib.metadata import version
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from asclepius.dataset import FileHash, LabDataset, load_lab_reports
from asclepius.ratios import STATUS
from asclepius.selfreport import SelfReportDataset

if TYPE_CHECKING:
    from asclepius.run import RunResult

TOOL = "asclepius"
RECORD_FILE = "run_record.json"


class ReproductionError(ValueError):
    """A run record cannot be reproduced: an input file or the tool version differs."""


def tool_version() -> str:
    """`asclepius` and its installed version, e.g. `asclepius 0.1.0`."""
    return f"{TOOL} {version(TOOL)}"


def code_hash() -> str:
    """SHA-256 over this package's source files, so code changed without a version bump shows."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _file_entry(file: FileHash) -> dict[str, str]:
    return {"role": file.role, "sha256": file.sha256}


def run_record(
    data: LabDataset | SelfReportDataset, groups: Iterable[str], settings: dict[str, object]
) -> dict[str, object]:
    """The record of one run, as plain JSON-ready values."""
    dataset: dict[str, object] = {
        "kind": data.kind,
        "fingerprint": data.fingerprint,
        "files": [_file_entry(f) for f in data.files],
    }
    if isinstance(data, SelfReportDataset):
        dataset["loader"] = dict(data.settings)
    return {
        "tool": tool_version(),
        "code_sha256": code_hash(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": dataset,
        "groups": list(groups),
        "settings": dict(settings),
        "status": STATUS,
    }


def export_run(result: RunResult, out_dir: str | PathLike[str]) -> Path:
    """Write a run's `findings.csv`, `refused.csv` and `skipped.csv` with its record, `run_record.json`.

    The written record is `result.record` plus `outputs`, the SHA-256 of each CSV file, so every
    exported file can be matched to the record it came with. `out_dir` must not exist yet or be
    empty, so files of two runs are never mixed. Returns `out_dir`.
    """
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out}: the folder is not empty; export each run to a folder of its own.")
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, table in (
        ("findings", result.findings),
        ("refused", result.refused),
        ("skipped", result.skipped),
    ):
        path = out / f"{name}.csv"
        table.to_csv(path, index=False, lineterminator="\n")
        outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {**result.record, "outputs": outputs}
    (out / RECORD_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def reproduce(
    record: dict, paths: Iterable[str | PathLike[str]], metadata: str | PathLike[str] | None = None
) -> RunResult:
    """Run a lab-report run record (from `RunResult.record` or a `run_record.json`) again on its files.

    Refuses with `ReproductionError` when this tool is another version than the record's, or when
    a file is missing, extra or changed, naming the file.
    """
    from asclepius.run import Settings, run_analysis  # run attaches records, so it imports this module

    if record["tool"] != tool_version():
        raise ReproductionError(
            f"The record was made with {record['tool']}, and this is {tool_version()}; "
            "install that version to reproduce it."
        )
    if record["code_sha256"] != code_hash():
        raise ReproductionError(
            f"The record was made with other {TOOL} code than this ({tool_version()}, code "
            f"{record['code_sha256'][:12]}... in the record, {code_hash()[:12]}... here); "
            "check out the code the record was made with to reproduce it."
        )
    data = load_lab_reports(paths, metadata=metadata)
    expected = record["dataset"]["files"]
    for index in range(max(len(expected), len(data.files))):
        want = expected[index] if index < len(expected) else None
        given = data.files[index] if index < len(data.files) else None
        if given is None:
            raise ReproductionError(
                f"The record has {len(expected)} files, and {len(data.files)} were given."
            )
        if want != _file_entry(given):
            raise ReproductionError(
                f"{given.name}: not the file the record was made from (file {index + 1}, "
                f"{given.role}, SHA-256 {given.sha256[:12]}...; the record expects "
                + (f"{want['role']} with SHA-256 {want['sha256'][:12]}...)." if want else "no file here).")
            )
    group_a, group_b = record["groups"]
    return run_analysis(data, group_a, group_b, Settings(**record["settings"]))
