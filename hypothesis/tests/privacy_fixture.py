"""Write tests/fixtures/privacy/: two synthetic datasets and the findings the analysis app exports.

The privacy test (`test_outgoing_requests.py`, M4-08) runs the hypothesis engine on these findings
and looks in every request for the datasets' record ids and individual values. To make a leak
show, both are distinctive: record ids read `pt-...-x`, and every lab value carries five digits of
its own after the decimal point (`1140.0000123`), which no group statistic has by chance.

Run it from `hypothesis/` with the analysis engine's environment (it is not one of this
package's dependencies), after a change to either engine's findings:

    uv run --project ../analysis python tests/privacy_fixture.py
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

from asclepius import synthetic
from asclepius.run import Settings
from asclepius.web.workspace import Workspace

OUT = Path(__file__).parent / "fixtures" / "privacy"
SETTINGS = Settings(permutations=999, bootstrap=200)


def distinctive(value: float, k: int) -> str:
    """`value` to 2 decimals, then the 5 digits of `k` (1 to 99999): unique to one value."""
    return f"{value:.2f}{k:05d}"


def lab(out: Path) -> None:
    """Men only, 24 a group: every group and sex has an even size, so no median is one record's value."""
    written = synthetic.write_lab_reports(
        out,
        groups={"PSSD": 24, "healthy control": 24},
        female_share=0,
        planted=[
            synthetic.PlantedDifference("androsterone", 0.5),
            synthetic.PlantedDifference("etiocholanolone", 1.6),
        ],
        seed=3,
    )
    with written.reports.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for k, row in enumerate(rows, start=1):
        row["record_id"] = _record_id(row["record_id"])
        row["value"] = distinctive(float(row["value"]), k)
        value, low, high = float(row["value"]), float(row["ref_low"]), float(row["ref_high"])
        row["status"] = "Below range" if value < low else "Above range" if value > high else "Within range"
    _write(written.reports, rows)
    with written.metadata.open(encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    for row in meta:
        row["record_id"] = _record_id(row["record_id"])
    _write(written.metadata, meta)
    written.truth.unlink()

    workspace = Workspace(out.parent / "work")
    workspace.add_file("reports", "reports.csv", written.reports.read_bytes())
    workspace.add_file("metadata", "metadata.csv", written.metadata.read_bytes())
    workspace.load()
    workspace.start_run("PSSD", "healthy control", settings=SETTINGS)
    _finish(workspace)
    (out.parent / "lab_findings.csv").write_text(workspace.export_csv(), encoding="utf-8")


def self_report(out: Path) -> None:
    """300 authors, one planted drug-symptom association and one intervention outcome."""
    written = synthetic.write_self_reports(
        out,
        authors=300,
        associations=[synthetic.PlantedAssociation("paroxetine", "genital_numbness", 4.0)],
        outcomes=[synthetic.PlantedOutcome("bupropion", 0.5, 0.1)],
        seed=3,
    )
    written.truth.unlink()
    labs = written.export / "labs.csv"
    with labs.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for k, row in enumerate(rows, start=1):
        if row["value"]:
            row["value"] = distinctive(float(row["value"]), k)
    _write(labs, rows)

    workspace = Workspace(out.parent / "work")
    for path in sorted(written.export.glob("*.csv")):
        workspace.add_file("export", path.name, path.read_bytes())
    workspace.load()
    workspace.start_run(settings=SETTINGS)
    _finish(workspace)
    (out.parent / "self_report_findings.csv").write_text(workspace.export_csv(), encoding="utf-8")


def _record_id(synthetic_id: str) -> str:
    return f"pt-{synthetic_id[1:]}-x"  # r007 -> pt-007-x


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _finish(workspace: Workspace) -> None:
    while (status := workspace.status())["state"] == "running":
        time.sleep(0.1)
    if status["state"] != "finished":
        sys.exit(f"the analysis run did not finish: {status}")


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    lab(OUT / "lab")
    self_report(OUT / "self_report")
    shutil.rmtree(OUT / "work")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
