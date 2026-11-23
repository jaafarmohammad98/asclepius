"""The built-in examples: one invented dataset of each kind, for trying the tool before real data is
at hand (the web app's "Try an example").

Both are written on demand by `asclepius.synthetic` with fixed settings and seeds, so every install
gets byte-identical files and the same findings. Nothing in them is real: no patient's report and
nothing from the r/PSSD export, which is never shipped with the tool.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from asclepius import synthetic

SEED = 1


@dataclass(frozen=True)
class Example:
    description: str  # shown with the review: what it is and what was planted
    write: Callable[[Path], dict[str, list[Path]]]  # into a folder; role -> files, as the page picks them


def _lab_reports(folder: Path) -> dict[str, list[Path]]:
    out = synthetic.write_lab_reports(
        folder,
        groups={"PSSD": 40, "healthy control": 40},
        planted=[synthetic.PlantedDifference("androsterone", 0.6)],
        tier_a_share=0.5,  # enough tier A reports for the finding to be re-run there, and be strong
        seed=SEED,
    )
    return {"reports": [out.reports], "metadata": [out.metadata]}


def _self_reports(folder: Path) -> dict[str, list[Path]]:
    out = synthetic.write_self_reports(
        folder,
        authors=500,
        associations=[synthetic.PlantedAssociation("finasteride", "erectile_dysfunction", 5)],
        outcomes=[synthetic.PlantedOutcome("tadalafil", 0.6, 0.05)],
        seed=SEED,
    )
    return {"export": sorted(out.export.glob("*.csv"))}


EXAMPLES = {
    "lab-report": Example(
        "Example dataset, invented by the tool: 40 PSSD and 40 healthy-control DUTCH reports, with "
        "androsterone planted 40% lower in PSSD and half the reports tier A. No real patient is in it.",
        _lab_reports,
    ),
    "self-report": Example(
        "Example dataset, invented by the tool: 500 authors in the harvester's format, with erectile "
        "dysfunction planted at 5 times the odds after finasteride and tadalafil reported to help 60% "
        "of those who tried it. No real post is in it.",
        _self_reports,
    ),
}
