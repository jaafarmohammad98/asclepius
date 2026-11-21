"""What the web app does, apart from HTTP: hold the files a researcher picked, load them, sum them
up for review and run the full analysis in the background, step by step.

The files are copied into a folder of the workspace's own, so the loaders read the exact bytes
picked and the fingerprint matches a run made from the command line on the same files.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from asclepius.cluster import MIN_SHOWN, ClusterResult, TooFewPatients, cluster_patients
from asclepius.dataset import DatasetError, LabDataset, load_lab_reports
from asclepius.metadata import GROUPS
from asclepius.refrange import lab_range_findings, self_report_range_findings
from asclepius.run import GroupTooSmall, Settings, run_analysis
from asclepius.selfreport import SelfReportDataset, load_self_reports
from asclepius.subgroups import SubgroupLinks, link_subgroups
from asclepius.syndromes import CONTROLS, REFERENCE_RANGE, SYNDROMES, compare_syndromes
from asclepius.web.findings import (
    COMPARISON,
    INTERVENTIONS,
    PROFILE,
    RANGES,
    SELF_REPORT_RANGES,
    SUBGROUPS,
    SYMPTOM_SUBGROUPS,
    findings_view,
)
from asclepius.web.findings import SYNDROMES as SYNDROMES_KIND

FEW = f"fewer than {MIN_SHOWN}"  # in place of a count too small to show (spec S11)
NO_SEX = "not given"
ROLES = ("reports", "metadata", "export")  # extractor CSVs, the metadata file, a harvester export's tables


# a step's state: waiting, then running, then one of the last four
WAITING, RUNNING = "waiting", "running"
DONE, REFUSED, FAILED, NOT_BUILT = "done", "refused", "failed", "not built yet"
IDLE, FINISHED = "idle", "finished"  # a run's state, with RUNNING


class LoadError(ValueError):
    """The picked files cannot be loaded. The message says why, in plain words."""


class RunError(ValueError):
    """A run cannot start. The message says why, in plain words."""


class NoPatients(ValueError):
    """Subgroups are looked for among patients, and neither group picked is a patient group."""


REFUSALS = (GroupTooSmall, TooFewPatients, NoPatients)  # the analysis declining, by its rules (spec S4)


@dataclass(frozen=True)
class Subgroups:
    """The subgroup step's result: the clusters, and their links to symptoms and drugs."""

    clusters: ClusterResult
    links: SubgroupLinks


@dataclass
class Step:
    """One part of a run. `work` returns its result and a one-line message, or raises one of
    `REFUSALS`; None means the analysis is not built yet and `message` says which issue adds it."""

    name: str
    kind: str  # what the step does, one of the `asclepius.web.findings` section kinds
    work: Callable[[], tuple[Any, str]] | None
    state: str = WAITING
    message: str = ""


def shown(count: int) -> int | None:
    """`count`, or None when it is below 5 and must not be shown (spec S11)."""
    return int(count) if count >= MIN_SHOWN else None


def shown_parts(parts: pd.Series) -> pd.Series:
    """`parts` of a total that is shown, with every count below 5 hidden (None), and then the next
    smallest too until the hidden ones add up to 0 or at least 5: otherwise the total minus the
    parts shown would give a hidden count back (spec S11)."""
    hidden = parts < MIN_SHOWN
    for label in parts.sort_values(kind="stable").index:
        if hidden[label]:
            continue
        if not 0 < parts[hidden].sum() < MIN_SHOWN:
            break
        hidden[label] = True
    return pd.Series(
        [None if h else int(n) for n, h in zip(parts, hidden, strict=True)], index=parts.index, dtype=object
    )


class Workspace:
    """The files picked, the dataset loaded from them, and the latest run."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, list[Path]] = {role: [] for role in ROLES}
        self.data: LabDataset | SelfReportDataset | None = None
        self.steps: list[Step] = []
        self.results: dict[str, Any] = {}  # step name -> what it returned, for the findings view
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def clear(self) -> None:
        """Forget the files picked and the dataset loaded from them."""
        with self._lock:
            if self.running():
                raise LoadError("A run is in progress; wait for it to finish.")
            for role in ROLES:
                shutil.rmtree(self.folder / role, ignore_errors=True)
                self.files[role] = []
            self.data = None
            self.steps, self.results = [], {}

    def add_file(self, role: str, name: str, content: bytes) -> None:
        """Keep one picked file under its own name (any folder part is dropped), so loading
        messages name it as the researcher knows it."""
        if role not in ROLES:
            raise LoadError(f"Unknown kind of file {role!r}.")
        name = Path(name.replace("\\", "/")).name
        if not name or name.startswith("."):
            raise LoadError(f"{name or 'A file'}: not a file name the app can use.")
        if role == "export":  # the loader reads the tables by name from one folder
            path = self.folder / role / name
            if path in self.files[role]:
                raise LoadError(f"{name} was picked twice for one export folder.")
        else:  # extractor files may share a name, and their order matters
            path = self.folder / role / f"{len(self.files[role]) + 1:03d}" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.files[role].append(path)

    def load(self) -> dict[str, object]:
        """Load the picked files and return the review: what is in the dataset. Raises
        `LoadError`, in plain words, when they cannot be loaded."""
        with self._lock:
            if self.running():
                raise LoadError("A run is in progress; wait for it to finish before loading other files.")
            self.data = None
            self.steps, self.results = [], {}
        reports, metadata, export = (self.files[role] for role in ROLES)
        if not (reports or metadata or export):
            raise LoadError("No files were picked. Pick the dataset files, or a harvester export folder.")
        if export and (reports or metadata):
            raise LoadError(
                "Lab-report files and a self-report export folder were both picked. The two kinds "
                "are never pooled (spec, Input): load one kind at a time."
            )
        if metadata and not reports:
            raise LoadError("A metadata file was picked, but no dataset file for it to describe.")
        if len(metadata) > 1:
            raise LoadError(f"{len(metadata)} metadata files were picked; one dataset takes one.")
        try:
            if export:
                self.data = load_self_reports(self.folder / "export")
            else:
                self.data = load_lab_reports(reports, metadata=metadata[0] if metadata else None)
        except DatasetError as error:
            raise LoadError(str(error)) from error
        return review(self.data)

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_run(
        self, group_a: str | None = None, group_b: str | None = None, *, settings: Settings | None = None
    ) -> None:
        """Start the full analysis of the loaded dataset in the background; `status()` follows it.
        A lab-report run compares `group_a` with `group_b` (self-report runs take no groups)."""
        settings = settings or Settings()
        with self._lock:  # checked and started at once, so two requests cannot both start a run
            if self.data is None:
                raise RunError("Load a dataset before running the analysis.")
            if self.running():
                raise RunError("A run is already in progress.")
            if isinstance(self.data, SelfReportDataset):
                steps = self_report_steps(self.data)
            else:
                choices = lab_groups(self.data)
                if group_a not in choices or group_b not in choices:
                    raise RunError(f"Pick two groups from the dataset ({', '.join(choices)}).")
                if group_a == group_b:
                    raise RunError("Pick two different groups to compare.")
                steps = lab_steps(self.data, str(group_a), str(group_b), settings)
            self.steps, self.results = steps, {}
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        for step in self.steps:
            if step.work is None:
                with self._lock:
                    step.state = NOT_BUILT
                continue
            with self._lock:
                step.state = RUNNING
            try:
                result, message = step.work()
            except REFUSALS as refusal:
                state, message = REFUSED, str(refusal)
            except Exception as error:  # shown on the page rather than lost in the terminal
                state, message = FAILED, f"Unexpected error: {error}"
            else:
                state = DONE
                with self._lock:
                    self.results[step.name] = result
            with self._lock:
                step.state, step.message = state, message

    def findings(self) -> dict[str, object]:
        """The latest run's findings, one section per step (`asclepius.web.findings`). Steps still
        waiting or running have no findings yet."""
        with self._lock:
            steps = [replace(s) for s in self.steps]
            results = dict(self.results)
        return findings_view(self.data, steps, results)

    def status(self) -> dict[str, object]:
        """The latest run: its state (`idle`, `running`, `finished`), and each step's state and message."""
        with self._lock:
            steps = [{"name": s.name, "state": s.state, "message": s.message} for s in self.steps]
        if not steps:
            state = IDLE
        else:
            state = FINISHED if all(s["state"] not in (WAITING, RUNNING) for s in steps) else RUNNING
        out: dict[str, object] = {
            "state": state,
            "steps": steps,
            "done": sum(s["state"] not in (WAITING, RUNNING) for s in steps),
            "total": len(steps),
        }
        if isinstance(self.data, SelfReportDataset):
            out["label"] = self.data.label
        return out


def lab_steps(data: LabDataset, group_a: str, group_b: str, settings: Settings) -> list[Step]:
    """Subgroups of `group_a`, the run comparing the two groups (with the subgroup links, so one
    correction covers both), the cross-syndrome comparison and the reference-range positions."""
    patients = [g for g in (group_a, group_b) if g in SYNDROMES]
    # the subgroup step runs first, so the comparison can correct its links with everything else
    links: list[Any] = [None]

    def subgroups() -> tuple[Any, str]:
        if not patients:
            raise NoPatients(
                f"Subgroups are looked for among patients (PSSD, PFS or PAS), and neither "
                f"{group_a} nor {group_b} is a patient group."
            )
        clusters = cluster_patients(data, groups=patients[:1])
        links[0] = link_subgroups(data, clusters, settings)
        return Subgroups(clusters, links[0]), links[0].statement

    def compare() -> tuple[Any, str]:
        result = run_analysis(data, group_a, group_b, settings, subgroups=links[0])
        return result, result.statement

    controls = [g for g in (group_b, group_a) if g in CONTROLS]
    reference = controls[0] if controls else REFERENCE_RANGE

    def syndromes() -> tuple[Any, str]:
        result = compare_syndromes(data, reference=reference, settings=settings)
        return result, f"{result.statement} (reference: {reference})"

    def ranges() -> tuple[Any, str]:
        found = lab_range_findings(data)
        return found, _placed(found)

    return [
        Step(
            f"Look for subgroups of {patients[0]}" if patients else "Look for subgroups", SUBGROUPS, subgroups
        ),
        Step(f"Compare {group_a} with {group_b}", COMPARISON, compare),
        Step(f"Compare {', '.join(SYNDROMES[:-1])} and {SYNDROMES[-1]}", SYNDROMES_KIND, syndromes),
        Step("Positions within reference ranges", RANGES, ranges),
    ]


def self_report_steps(data: SelfReportDataset) -> list[Step]:
    """The self-report analyses: the ones built so far, and the ones still to come."""

    def ranges() -> tuple[Any, str]:
        found = self_report_range_findings(data)
        return found, _placed(found, "author and test pairs")

    return [
        Step("Typed-in lab values within reference ranges", SELF_REPORT_RANGES, ranges),
        Step("Drug and symptom profile", PROFILE, None, message="added by M3-12"),
        Step("Symptom subgroups", SYMPTOM_SUBGROUPS, None, message="added by M3-13"),
        Step("Intervention outcomes", INTERVENTIONS, None, message="added by M3-14"),
    ]


def _placed(found: Any, what: str = "values") -> str:
    n = shown(len(found.patients))
    return f"{n if n is not None else FEW} {what} placed in their own reference range"


def review(data: LabDataset | SelfReportDataset) -> dict[str, object]:
    """What a loaded dataset holds, for the researcher to check before a run."""
    return self_report_review(data) if isinstance(data, SelfReportDataset) else lab_review(data)


def self_report_review(data: SelfReportDataset) -> dict[str, object]:
    """Authors per condition, and the posts left out, with counts below 5 hidden."""
    order = ["PSSD", "PFS", "PAS", "multiple", "unclear", "none"]
    counts = data.authors["condition"].value_counts()
    counts = counts.reindex(
        [c for c in order if c in counts.index] + [c for c in counts.index if c not in order]
    )
    return {
        "kind": data.kind,
        "label": data.label,
        "authors": shown(len(data.authors)),
        "conditions": [{"condition": c, "authors": n} for c, n in shown_parts(counts).items()],
        "excluded": {reason: shown(n) for reason, n in data.excluded.items()},
        "choices": [],
    }


def lab_review(data: LabDataset) -> dict[str, object]:
    """Records per group, and per group, sex and tier, with counts below 5 hidden."""
    records = data.records.assign(sex=data.records["sex"].fillna(NO_SEX))
    per_group = records["group"].value_counts()
    cells = records.groupby(["group", "sex", "tier"]).size()
    cells_shown = cells.groupby(level="group", group_keys=False).apply(shown_parts)
    return {
        "kind": data.kind,
        "groups": [{"group": g, "records": shown(n)} for g, n in _in_order(per_group).items()],
        "records": [{"group": g, "sex": s, "tier": t, "records": n} for (g, s, t), n in cells_shown.items()],
        "choices": lab_groups(data),
        # 0 stays 0: no metadata row was left out, and no person is counted
        "unmatched": shown(len(data.unmatched)) if data.unmatched else 0,
    }


def lab_groups(data: LabDataset) -> list[str]:
    """The groups a run can compare: those with reports, in the plan's order (not `unassigned`)."""
    present = set(data.records["group"])
    return [g for g in GROUPS if g in present]


def _in_order(counts: pd.Series) -> pd.Series:
    """Groups in the plan's order, `unassigned` last."""
    order = [g for g in GROUPS if g in counts.index] + [g for g in counts.index if g not in GROUPS]
    return counts.reindex(order)
