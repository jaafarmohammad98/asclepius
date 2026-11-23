"""What the web app does, apart from HTTP: hold the files a researcher picked, load them, sum them
up for review and run the full analysis in the background, step by step.

The files are copied into a folder of the workspace's own, so the loaders read the exact bytes
picked and the fingerprint matches a run made from the command line on the same files.

A finished run's findings are added to the app's knowledge graph (`asclepius.web.graph`), and a new
load, run or clear drops them, as it drops the hypotheses.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
from hypothesis_engine.evidence import Finding, FindingsError, read_findings

from asclepius.cluster import ClusterResult, TooFewPatients, cluster_patients
from asclepius.dataset import DatasetError, LabDataset, load_lab_reports
from asclepius.examples import EXAMPLES
from asclepius.interventions import rank_interventions
from asclepius.metadata import GROUPS
from asclepius.profile import profile_drugs
from asclepius.record import run_record
from asclepius.refrange import lab_range_findings, self_report_range_findings
from asclepius.run import GroupTooSmall, Settings, run_analysis
from asclepius.selfreport import SelfReportDataset, load_self_reports
from asclepius.subgroups import SubgroupLinks, link_subgroups
from asclepius.symptom_subgroups import cluster_symptoms
from asclepius.syndromes import CONTROLS, REFERENCE_RANGE, SYNDROMES, compare_syndromes
from asclepius.web import export, findings
from asclepius.web.findings import (
    COMPARISON,
    FEW,
    INTERVENTIONS,
    PROFILE,
    RANGES,
    SELF_REPORT_RANGES,
    SUBGROUPS,
    SYMPTOM_SUBGROUPS,
    findings_view,
    shown,
)
from asclepius.web.findings import SYNDROMES as SYNDROMES_KIND
from asclepius.web.graph import KnowledgeGraph, Run
from asclepius.web.hypotheses import Busy, Engine, HypothesisRun

NO_SEX = "not given"
ROLES = ("reports", "metadata", "export")  # extractor CSVs, the metadata file, a harvester export's tables


# a step's state: waiting, then running, then one of the last four
WAITING, RUNNING = "waiting", "running"
DONE, REFUSED, FAILED, NOT_BUILT = "done", "refused", "failed", "not built yet"
IDLE, FINISHED = "idle", "finished"  # a run's state, with RUNNING
FAILED_EXPORTED = "Unexpected error; the details are shown in the app only."


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


def shown_parts(parts: pd.Series) -> pd.Series:
    """`asclepius.web.findings.shown_parts` for counts in a Series, keeping its order."""
    kept = findings.shown_parts({label: int(n) for label, n in parts.items()})
    return pd.Series([kept[label] for label in parts.index], index=parts.index, dtype=object)


class Workspace:
    """The files picked, the dataset loaded from them, and the latest run."""

    def __init__(
        self, folder: str | Path, engine: Engine | None = None, knowledge: KnowledgeGraph | None = None
    ) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, list[Path]] = {role: [] for role in ROLES}
        self.data: LabDataset | SelfReportDataset | None = None
        self.steps: list[Step] = []
        self.results: dict[str, Any] = {}  # step name -> what it returned, for the findings view
        self.record: dict[str, object] | None = None  # the latest run's record (spec S7)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.hypotheses = HypothesisRun(engine or Engine())  # written from the latest run's findings
        self.knowledge = knowledge  # the graph the latest run's findings are added to, if any

    def clear(self) -> None:
        """Forget the files picked and the dataset loaded from them."""
        with self._lock:
            if self.running():
                raise LoadError("A run is in progress; wait for it to finish.")
            self._forget_hypotheses(LoadError)
            self._drop_findings()
            for role in ROLES:
                shutil.rmtree(self.folder / role, ignore_errors=True)
                self.files[role] = []
            self.data = None
            self.steps, self.results, self.record = [], {}, None

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
            self._forget_hypotheses(LoadError)
            self._drop_findings()
            self.data = None
            self.steps, self.results, self.record = [], {}, None
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

    def load_example(self, kind: str) -> dict[str, object]:
        """Replace the files picked with the built-in example of `kind` (`asclepius.examples`), load
        it, and return the review, which says it is invented (`example`)."""
        example = EXAMPLES[kind]
        source = self.folder / "example"
        shutil.rmtree(source, ignore_errors=True)
        try:  # written before the files picked are forgotten, so a failure keeps them
            written = example.write(source)
            self.clear()
            for role, paths in written.items():
                for path in paths:
                    self.add_file(role, path.name, path.read_bytes())
        finally:
            shutil.rmtree(source, ignore_errors=True)
        return self.load() | {"example": example.description}

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
            self._forget_hypotheses(RunError)
            if isinstance(self.data, SelfReportDataset):
                steps, groups = self_report_steps(self.data, settings), []
            else:
                choices = lab_groups(self.data)
                if group_a not in choices or group_b not in choices:
                    raise RunError(f"Pick two groups from the dataset ({', '.join(choices)}).")
                if group_a == group_b:
                    raise RunError("Pick two different groups to compare.")
                steps = lab_steps(self.data, str(group_a), str(group_b), settings)
                groups = [str(group_a), str(group_b)]
            self.steps, self.results = steps, {}
            self.record = run_record(self.data, groups, settings.record())
            graph_run = self.knowledge.expect(self.data.fingerprint) if self.knowledge else None
            self._thread = threading.Thread(target=self._run, args=(graph_run,), daemon=True)
            self._thread.start()

    def _run(self, graph_run: Run | None = None) -> None:
        try:
            self._steps()
        finally:  # the graph says it is updating until it has the run's findings, even none
            if self.knowledge is not None and graph_run is not None:
                # on a thread of its own, so the run has ended once its last step has
                threading.Thread(target=self._add_to_graph, args=(graph_run,), daemon=True).start()

    def _add_to_graph(self, graph_run: Run) -> None:
        """Hand the run's findings, as its export holds them (`hypothesis_engine.evidence.read_findings`),
        to the knowledge graph. A load or run since then has made them another run's, and the graph
        ignores them."""
        assert self.knowledge is not None
        findings: list[Finding] = []
        file, name = tempfile.mkstemp(suffix=".csv", dir=self.folder)
        os.close(file)
        path = Path(name)
        try:
            path.write_text(export.findings_csv(*self._snapshot()), encoding="utf-8")
            findings = read_findings(path)
        except (RunError, FindingsError, OSError):  # dropped by a load meanwhile, or not exportable
            pass
        finally:
            path.unlink(missing_ok=True)
            self.knowledge.add(graph_run, findings)

    def _steps(self) -> None:
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

    def export_record(self) -> dict[str, object]:
        """The finished run's record, as the exports carry it (`exported`)."""
        return self._exported()[1]

    def export_csv(self) -> str:
        """The finished run's findings as CSV (`asclepius.web.export`)."""
        return export.findings_csv(*self._exported())

    def export_report(self) -> str:
        """The finished run's report as one HTML file (`asclepius.web.export`)."""
        return export.report_html(*self._exported())

    def _exported(self) -> tuple[dict[str, object], dict[str, object]]:
        """The finished run's findings view and record, taken together so both are of one run.

        The record is the run's (`asclepius.record.run_record`) with `steps`: each step's name,
        state and message, and the record of the analysis it ran, if it has one. A failed step's
        message is the error's own text, which can name a file or a value, so the exports say only
        that it failed. Raises `RunError` until a run has finished."""
        with self._lock:
            if self.record is None or self.running():
                raise RunError("Run the analysis, and let it finish, before exporting.")
        return self._snapshot()

    def _snapshot(self) -> tuple[dict[str, object], dict[str, object]]:
        """`_exported`, without checking the run has finished (its own thread adds it to the graph
        as it ends)."""
        with self._lock:
            if self.record is None:
                raise RunError("Run the analysis, and let it finish, before exporting.")
            steps = [
                replace(s, message=FAILED_EXPORTED) if s.state == FAILED else replace(s) for s in self.steps
            ]
            data, results, record = self.data, dict(self.results), self.record
        entries = []
        for step in steps:
            entry: dict[str, object] = {"name": step.name, "state": step.state, "message": step.message}
            made = step_record(results.get(step.name))
            if made is not None:
                entry["record"] = made
            entries.append(entry)
        return findings_view(data, steps, results), {**record, "steps": entries}

    def _forget_hypotheses(self, error: type[ValueError]) -> None:
        """The hypotheses are of the run they were written from: a new dataset or run drops them."""
        try:
            self.hypotheses.reset()
        except Busy as busy:
            raise error(str(busy)) from None

    def _drop_findings(self) -> None:
        """A new dataset, run or clear drops the latest run's findings from the knowledge graph."""
        if self.knowledge is not None:
            self.knowledge.drop()

    def start_hypotheses(self) -> None:
        """Write hypotheses from the finished run's findings, in the background (`asclepius.web.hypotheses`);
        `hypotheses_status()` follows them. Raises `RunError` until a run has finished."""
        view, record = self._exported()
        try:  # the run checks it is not already writing, and starts, under its own lock
            self.hypotheses.start(export.findings_csv(view, record), record, self.folder / "hypotheses")
        except Busy as busy:
            raise RunError(str(busy)) from None

    def hypotheses_status(self) -> dict[str, object]:
        """The hypotheses' state and progress (`HypothesisRun.status`)."""
        return self.hypotheses.status()

    def hypotheses_view(self) -> dict[str, object]:
        """The hypotheses as the page shows them (`HypothesisRun.view`)."""
        return self.hypotheses.view()

    def export_hypotheses(self) -> str:
        """The finished hypotheses as one HTML file, with the run records. Raises `RunError` before."""
        try:
            return self.hypotheses.report()
        except ValueError as error:
            raise RunError(str(error)) from None

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


def self_report_steps(data: SelfReportDataset, settings: Settings) -> list[Step]:
    """The self-report analyses."""

    def profile() -> tuple[Any, str]:
        result = profile_drugs(data, settings)
        return result, result.statement

    def ranges() -> tuple[Any, str]:
        found = self_report_range_findings(data)
        return found, _placed(found, "author and test pairs")

    def interventions() -> tuple[Any, str]:
        result = rank_interventions(data, settings)
        return result, result.statement

    def symptom_subgroups() -> tuple[Any, str]:
        result = cluster_symptoms(data, settings)
        return result, result.statement

    return [
        Step("Typed-in lab values within reference ranges", SELF_REPORT_RANGES, ranges),
        Step("Drug and symptom profile", PROFILE, profile),
        Step("Symptom subgroups", SYMPTOM_SUBGROUPS, symptom_subgroups),
        Step("Intervention outcomes", INTERVENTIONS, interventions),
    ]


def step_record(result: Any) -> dict[str, object] | None:
    """The run record an analysis attached to its result, if it did (the subgroup step's is the
    clustering's; the reference-range positions have none, being descriptions)."""
    if isinstance(result, Subgroups):
        return result.clusters.record
    record = getattr(result, "record", None)
    return record if isinstance(record, dict) else None


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
