"""Evidence for each finding: what the library says, gathered before any LLM is asked (spec H2, H5, H7).

The input is the analysis findings export (the web app's `findings.csv`, `asclepius.web.export`),
of a lab-report or a self-report run. `read_findings` keeps the finding rows and, of each, the
group statistics only: no column outside `REQUIRED`, `COUNTS`, `STATISTICS` and `TEXTS` is read, so a record id or an
individual value cannot reach a bundle even if a later export carried one.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Protocol

from hypothesis_engine.search import Hit

# The export's group statistics: sizes, medians, shares, effect sizes and their intervals, tests,
# the Tier A re-run. Counts below 5 are already hidden in the export (analysis spec S11).
COUNTS = ("n_a", "n_b", "tier_a_n_a", "tier_a_n_b")
STATISTICS = (
    "median_a",
    "median_b",
    "share_a",
    "share_b",
    "effect_size",
    "ci_low",
    "ci_high",
    "odds_ratio",
    "odds_ratio_low",
    "odds_ratio_high",
    "p_value",
    "q_value",
    "tier_a_effect_size",
    "tier_a_ci_low",
    "tier_a_ci_high",
)
TEXTS = ("tier_a_result",)
# the analysis labels a finding worth testing has (spec S5); noise-level and descriptions are not
STRONG, SUGGESTIVE = "strong", "suggestive"
TESTED = frozenset({STRONG, SUGGESTIVE})
SELF_REPORT = "self-report, no controls"  # on every self-report finding (analysis spec S10, H7)


SUPPORTING, CONTRADICTING = "supporting", "contradicting"

# A group's name as papers write it, and the drug behind a syndrome (analysis spec, Input).
SYNDROMES = {
    "PSSD": ("post-SSRI sexual dysfunction", "SSRI antidepressants"),
    "PFS": ("post-finasteride syndrome", "finasteride"),
    "PAS": ("post-isotretinoin syndrome", "isotretinoin"),
}
COMPARATORS = {
    "healthy control": "healthy controls",
    "SSRI-exposed control": "SSRI users without lasting symptoms",
}
# The enzyme an enzyme-activity ratio stands for (analysis spec A2; asclepius.ratios.RATIOS, worded
# as papers name the enzyme; a ratio added there needs a line here), and whether the ratio rises
# with its activity. Free cortisol / free cortisone falls as 11-beta-HSD2 turns cortisol into cortisone.
ENZYMES = {
    "androsterone/etiocholanolone": ("5-alpha reductase", True),
    "5a_androstanediol/5b_androstanediol": ("5-alpha reductase", True),
    "a_tetrahydrocortisol_a_thf/b_tetrahydrocortisol_b_thf": ("5-alpha reductase", True),
    "a_pregnanediol/b_pregnanediol": ("5-alpha reductase", True),
    "24hr_free_cortisol/24hr_free_cortisone": ("11-beta-hydroxysteroid dehydrogenase type 2", False),
    "2_methoxy_e1/2_oh_e1": ("COMT methylation", True),
    "2_oh_e1/16_oh_e1": ("estrogen 2-hydroxylation over 16-hydroxylation", True),
    "2_oh_e1/4_oh_e1": ("estrogen 2-hydroxylation over 4-hydroxylation", True),
}
RATIO = "enzyme-activity ratio"
# Group names that name no one a paper would write about: "cluster 2", "other causal drugs".
UNNAMED = ("cluster ", "other ")


class Searcher(Protocol):
    """What gathering needs of the search index (`hypothesis_engine.search.Index`)."""

    library_version: int
    model: str

    def search(self, query: str, k: int = 10) -> list[Hit]: ...


@dataclass(frozen=True)
class Passage:
    """A library passage offered as evidence, with the search that found it best."""

    passage_id: str  # "<paper id>:<n>", readable back with `Index.passage`
    paper_id: str  # a PMID, or a private paper's id
    private: bool  # from the researcher's private folder: never sent whole (`privacy.py`)
    title: str
    text: str
    query: str
    score: float  # reciprocal rank fusion score for that search


@dataclass(frozen=True)
class Finding:
    """One finding of the export, group statistics only."""

    section: str
    comparison: str
    marker_id: str
    marker_name: str
    sex: str | None
    group_a: str | None
    group_b: str | None
    label: str | None
    statistics: dict[str, int | float | str | None]  # COUNTS, STATISTICS and TEXTS, by column
    labels: tuple[str, ...]  # the export's labels: "self-report, no controls", "provisional, ..."
    tool: str  # the analysis tool and version that found it
    dataset_fingerprint: str
    run_date: str | None

    def as_dict(self) -> dict[str, object]:
        """The finding as plain JSON-ready values, its statistics beside its names."""
        return {
            "section": self.section,
            "comparison": self.comparison,
            "marker_id": self.marker_id,
            "marker_name": self.marker_name,
            "sex": self.sex,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "label": self.label,
            **self.statistics,
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """What the library says about one finding: candidates only. A passage's stance is the stance of
    the search that ranked it best; whether it really supports or contradicts is for the reader
    (and, from M4-05b, the LLM) to judge."""

    finding: Finding
    queries: tuple[Query, ...]
    supporting: tuple[Passage, ...]
    contradicting: tuple[Passage, ...]
    library_version: int
    search_model: str

    @property
    def self_report(self) -> bool:
        """From a self-report dataset: its hypothesis must say so and name a test that does not
        rely on self-reports (spec H7)."""
        return SELF_REPORT in self.finding.labels

    @property
    def labels(self) -> tuple[str, ...]:
        """The finding's labels, "self-report, no controls" first when it applies."""
        rest = tuple(label for label in self.finding.labels if label != SELF_REPORT)
        return ((SELF_REPORT,) if self.self_report else ()) + rest

    def as_dict(self) -> dict[str, object]:
        """The bundle as plain JSON-ready values."""
        return {
            "finding": self.finding.as_dict(),
            "self_report": self.self_report,
            "labels": list(self.labels),
            "tool": self.finding.tool,
            "dataset_fingerprint": self.finding.dataset_fingerprint,
            "run_date": self.finding.run_date,
            "library_version": self.library_version,
            "search_model": self.search_model,
            "queries": [{"text": q.text, "stance": q.stance} for q in self.queries],
            "supporting": [asdict(p) for p in self.supporting],
            "contradicting": [asdict(p) for p in self.contradicting],
        }


REQUIRED = (
    "row",
    "section",
    "comparison",
    "marker_id",
    "marker_name",
    "sex",
    "group_a",
    "group_b",
    "label",
    "labels",
    "tool",
    "dataset_fingerprint",
)


class FindingsError(Exception):
    """The file is not an analysis findings export."""


def read_findings(path: str | PathLike[str]) -> list[Finding]:
    """The finding rows of a findings export, in its order. Refuses another file with
    `FindingsError`, naming the columns it lacks, never what it holds."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or ())]
        if missing:
            raise FindingsError(
                f"{path} is not an analysis findings export (the web app's findings.csv): "
                f"no {', '.join(missing)} column"
            )
        rows = list(reader)
    return [_finding(r) for r in rows if r["row"] == "finding"]


def _number(cell: str, kind: type[int] | type[float]) -> int | float | None:
    """A count or statistic; an empty cell, or a count the export hid, is None."""
    try:
        return kind(cell) if cell else None
    except ValueError:
        return None


def _finding(row: dict[str, str]) -> Finding:
    statistics: dict[str, int | float | str | None] = {c: _number(row.get(c, ""), int) for c in COUNTS}
    statistics |= {c: _number(row.get(c, ""), float) for c in STATISTICS}
    statistics |= {c: row.get(c) or None for c in TEXTS}
    return Finding(
        section=row["section"],
        comparison=row["comparison"],
        marker_id=row["marker_id"],
        marker_name=row["marker_name"],
        sex=row["sex"] or None,
        group_a=row["group_a"] or None,
        group_b=row["group_b"] or None,
        label=row["label"] or None,
        statistics=statistics,
        labels=tuple(label.strip() for label in row["labels"].split(";") if label.strip()),
        tool=row["tool"],
        dataset_fingerprint=row["dataset_fingerprint"],
        run_date=row.get("run_date") or None,
    )


@dataclass(frozen=True)
class Query:
    """One search of the library for a finding, and the stance a passage it finds is filed under."""

    text: str
    stance: str  # SUPPORTING: what a passage agreeing with the finding would say; CONTRADICTING: the opposite


def direction(finding: Finding) -> int:
    """1 when group a is higher (or reports more often), -1 when lower, 0 when the export does not say."""
    stats = finding.statistics
    effect = stats.get("effect_size")
    if isinstance(effect, float) and effect:
        return 1 if effect > 0 else -1
    for a, b in (("median_a", "median_b"), ("share_a", "share_b")):
        x, y = stats.get(a), stats.get(b)
        if isinstance(x, float) and isinstance(y, float) and x != y:
            return 1 if x > y else -1
    return 0


def _named(group: str | None) -> str | None:
    if not group or group.startswith(UNNAMED):
        return None
    return SYNDROMES[group][0] if group in SYNDROMES else COMPARATORS.get(group, group)


def queries_for(finding: Finding) -> list[Query]:
    """The searches for a finding: what a supporting passage would say, and what a contradicting
    one would (the opposite direction, or no difference). Built from the marker, the enzyme a
    ratio stands for, the direction, the syndrome and its drug, or the drug and symptom of a
    self-report finding. Only names and directions go in, never a number."""
    sign = direction(finding)
    if not sign:
        return []
    if is_report(finding):
        return _report_queries(finding, sign)
    return _marker_queries(finding, sign)


def is_report(finding: Finding) -> bool:
    """A finding about how often a symptom, onset or course is reported (shares), not a marker's level."""
    return finding.statistics.get("median_a") is None and finding.statistics.get("share_a") is not None


def _marker_queries(finding: Finding, direction: int) -> list[Query]:
    word = {1: "higher", -1: "lower"}
    marker = finding.marker_name
    a, b = _named(finding.group_a), _named(finding.group_b)
    queries: list[Query] = []
    for stance, sign in ((SUPPORTING, direction), (CONTRADICTING, -direction)):
        queries.append(Query(f"{word[sign]} {marker} levels", stance))  # so a syndrome's name cannot drown it
        if a and b:
            queries.append(Query(f"{word[sign]} {marker} in {a} than in {b}", stance))
        elif a:
            queries.append(Query(f"{word[sign]} {marker} in {a}", stance))
        else:
            queries.append(
                Query(f"{word[sign]} {marker} in a subgroup with lasting symptoms after a drug", stance)
            )
        if finding.group_a in SYNDROMES:
            queries.append(Query(f"{word[sign]} {marker} after {SYNDROMES[finding.group_a][1]}", stance))
        if finding.comparison == RATIO and finding.marker_id in ENZYMES:
            enzyme, rises = ENZYMES[finding.marker_id]
            where = f" in {a}" if a else ""
            queries.append(Query(f"{word[sign if rises else -sign]} {enzyme} activity{where}", stance))
    if a and b:
        queries.append(Query(f"no difference in {marker} between {a} and {b}", CONTRADICTING))
    return queries


def _report_queries(finding: Finding, direction: int) -> list[Query]:
    """A self-report finding: how often authors who took a drug report a symptom, an onset or a
    course (`onset:` and `trajectory:` rows, which are not symptoms: "recovered" is no lasting harm)."""
    word = {1: "more", -1: "less"}
    kind = finding.marker_id.split(":", 1)[0]
    what = {"onset": "symptom ", "trajectory": "symptoms "}.get(kind, "") + finding.marker_name
    drug = _named(finding.group_a)
    if drug is None:  # a symptom subgroup: the cluster has no name, what it differs in has
        where = "in a subgroup with lasting symptoms"
    else:
        where = f"after {drug} than after {_named(finding.group_b) or 'other drugs'}"
    queries = [
        Query(f"{what} {word[sign]} often {where}", stance)
        for stance, sign in ((SUPPORTING, direction), (CONTRADICTING, -direction))
    ]
    if drug is not None and kind == "symptom":
        stance = SUPPORTING if direction > 0 else CONTRADICTING
        queries.append(Query(f"persistent {what} after stopping {drug}", stance))
    return queries


def worth_testing(finding: Finding) -> bool:
    """A strong or suggestive finding; noise-level ones and descriptions get no bundle (spec H2)."""
    return finding.label in TESTED


def gather_evidence(findings: Iterable[Finding], index: Searcher, k: int = 5) -> list[EvidenceBundle]:
    """One bundle per finding worth testing, in the findings' order, with up to `k` candidate
    passages each way. Each passage is offered once: under the stance whose best search scored it
    higher, and on a tie as contradicting, so a passage that may contradict is never filed as support."""
    bundles = []
    for finding in findings:
        if not worth_testing(finding):
            continue
        queries = queries_for(finding)
        best: dict[str, dict[str, Passage]] = {SUPPORTING: {}, CONTRADICTING: {}}
        for query in queries:
            for hit in index.search(query.text, k):
                if hit.text.strip() == hit.title.strip():
                    continue  # a paper known only by its title: nothing in it to quote
                found = best[query.stance].get(hit.passage_id)
                if found is None or hit.score > found.score:
                    best[query.stance][hit.passage_id] = Passage(
                        hit.passage_id, hit.paper_id, hit.private, hit.title, hit.text, query.text, hit.score
                    )
        support, against = best[SUPPORTING], best[CONTRADICTING]
        for passage_id in support.keys() & against.keys():
            if support[passage_id].score > against[passage_id].score:
                del against[passage_id]
            else:
                del support[passage_id]
        bundles.append(
            EvidenceBundle(
                finding,
                tuple(queries),
                _best(support.values(), k),
                _best(against.values(), k),
                index.library_version,
                index.model,
            )
        )
    return bundles


def _best(passages: Iterable[Passage], k: int) -> tuple[Passage, ...]:
    return tuple(sorted(passages, key=lambda p: (-p.score, p.passage_id))[:k])
