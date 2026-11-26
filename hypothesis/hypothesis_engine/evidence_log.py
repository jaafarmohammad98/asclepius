"""The Bayesian evidence log: each knowledge-graph node as a hypothesis with a Beta posterior
(spec B1 to B5).

A node starts at Beta(1, 1). Its literature adds 0.25 pseudo-supports per distinct paper behind it
(the node's `papers` in `graph.library_graph`), at most +2. Each logged result adds its study
type's pseudo-count (`STUDY_TYPES`) to the supports or the refutes. The table is provisional and
needs expert review (`PROVISIONAL`).

Every entry names its source (B3): a PMID, kept as `PMID:<digits>`; a DOI, kept as `doi:<doi>` in
lower case (DOIs ignore case); or `UNPUBLISHED`. An entry repeating another's source, node and
direction is dropped (B5). Every unpublished observation shares one source, so those are told apart
by the random `id` each entry is given when logged: importing a log twice adds nothing, and two
observations of the same node are both kept. Agreed 2026-11-26.

The log is a JSON file in the data folder (`LOG_PATH`, `hypothesis_engine.home`), never synced
online. `to_json` and `from_json` export and import it; `merge` adds an imported log's new entries.

`weighted` multiplies each bottleneck score by its node's posterior mean (B4), so the bottleneck map
and the leaderboard re-rank as results are logged. A node with no entries keeps its literature's
posterior, so the order before any entry is already the literature's.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import networkx as nx

from hypothesis_engine.bottlenecks import Bottleneck
from hypothesis_engine.home import DATA

PRIOR = 1.0  # Beta(1, 1)
PER_PAPER = 0.25
LITERATURE_CAP = 2.0
SUPPORTS, REFUTES = "supports", "refutes"
DIRECTIONS = (SUPPORTS, REFUTES)
PROVISIONAL = "provisional, needs expert review"
# A study type's key, its label and its pseudo-count (B2). Provisional: see `PROVISIONAL`.
STUDY_TYPES: dict[str, tuple[str, float]] = {
    "rct": ("randomised controlled trial", 4.0),
    "cohort": ("cohort study", 2.0),
    "case-series": ("case series", 1.0),
    "in-vitro-or-animal": ("in vitro or animal study", 0.5),
    "self-report": ("self-report", 0.25),
}
UNPUBLISHED = "unpublished observation"
LOG_PATH = DATA / "evidence-log.json"
VERSION = 1  # of the file's layout

_PMID = re.compile(r"(?:pmid\s*:?\s*)?(\d{1,9})", re.IGNORECASE)
_DOI = re.compile(r"(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/\S+)", re.IGNORECASE)


class EvidenceLogError(Exception):
    """An entry or a log that cannot be read or kept, with the reason."""


def _source(text: str) -> str:
    """The source in its one kept form, or EvidenceLogError."""
    text = text.strip()
    if text.lower() == UNPUBLISHED:
        return UNPUBLISHED
    if pmid := _PMID.fullmatch(text):
        return f"PMID:{int(pmid[1])}"
    if doi := _DOI.fullmatch(text):
        return f"doi:{doi[1].lower()}"
    raise EvidenceLogError(f"a source is a PMID, a DOI or {UNPUBLISHED!r}, not {text!r}")


@dataclass(frozen=True)
class Entry:
    """One logged result: a study supporting or refuting a node."""

    id: str
    node: str
    direction: str
    study_type: str
    source: str

    def __post_init__(self) -> None:
        for f in fields(self):
            if not isinstance(getattr(self, f.name), str):
                raise EvidenceLogError(f"an entry's {f.name} is text: {self!r}")
        if not self.id or not self.node.strip():
            raise EvidenceLogError(f"an entry needs an id and a node: {self!r}")
        if self.direction not in DIRECTIONS:
            raise EvidenceLogError(f"a direction is {' or '.join(DIRECTIONS)}, not {self.direction!r}")
        if self.study_type not in STUDY_TYPES:
            raise EvidenceLogError(
                f"a study type is one of {', '.join(STUDY_TYPES)}, not {self.study_type!r}"
            )
        object.__setattr__(self, "source", _source(self.source))

    @classmethod
    def new(cls, node: str, direction: str, study_type: str, source: str) -> Entry:
        return cls(uuid.uuid4().hex, node, direction, study_type, source)

    @property
    def weight(self) -> float:
        return STUDY_TYPES[self.study_type][1]

    @property
    def key(self) -> tuple[str, str, str]:
        """What makes two entries the same: their source (an unpublished one's id), node and direction."""
        return (self.id if self.source == UNPUBLISHED else self.source, self.node, self.direction)


@dataclass(frozen=True)
class Posterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def posterior(papers: int, entries: Iterable[Entry]) -> Posterior:
    """The posterior of a node with `papers` behind it in the library, given its `entries`."""
    alpha = PRIOR + min(PER_PAPER * papers, LITERATURE_CAP)
    beta = PRIOR
    for entry in entries:
        if entry.direction == SUPPORTS:
            alpha += entry.weight
        else:
            beta += entry.weight
    return Posterior(alpha, beta)


class EvidenceLog:
    """The logged entries, in the order logged, with no two the same (`Entry.key`)."""

    def __init__(self, entries: Iterable[Entry] = ()) -> None:
        self._entries: list[Entry] = []
        self._keys: set[tuple[str, str, str]] = set()
        self._ids: set[str] = set()
        for entry in entries:
            self.add(entry)

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    def add(self, entry: Entry) -> bool:
        """Log `entry`; False, and nothing logged, when the log has it already."""
        if entry.key in self._keys or entry.id in self._ids:
            return False
        self._entries.append(entry)
        self._keys.add(entry.key)
        self._ids.add(entry.id)
        return True

    def merge(self, other: EvidenceLog) -> int:
        """Add the entries of `other` (an imported log) this log does not have; how many were added."""
        return sum(self.add(e) for e in other.entries)

    def posterior(self, node: str, papers: int) -> Posterior:
        """The posterior of `node`, with `papers` behind it in the library."""
        return posterior(papers, (e for e in self._entries if e.node == node))

    def to_json(self) -> str:
        return json.dumps({"version": VERSION, "entries": [asdict(e) for e in self._entries]}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> EvidenceLog:
        """The log `to_json` wrote, or EvidenceLogError when it is not one."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise EvidenceLogError(f"not an evidence log: {error}") from error
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            raise EvidenceLogError("not an evidence log: no list of entries")
        if data.get("version") != VERSION:
            raise EvidenceLogError(
                f"an evidence log of version {data.get('version')!r}; this reads {VERSION}"
            )
        names = {f.name for f in fields(Entry)}
        entries = []
        for item in data["entries"]:
            if not isinstance(item, dict) or set(item) != names:
                raise EvidenceLogError(f"an entry has {', '.join(sorted(names))}: {item!r}")
            entries.append(Entry(**item))
        return cls(entries)

    @classmethod
    def load(cls, path: Path = LOG_PATH) -> EvidenceLog:
        """The log kept at `path`; empty when there is none yet."""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls()
        except OSError as error:
            raise EvidenceLogError(f"cannot read the evidence log {path}: {error}") from error
        return cls.from_json(text)

    def save(self, path: Path = LOG_PATH) -> None:
        """Keep the log at `path`, whole or not at all."""
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_name(path.name + ".tmp")
        scratch.write_text(self.to_json(), encoding="utf-8")
        os.replace(scratch, path)


def weighted(scores: Iterable[Bottleneck], graph: nx.DiGraph, log: EvidenceLog) -> list[Bottleneck]:
    """`scores` (`bottlenecks.bottlenecks` of `graph`), each times its node's posterior mean, highest
    first (then by name); the raw metrics are kept."""
    reweighted = [
        replace(b, score=b.score * log.posterior(b.name, graph.nodes[b.name]["papers"]).mean) for b in scores
    ]
    return sorted(reweighted, key=lambda b: (-b.score, b.name))
