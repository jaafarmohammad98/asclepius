"""TOPSIS ranking of the cached drugs: candidates for study, not treatments (spec R3, R5, R6).

Each drug `drugs.py` cached is scored on five criteria:

- *bottleneck impact*: the best bottleneck score of the targets it acts on;
- *cross-condition coverage*: how many of PSSD, PFS and PAS link to any of those targets;
- *safety*, a cost criterion: its penalty, the label's concerns and the listed risks
  (`Safety.penalty`), plus one for a withdrawal from a market after approval (alpidem);
- *BBB permeability*: the CNS MPO estimate (`cns_mpo`), from four of its six terms when ChEMBL
  gives no ClogD or pKa, rescaled to six; fewer than four is unknown;
- *readiness and cost*: half for being approved (`max_phase` 4), half for a marketed generic.

TOPSIS vector-normalises each criterion, weights it (by default 30/25/20/15/10), and scores each
drug by its closeness to the ideal best: its distance to the ideal worst over the sum of its
distances to both. A drug alone, or among drugs it cannot be told from, scores one half.

An unknown is never better than what is known: an unknown safety penalty (no US label, or openFDA
could not answer) or CNS MPO is taken as the worst seen among the drugs ranked, an unknown generic
as none marketed, an unknown withdrawal as one, and the row says what was assumed (`Row.assumed`).

The drugs that cause the syndromes (`Drug.known_cause`) are never ranked: they are returned apart
as "known causes, excluded". Every row is a candidate for study, with its known risks.

This is the reference implementation. The web app's leaderboard (M6-07, `asclepius.web.leaderboard`)
ranks in the browser from each row's criteria (`asclepius/web/topsis.js`), and a test checks the two
agree.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import astuple, dataclass, fields
from typing import Literal

from hypothesis_engine.drugs import APPROVED, Drug, DrugCache
from hypothesis_engine.hypotheses import CANDIDATE
from hypothesis_engine.safety import CONCERNS

# CNS MPO (Wager et al. 2010, doi:10.1021/cn100008c): each property scores 0 to 1 by a
# desirability function, and the six sum to 0 to 6.


def _falling(best: float, worst: float) -> Callable[[float], float]:
    """1 at or below `best`, 0 at or above `worst`, linear between."""
    return lambda x: max(0.0, min(1.0, (worst - x) / (worst - best)))


def _tpsa(x: float) -> float:
    """0 below 20 and above 120, 1 from 40 to 90, linear between."""
    if x <= 40:
        return max(0.0, (x - 20) / 20)
    return 1.0 if x <= 90 else max(0.0, (120 - x) / 30)


MPO_TERMS = 6
MIN_MPO_TERMS = 4  # all ChEMBL 36 gives; fewer is unknown, not a full score from one property
_DESIRABILITY: dict[str, Callable[[float], float]] = {
    "logp": _falling(3, 5),
    "cx_logd": _falling(2, 4),
    "mw_freebase": _falling(360, 500),
    "psa": _tpsa,
    "hbd": _falling(0.5, 3.5),
    "cx_most_bpka": _falling(8, 10),
}


def cns_mpo(properties: Mapping[str, float | None]) -> tuple[float, int] | None:
    """The CNS MPO estimate (0 to 6) and how many of its six terms it is scored from; None when
    fewer than `MIN_MPO_TERMS` can be.

    ChEMBL 36 gives no ClogD or pKa, so most drugs are scored from four terms (ClogP, MW, TPSA,
    HBD), their sum rescaled to six. `cx_logp` is the ClogP when ChEMBL gives it, `alogp` when not.
    """
    values = dict(properties)
    values["logp"] = values.get("cx_logp") if values.get("cx_logp") is not None else values.get("alogp")
    scored = [f(x) for name, f in _DESIRABILITY.items() if (x := values.get(name)) is not None]
    if len(scored) < MIN_MPO_TERMS:
        return None
    return sum(scored) * MPO_TERMS / len(scored), len(scored)


@dataclass(frozen=True)
class Criteria:
    """The values a drug is ranked on, stand-ins included. The field order is the criteria's."""

    impact: float  # its targets' best bottleneck score, 0 to 1
    coverage: int  # how many of PSSD, PFS, PAS link to any of its targets
    safety: float  # its penalty: label concerns, listed risks, a withdrawal
    bbb: float  # the CNS MPO estimate, 0 to 6
    readiness: float  # half for being approved (max_phase 4), half for a marketed generic

    @property
    def values(self) -> tuple[float, ...]:
        return astuple(self)


CRITERIA = tuple(f.name for f in fields(Criteria))
COST = frozenset({"safety"})  # the lower its penalty, the better; the rest are benefits


@dataclass(frozen=True)
class Weights:
    """How much each criterion counts; any non-negative numbers, used as shares of their sum."""

    impact: float = 0.30
    coverage: float = 0.25
    safety: float = 0.20
    bbb: float = 0.15
    readiness: float = 0.10

    def __post_init__(self) -> None:
        values = self.values
        if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) == 0:
            raise ValueError(f"weights must be non-negative and not all zero: {values}")

    @property
    def values(self) -> tuple[float, ...]:
        """In `CRITERIA` order."""
        return tuple(getattr(self, name) for name in CRITERIA)


# An unknown a row was given a stand-in for: the worst seen (or possible), no generic, withdrawn.
Assumed = Literal["safety penalty", "CNS MPO", "generic", "withdrawal"]
WITHDRAWN = "withdrawn from a market after approval"


@dataclass(frozen=True)
class Row:
    """A ranked drug: its score, the criteria it was scored on, its known risks, and what was
    assumed for it."""

    drug: Drug
    score: float  # TOPSIS closeness, 0 to 1
    criteria: Criteria
    risks: tuple[str, ...]  # its label's concerns, its listed risks, a withdrawal
    assumed: tuple[Assumed, ...]  # the unknowns given a stand-in
    mpo_terms: int  # how many of CNS MPO's six terms it was scored from; 0 when assumed
    label: str = CANDIDATE  # a candidate for study, not a treatment (rule 6)

    @property
    def bbb_note(self) -> str:
        """What the CNS MPO shown is: always an estimate."""
        if not self.mpo_terms:
            return "unknown, worst seen assumed"
        return f"estimate, {self.mpo_terms} of {MPO_TERMS} terms"


@dataclass(frozen=True)
class Ranking:
    """The ranked drugs, and apart from them the known causes, never ranked (spec R6)."""

    ranked: tuple[Row, ...]  # highest score first, then by name
    known_causes: tuple[Drug, ...]  # the drugs that cause the syndromes: "known causes, excluded"


@dataclass(frozen=True)
class _Scored:
    criteria: Criteria
    assumed: tuple[Assumed, ...]
    mpo_terms: int


def _criteria(drugs: Sequence[Drug], cache: DrugCache) -> list[_Scored]:
    """Each drug's criteria and what was assumed for them; an unknown is never better than what is
    known.

    An unknown safety penalty (no US label, or openFDA unchecked) is the worst seen among the drugs
    ranked, or when none is known the worst possible (every concern), and never less than the
    drug's own listed risks. An unknown CNS MPO is the worst seen, an unknown generic none
    marketed, an unknown withdrawal (MyChem) a withdrawal.
    """
    targets = {t.symbol: t for t in cache.targets}
    penalties = [d.safety.penalty if d.safety else None for d in drugs]
    mpos = [cns_mpo(d.properties) for d in drugs]
    worst_penalty = max((p for p in penalties if p is not None), default=None)
    worst_mpo = min((m[0] for m in mpos if m is not None), default=0.0)
    found = []
    for drug, penalty, mpo in zip(drugs, penalties, mpos, strict=True):
        acted = [targets[a.target] for a in drug.actions if a.target in targets]
        generic = drug.safety.generic if drug.safety else None
        if penalty is None:
            listed = len(drug.safety.risks) if drug.safety else 0
            penalty = max(listed, len(CONCERNS) + listed if worst_penalty is None else worst_penalty)
        unknowns: tuple[tuple[Assumed, bool], ...] = (
            ("safety penalty", drug.safety is None or drug.safety.penalty is None),
            ("CNS MPO", mpo is None),
            ("generic", generic is None),
            ("withdrawal", drug.withdrawn is None),
        )
        criteria = Criteria(
            max((t.score for t in acted), default=0.0),
            len({s for t in acted for s in t.syndromes}),
            float(penalty) + (0.0 if drug.withdrawn is False else 1.0),
            worst_mpo if mpo is None else mpo[0],
            (min(drug.max_phase, APPROVED) / APPROVED + (1.0 if generic else 0.0)) / 2,
        )
        found.append(
            _Scored(criteria, tuple(name for name, unknown in unknowns if unknown), mpo[1] if mpo else 0)
        )
    return found


def _risks(drug: Drug) -> tuple[str, ...]:
    safety = drug.safety
    listed = (*safety.concerns, *(r.risk for r in safety.risks)) if safety else ()
    return (*listed, WITHDRAWN) if drug.withdrawn else tuple(listed)


def _closeness(matrix: Sequence[Sequence[float]], weights: Weights) -> list[float]:
    """Each row's TOPSIS closeness to the ideal best: vector-normalised, weighted columns."""
    shares = [w / sum(weights.values) for w in weights.values]
    norms = [math.sqrt(sum(row[j] ** 2 for row in matrix)) for j in range(len(shares))]
    weighted = [
        [shares[j] * row[j] / norms[j] if norms[j] else 0.0 for j in range(len(shares))] for row in matrix
    ]
    columns = list(zip(*weighted, strict=True))
    best = [min(c) if name in COST else max(c) for c, name in zip(columns, CRITERIA, strict=True)]
    worst = [max(c) if name in COST else min(c) for c, name in zip(columns, CRITERIA, strict=True)]
    scores = []
    for row in weighted:
        to_best, to_worst = math.dist(row, best), math.dist(row, worst)
        scores.append(to_worst / (to_best + to_worst) if to_best + to_worst else 0.5)
    return scores


def rank(cache: DrugCache, weights: Weights = Weights()) -> Ranking:
    """The cache's drugs ranked by TOPSIS, the known causes apart."""
    drugs = cache.ranked
    found = _criteria(drugs, cache)
    scores = _closeness([f.criteria.values for f in found], weights) if drugs else []
    rows = [
        Row(d, s, f.criteria, _risks(d), f.assumed, f.mpo_terms)
        for d, s, f in zip(drugs, scores, found, strict=True)
    ]
    return Ranking(tuple(sorted(rows, key=lambda r: (-r.score, r.drug.name))), tuple(cache.excluded))
