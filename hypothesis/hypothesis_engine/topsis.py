"""TOPSIS ranking of the cached drugs: candidates for study, not treatments (spec R3, R5, R6).

Each drug `drugs.py` cached is scored on five criteria:

- *bottleneck impact*: the best bottleneck score of the targets it acts on;
- *cross-condition coverage*: how many of PSSD, PFS and PAS link to any of those targets;
- *safety*, a cost criterion: its penalty, the label's concerns and the listed risks
  (`Safety.penalty`), plus one for a withdrawal from a market after approval (alpidem);
- *BBB permeability*: the CNS MPO estimate (`cns_mpo`), from four of its six terms when ChEMBL
  gives no ClogD or pKa, rescaled to six;
- *readiness and cost*: half for being approved (`max_phase` 4), half for a marketed generic.

TOPSIS vector-normalises each criterion, weights it (by default 30/25/20/15/10), and scores each
drug by its closeness to the ideal best: its distance to the ideal worst over the sum of its
distances to both. A drug alone, or among drugs it cannot be told from, scores one half.

An unknown is never better than what is known: an unknown safety penalty (no US label, or openFDA
could not answer) or CNS MPO (no properties) is taken as the worst seen among the drugs ranked, an
unknown generic as none marketed, and the row says what was assumed (`Row.assumed`).

The drugs that cause the syndromes (`Drug.known_cause`) are never ranked: they are returned apart
as "known causes, excluded". Every row is a candidate for study, with its known risks.

This is the reference implementation; the web app's leaderboard (M6-07) ranks in the browser and a
test checks the two agree.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from hypothesis_engine.drugs import APPROVED, Drug, DrugCache
from hypothesis_engine.hypotheses import CANDIDATE

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
    none can be.

    ChEMBL 36 gives no ClogD or pKa, so most drugs are scored from four terms (ClogP, MW, TPSA,
    HBD), their sum rescaled to six. `cx_logp` is the ClogP when ChEMBL gives it, `alogp` when not.
    """
    values = dict(properties)
    values["logp"] = values.get("cx_logp") if values.get("cx_logp") is not None else values.get("alogp")
    scored = [f(x) for name, f in _DESIRABILITY.items() if (x := values.get(name)) is not None]
    if not scored:
        return None
    return sum(scored) * MPO_TERMS / len(scored), len(scored)


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
        return (self.impact, self.coverage, self.safety, self.bbb, self.readiness)


# What a row says was assumed for it, an unknown taken as the worst seen (or no generic).
SAFETY = "safety penalty"
MPO = "CNS MPO"
GENERIC = "generic"
WITHDRAWN = "withdrawn from a market after approval"
BENEFIT = (True, True, False, True, True)  # safety is a cost: the lower its penalty, the better


@dataclass(frozen=True)
class Criteria:
    """The values a drug is ranked on, stand-ins included."""

    impact: float  # its targets' best bottleneck score, 0 to 1
    coverage: int  # how many of PSSD, PFS, PAS link to any of its targets
    safety: float  # its penalty: label concerns, listed risks, a withdrawal
    bbb: float  # the CNS MPO estimate, 0 to 6
    readiness: float  # half for being approved (max_phase 4), half for a marketed generic

    @property
    def values(self) -> tuple[float, ...]:
        return (self.impact, self.coverage, self.safety, self.bbb, self.readiness)


@dataclass(frozen=True)
class Row:
    drug: Drug
    score: float  # TOPSIS closeness, 0 to 1
    criteria: Criteria
    risks: tuple[str, ...]  # its label's concerns, its listed risks, a withdrawal
    assumed: tuple[str, ...]  # `SAFETY`, `MPO`, `GENERIC`: unknowns given a stand-in
    mpo_terms: int  # how many of CNS MPO's six terms it was scored from; 0 when assumed
    label: str = CANDIDATE


@dataclass(frozen=True)
class Ranking:
    ranked: tuple[Row, ...]  # highest score first, then by name
    known_causes: tuple[Drug, ...]  # the drugs that cause the syndromes: "known causes, excluded"


def _criteria(drugs: Sequence[Drug], cache: DrugCache) -> list[tuple[Criteria, tuple[str, ...], int]]:
    """Each drug's criteria, what was assumed for them, and its CNS MPO's number of terms.

    An unknown safety penalty (no US label, or openFDA unchecked) and an unknown CNS MPO (no
    properties) are taken as the worst seen among the drugs ranked, an unknown generic as none
    marketed: an unknown is never better than what is known.
    """
    targets = {t.symbol: t for t in cache.targets}
    penalties = [d.safety.penalty if d.safety else None for d in drugs]
    mpos = [cns_mpo(d.properties) for d in drugs]
    worst_penalty = max((p for p in penalties if p is not None), default=0)
    worst_mpo = min((m[0] for m in mpos if m is not None), default=0.0)
    found = []
    for drug, penalty, mpo in zip(drugs, penalties, mpos, strict=True):
        acted = [targets[a.target] for a in drug.actions if a.target in targets]
        generic = drug.safety.generic if drug.safety else None
        assumed = tuple(
            name
            for name, unknown in ((SAFETY, penalty is None), (MPO, mpo is None), (GENERIC, generic is None))
            if unknown
        )
        criteria = Criteria(
            max((t.score for t in acted), default=0.0),
            len({s for t in acted for s in t.syndromes}),
            float(worst_penalty if penalty is None else penalty) + (1.0 if drug.withdrawn else 0.0),
            worst_mpo if mpo is None else mpo[0],
            (min(drug.max_phase, APPROVED) / APPROVED + (1.0 if generic else 0.0)) / 2,
        )
        found.append((criteria, assumed, 0 if mpo is None else mpo[1]))
    return found


def _risks(drug: Drug) -> tuple[str, ...]:
    safety = drug.safety
    listed = (*safety.concerns, *(r.risk for r in safety.risks)) if safety else ()
    return (*listed, WITHDRAWN) if drug.withdrawn else tuple(listed)


def closeness(matrix: Sequence[Sequence[float]], weights: Weights) -> list[float]:
    """Each row's TOPSIS closeness to the ideal best: vector-normalised, weighted columns."""
    shares = [w / sum(weights.values) for w in weights.values]
    norms = [math.sqrt(sum(row[j] ** 2 for row in matrix)) for j in range(len(shares))]
    weighted = [
        [shares[j] * row[j] / norms[j] if norms[j] else 0.0 for j in range(len(shares))] for row in matrix
    ]
    columns = list(zip(*weighted, strict=True))
    best = [max(c) if up else min(c) for c, up in zip(columns, BENEFIT, strict=True)]
    worst = [min(c) if up else max(c) for c, up in zip(columns, BENEFIT, strict=True)]
    scores = []
    for row in weighted:
        to_best, to_worst = math.dist(row, best), math.dist(row, worst)
        scores.append(to_worst / (to_best + to_worst) if to_best + to_worst else 0.5)
    return scores


def rank(cache: DrugCache, weights: Weights = Weights()) -> Ranking:
    """The cache's drugs ranked by TOPSIS."""
    drugs = cache.ranked
    found = _criteria(drugs, cache)
    scores = closeness([c.values for c, _, _ in found], weights) if drugs else []
    rows = [
        Row(d, s, c, _risks(d), assumed, terms)
        for d, s, (c, assumed, terms) in zip(drugs, scores, found, strict=True)
    ]
    return Ranking(tuple(sorted(rows, key=lambda r: (-r.score, r.drug.name))), tuple(cache.excluded))
