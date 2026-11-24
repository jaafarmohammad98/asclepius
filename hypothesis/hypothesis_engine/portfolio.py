"""Portfolio integer program: which single compounds to study, within the researcher's limits (spec P1-P6).

One binary choice per compound and, under the default objective (weighted coverage), one binary per
top-K bottleneck target, covered only if a chosen compound acts on it. A covered target earns its
bottleneck score (with the evidence multiplier) times the number of syndromes it touches. The
alternative objective is the total TOPSIS score of the chosen compounds. Among equally good
portfolios the cheapest is taken. Solved with `scipy.optimize.milp`.

Constraints: total entered cost within the budget, at most the maximum number of compounds, no
excluded pair both chosen, each compound's safety penalty within the ceiling, and optionally generics
only. A portfolio holds at least one compound. Known causes and compounds without an entered cost are
removed before solving, each with its reason, so neither can appear in any solution. When no
portfolio fits, each constraint is relaxed in turn and re-solved to say which one binds.

Every chosen compound is a candidate for study, not a treatment, and covering a target says the
compound acts on it, not that acting on it helps.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

COVERAGE = "coverage"
TOPSIS = "topsis"

KNOWN_CAUSE = "known cause, never selectable"
NO_COST = "no cost entered"

BUDGET = "budget"
MAX_COMPOUNDS = "maximum compounds"
EXCLUDED_PAIRS = "excluded pairs"
SAFETY_CEILING = "safety ceiling"
GENERICS_ONLY = "generics only"


@dataclass(frozen=True)
class Target:
    """A bottleneck target: its score with the evidence multiplier, and how many syndromes it touches."""

    name: str
    score: float
    syndromes: int

    @property
    def weight(self) -> float:
        return self.score * self.syndromes


@dataclass(frozen=True)
class Compound:
    """A leaderboard compound: the targets it acts on, its entered cost (None if none) and TOPSIS score."""

    name: str
    targets: frozenset[str]
    cost: float | None
    topsis: float
    safety_penalty: float = 0.0
    generic: bool = False
    known_cause: bool = False


@dataclass(frozen=True)
class Constraints:
    """The researcher's limits. `excluded_pairs` holds both the label-flagged and the researcher's pairs."""

    budget: float
    max_compounds: int
    excluded_pairs: frozenset[frozenset[str]] = frozenset()
    safety_ceiling: float | None = None
    generics_only: bool = False

    def __post_init__(self) -> None:
        if math.isnan(self.budget) or self.budget < 0:
            raise ValueError(f"budget must be a non-negative number: {self.budget}")
        if self.max_compounds < 0:
            raise ValueError(f"maximum compounds must be non-negative: {self.max_compounds}")
        if any(len(pair) != 2 for pair in self.excluded_pairs):
            raise ValueError("each excluded pair names two different compounds")


@dataclass(frozen=True)
class Portfolio:
    """The solution, or, when nothing fits, which constraints bind (each one whose relaxation alone fits)."""

    chosen: tuple[str, ...]
    objective: float
    covers: dict[str, tuple[str, ...]]  # per chosen compound, the top-K targets it acts on
    removed: tuple[tuple[str, str], ...]  # (compound, why) for those removed before solving
    binding: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return bool(self.chosen)


def portfolio(
    compounds: Iterable[Compound],
    targets: Iterable[Target],
    constraints: Constraints,
    objective: str = COVERAGE,
    top_k: int = 20,
) -> Portfolio:
    """The best portfolio of single compounds under `constraints`, by `objective` (COVERAGE or TOPSIS)."""
    if objective not in (COVERAGE, TOPSIS):
        raise ValueError(f"objective must be {COVERAGE!r} or {TOPSIS!r}: {objective!r}")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1: {top_k}")
    compounds = sorted(compounds, key=lambda c: c.name)
    if len({c.name for c in compounds}) != len(compounds):
        raise ValueError("compound names must be unique")
    removed = tuple(
        (c.name, KNOWN_CAUSE if c.known_cause else NO_COST)
        for c in compounds
        if c.known_cause or c.cost is None
    )
    selectable = [c for c in compounds if not c.known_cause and c.cost is not None]
    top = sorted(targets, key=lambda t: (-t.score, t.name))[:top_k]

    chosen = _solve(selectable, top, constraints, objective)
    if chosen is None:
        return Portfolio((), 0.0, {}, removed, _binding(selectable, top, constraints, objective))
    top_names = {t.name for t in top}
    covers = {c.name: tuple(sorted(c.targets & top_names)) for c in chosen}
    if objective == COVERAGE:
        covered = set().union(*covers.values())
        value = sum(t.weight for t in top if t.name in covered)
    else:
        value = sum(c.topsis for c in chosen)
    return Portfolio(tuple(c.name for c in chosen), value, covers, removed)


def _solve(
    compounds: Sequence[Compound], top: Sequence[Target], limits: Constraints, objective: str
) -> list[Compound] | None:
    """The chosen compounds, or None when no portfolio of at least one compound fits."""
    n = len(compounds)
    if n == 0:
        return None
    top = top if objective == COVERAGE else ()
    k = len(top)
    names = [c.name for c in compounds]
    costs = np.array([c.cost for c in compounds], dtype=float)

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(row: np.ndarray, lo: float, hi: float) -> None:
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    add(np.concatenate([costs, np.zeros(k)]), -np.inf, limits.budget)
    add(np.concatenate([np.ones(n), np.zeros(k)]), 1, limits.max_compounds)
    for pair in limits.excluded_pairs:
        a, b = sorted(pair)
        if a in names and b in names:
            row = np.zeros(n + k)
            row[names.index(a)] = row[names.index(b)] = 1
            add(row, -np.inf, 1)
    # A target counts as covered only if a chosen compound acts on it: y_t - Σ x_i ≤ 0.
    for j, target in enumerate(top):
        row = np.zeros(n + k)
        row[n + j] = 1
        for i, c in enumerate(compounds):
            if target.name in c.targets:
                row[i] = -1
        add(row, -np.inf, 0)

    allowed = np.array(
        [
            (limits.safety_ceiling is None or c.safety_penalty <= limits.safety_ceiling)
            and (not limits.generics_only or c.generic)
            for c in compounds
        ],
        dtype=float,
    )
    bounds = Bounds(np.zeros(n + k), np.concatenate([allowed, np.ones(k)]))
    integrality = np.ones(n + k)
    if objective == COVERAGE:
        gain = np.concatenate([np.zeros(n), [t.weight for t in top]])
    else:
        gain = np.concatenate([[c.topsis for c in compounds], np.zeros(k)])

    first = milp(-gain, integrality=integrality, bounds=bounds, constraints=_constraint(rows, lower, upper))
    if first.status != 0:
        return None
    # Among the best portfolios, the cheapest: keep the objective, minimise cost.
    best = -first.fun
    add(gain, best - 1e-9 * max(1.0, abs(best)), np.inf)
    cheapest = milp(
        np.concatenate([costs, np.zeros(k)]),
        integrality=integrality,
        bounds=bounds,
        constraints=_constraint(rows, lower, upper),
    )
    x = (cheapest if cheapest.status == 0 else first).x
    return [c for i, c in enumerate(compounds) if x[i] > 0.5]


def _constraint(rows: list[np.ndarray], lower: list[float], upper: list[float]) -> LinearConstraint:
    return LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))


def _binding(
    compounds: Sequence[Compound], top: Sequence[Target], limits: Constraints, objective: str
) -> tuple[str, ...]:
    """The constraints each of whose relaxation alone lets a portfolio fit."""
    relaxed = {
        BUDGET: replace(limits, budget=math.inf),
        MAX_COMPOUNDS: replace(limits, max_compounds=max(len(compounds), 1)),
        EXCLUDED_PAIRS: replace(limits, excluded_pairs=frozenset()),
        SAFETY_CEILING: replace(limits, safety_ceiling=None),
        GENERICS_ONLY: replace(limits, generics_only=False),
    }
    return tuple(
        name for name, loose in relaxed.items() if _solve(compounds, top, loose, objective) is not None
    )
