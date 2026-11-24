"""Portfolio integer program: which single compounds to study within the budget (spec P1-P3, P5, P6)."""

from __future__ import annotations

import itertools

import pytest

from hypothesis_engine.portfolio import (
    BUDGET,
    CANDIDATE,
    COVERAGE,
    GENERICS_ONLY,
    KNOWN_CAUSE,
    MAX_COMPOUNDS,
    NO_COST,
    NOTHING_SELECTABLE,
    SAFETY_CEILING,
    TOPSIS,
    Compound,
    Constraints,
    Target,
    portfolio,
)

# Five targets, weight = bottleneck score × syndromes touched:
# T1 0.9×3 = 2.7, T2 0.8×2 = 1.6, T3 0.6×2 = 1.2, T4 0.5×1 = 0.5, T5 0.4×3 = 1.2 (total 7.2).
TARGETS = (
    Target("T1", 0.9, 3),
    Target("T2", 0.8, 2),
    Target("T3", 0.6, 2),
    Target("T4", 0.5, 1),
    Target("T5", 0.4, 3),
)


def _compound(name: str, cost: float | None, targets: str, topsis: float, **kw: object) -> Compound:
    return Compound(name, frozenset(targets.split()), cost, topsis, **kw)  # type: ignore[arg-type]


# Six compounds: cost, the targets it acts on, its TOPSIS score.
COMPOUNDS = (
    _compound("A", 40, "T1 T2", 0.70),
    _compound("B", 30, "T1", 0.80),
    _compound("C", 20, "T3 T4", 0.55),
    _compound("D", 25, "T5", 0.60),
    _compound("E", 35, "T2 T3 T5", 0.65),
    _compound("F", 10, "T4", 0.30),
)
LIMITS = Constraints(budget=80, max_compounds=3)


def test_weighted_coverage_optimum_matches_the_hand_worked_answer() -> None:
    # Budget 80, at most 3. Only B+E+F (cost 75) covers all five targets: covering T1 needs A or B,
    # T4 needs C or F, T5 needs D or E, T3 needs C or E, T2 needs A or E, and every other cover of all
    # five (B+E+C, A+E+F, A+C+D) costs 85. So the optimum is B, E, F at 7.2.
    result = portfolio(COMPOUNDS, TARGETS, LIMITS)
    assert result.feasible
    assert result.chosen == ("B", "E", "F")
    assert result.objective == pytest.approx(7.2)
    assert result.covers == {"B": ("T1",), "E": ("T2", "T3", "T5"), "F": ("T4",)}


def test_topsis_objective_optimum_matches_the_hand_worked_answer() -> None:
    # Every triple within 80: B+C+D 1.95, C+D+E 1.80, A+B+F 1.80, B+E+F 1.75, B+D+F 1.70, B+C+F 1.65,
    # A+D+F 1.60, A+C+F 1.55, D+E+F 1.55, C+E+F 1.50, C+D+F 1.45; any pair is at most A+B 1.50.
    result = portfolio(COMPOUNDS, TARGETS, LIMITS, objective=TOPSIS)
    assert result.chosen == ("B", "C", "D")
    assert result.objective == pytest.approx(1.95)


def test_an_excluded_pair_changes_the_optimum() -> None:
    # Without B+E, the only set within 80 still covering T1, T2, T3 and T5 is A+E (75); adding F costs 85.
    limits = Constraints(budget=80, max_compounds=3, excluded_pairs=frozenset({frozenset({"B", "E"})}))
    result = portfolio(COMPOUNDS, TARGETS, limits)
    assert result.chosen == ("A", "E")
    assert result.objective == pytest.approx(6.7)


def test_an_infeasible_budget_reports_that_the_budget_binds() -> None:
    # The cheapest compound, F, costs 10.
    result = portfolio(COMPOUNDS, TARGETS, Constraints(budget=5, max_compounds=3))
    assert not result.feasible
    assert result.chosen == ()
    assert result.binding == (BUDGET,)


def test_each_binding_constraint_is_named() -> None:
    assert portfolio(COMPOUNDS, TARGETS, Constraints(budget=80, max_compounds=0)).binding == (MAX_COMPOUNDS,)
    risky = tuple(
        _compound(c.name, c.cost, " ".join(c.targets), c.topsis, safety_penalty=0.9) for c in COMPOUNDS
    )
    assert portfolio(risky, TARGETS, Constraints(80, 3, safety_ceiling=0.5)).binding == (SAFETY_CEILING,)
    assert portfolio(COMPOUNDS, TARGETS, Constraints(80, 3, generics_only=True)).binding == (GENERICS_ONLY,)


def test_known_causes_and_uncosted_compounds_never_appear() -> None:
    # G is a known cause and H has no cost; each alone would cover every target for almost nothing.
    everything = " ".join(t.name for t in TARGETS)
    compounds = (
        *COMPOUNDS,
        _compound("G", 1, everything, 0.99, known_cause=True),
        _compound("H", None, everything, 0.99),
    )
    for objective in (COVERAGE, TOPSIS):
        result = portfolio(compounds, TARGETS, LIMITS, objective=objective)
        assert not {"G", "H"} & set(result.chosen)
        assert result.removed == (("G", KNOWN_CAUSE), ("H", NO_COST))
    only_removed = portfolio(compounds[-2:], TARGETS, LIMITS)
    assert not only_removed.feasible
    assert only_removed.binding == (NOTHING_SELECTABLE,)


def test_safety_ceiling_and_generics_only_limit_the_choice() -> None:
    compounds = tuple(
        _compound(
            c.name,
            c.cost,
            " ".join(c.targets),
            c.topsis,
            safety_penalty=0.7 if c.name == "B" else 0.1,
            generic=c.name != "E",
        )
        for c in COMPOUNDS
    )
    safe = portfolio(compounds, TARGETS, Constraints(80, 3, safety_ceiling=0.5))
    assert "B" not in safe.chosen
    generic = portfolio(compounds, TARGETS, Constraints(80, 3, generics_only=True))
    assert "E" not in generic.chosen


def test_only_the_top_k_targets_count() -> None:
    # With K = 1 only T1 (2.7) counts; B alone covers it most cheaply.
    result = portfolio(COMPOUNDS, TARGETS, LIMITS, top_k=1)
    assert result.chosen == ("B",)
    assert result.objective == pytest.approx(2.7)
    assert result.covers == {"B": ("T1",)}


def _enumerated_best(compounds: tuple[Compound, ...], limits: Constraints, objective: str) -> float | None:
    """The best objective over every subset that fits, by brute force."""
    best = None
    for size in range(1, limits.max_compounds + 1):
        for subset in itertools.combinations(compounds, size):
            names = {c.name for c in subset}
            if sum(c.cost or 0 for c in subset) > limits.budget:
                continue
            if any(pair <= names for pair in limits.excluded_pairs):
                continue
            if limits.safety_ceiling is not None and any(
                c.safety_penalty > limits.safety_ceiling for c in subset
            ):
                continue
            if limits.generics_only and not all(c.generic for c in subset):
                continue
            if objective == COVERAGE:
                covered = set().union(*(c.targets for c in subset))
                value = sum(t.weight for t in TARGETS if t.name in covered)
            else:
                value = sum(c.topsis for c in subset)
            best = value if best is None else max(best, value)
    return best


@pytest.mark.parametrize("objective", [COVERAGE, TOPSIS])
@pytest.mark.parametrize(
    "limits",
    [
        LIMITS,
        Constraints(50, 2),
        Constraints(160, 6),
        Constraints(60, 3, excluded_pairs=frozenset({frozenset({"B", "E"}), frozenset({"A", "C"})})),
        Constraints(15, 2),
    ],
)
def test_the_optimum_matches_enumeration(limits: Constraints, objective: str) -> None:
    result = portfolio(COMPOUNDS, TARGETS, limits, objective=objective)
    assert result.objective == pytest.approx(_enumerated_best(COMPOUNDS, limits, objective))


# B and D carry a safety penalty of 0.7; A and E are not generics.
VARIED = tuple(
    _compound(
        c.name,
        c.cost,
        " ".join(c.targets),
        c.topsis,
        safety_penalty=0.7 if c.name in "BD" else 0.2,
        generic=c.name not in "AE",
    )
    for c in COMPOUNDS
)


@pytest.mark.parametrize("objective", [COVERAGE, TOPSIS])
@pytest.mark.parametrize(
    "limits",
    [
        Constraints(80, 3, safety_ceiling=0.5),
        Constraints(80, 3, generics_only=True),
        Constraints(100, 4, safety_ceiling=0.5, generics_only=True),
    ],
)
def test_the_optimum_matches_enumeration_with_safety_and_generics(
    limits: Constraints, objective: str
) -> None:
    result = portfolio(VARIED, TARGETS, limits, objective=objective)
    assert result.objective == pytest.approx(_enumerated_best(VARIED, limits, objective))


def test_two_limits_that_bind_only_together_are_both_named() -> None:
    # Budget 5 shuts out every compound, and so does a safety ceiling of 0.1; relaxing either alone fits nothing.
    result = portfolio(VARIED, TARGETS, Constraints(5, 3, safety_ceiling=0.1))
    assert not result.feasible
    assert result.binding == (BUDGET, SAFETY_CEILING)


def test_every_chosen_compound_is_a_candidate_for_study() -> None:
    assert portfolio(COMPOUNDS, TARGETS, LIMITS).label == CANDIDATE == "candidate for study"


@pytest.mark.parametrize("cost", [-1.0, float("nan"), float("inf")])
def test_a_cost_that_is_not_a_non_negative_number_is_refused(cost: float) -> None:
    with pytest.raises(ValueError, match="cost"):
        _compound("X", cost, "T1", 0.5)
