"""TOPSIS ranking of the cached drugs (M6-06, spec R3, R5, R6).

The expected scores are worked by hand from Wager et al. 2010 (CNS MPO, doi:10.1021/cn100008c) and
the TOPSIS steps: vector-normalise each criterion, weight it, and take each drug's closeness to the
ideal best over its distances to the ideal best and the ideal worst.
"""

from __future__ import annotations

import pytest

from hypothesis_engine.drugs import Action, Drug, DrugCache, Target
from hypothesis_engine.hypotheses import CANDIDATE
from hypothesis_engine.safety import Label, Mention, Risk, Safety
from hypothesis_engine.topsis import Weights, cns_mpo, rank


def _properties(**given: float) -> dict[str, float | None]:
    keys = ("alogp", "cx_logp", "cx_logd", "cx_most_bpka", "mw_freebase", "psa", "hbd", "hba")
    return {k: given.get(k) for k in keys}


def test_cns_mpo_of_chembl_properties_scores_four_of_six_terms_rescaled_to_six():
    # ClogP 2 (1), MW 300 (1), TPSA 50 (1), HBD 0 (1): 4 of 4, so 6 of 6.
    assert cns_mpo(_properties(alogp=2, mw_freebase=300, psa=50, hbd=0)) == (6.0, 4)
    # ClogP 4 (0.5), MW 430 (0.5), TPSA 105 (0.5), HBD 2 (0.5): 2 of 4, so 3 of 6.
    assert cns_mpo(_properties(alogp=4, mw_freebase=430, psa=105, hbd=2)) == pytest.approx((3.0, 4))
    # ClogP 5 (0), MW 500 (0), TPSA 30 (0.5), HBD 2 (0.5): 1 of 4, so 1.5 of 6.
    assert cns_mpo(_properties(alogp=5, mw_freebase=500, psa=30, hbd=2)) == pytest.approx((1.5, 4))


def test_cns_mpo_tpsa_is_a_hump_low_and_high_both_score_zero():
    assert cns_mpo(_properties(psa=15)) == (0.0, 1)
    assert cns_mpo(_properties(psa=130)) == (0.0, 1)
    assert cns_mpo(_properties(psa=40)) == (6.0, 1)
    assert cns_mpo(_properties(psa=90)) == (6.0, 1)


def test_cns_mpo_uses_the_chemaxon_values_when_chembl_gives_them():
    # cx_logp 3 (1) wins over alogp 5 (0); cx_logd 3 (0.5) and pKa 9 (0.5) are scored: 6 terms.
    score, terms = cns_mpo(
        _properties(alogp=5, cx_logp=3, cx_logd=3, cx_most_bpka=9, mw_freebase=360, psa=60, hbd=0.5)
    )
    assert (score, terms) == (5.0, 6)


def test_cns_mpo_is_unknown_without_any_property():
    assert cns_mpo(_properties()) is None


TARGETS = (
    Target("SRD5A1", 0.8, ("PAS", "PFS", "PSSD"), (), "chembl"),
    Target("HTR2A", 0.4, ("PSSD",), (), "chembl"),
    Target("TSPO", 0.6, ("PAS", "PFS"), (), "chembl"),
)
GOOD_MPO = {"alogp": 2, "mw_freebase": 300, "psa": 50, "hbd": 0}  # 6.0
FAIR_MPO = {"alogp": 4, "mw_freebase": 430, "psa": 105, "hbd": 2}  # 3.0
POOR_MPO = {"alogp": 5, "mw_freebase": 500, "psa": 30, "hbd": 2}  # 1.5


def _safety(concerns: int = 0, generic: bool | None = True, label: bool = True, risks=()) -> Safety:
    names = ("sexual dysfunction", "libido", "anhedonia", "emotional blunting")[:concerns]
    mentions = tuple(Mention(c, "adverse_reactions", f"...{c}...") for c in names)  # type: ignore[arg-type]
    return Safety(
        Label("set", "20260101", "NDA000001", mentions) if label else None,
        tuple(risks),
        None if generic is None else (("ANDA000001",) if generic else ()),
    )


def _drug(
    name: str,
    targets: tuple[str, ...],
    mpo: dict[str, float],
    safety: Safety | None,
    known_cause: str | None = None,
    withdrawn: bool | None = False,
) -> Drug:
    return Drug(
        f"CHEMBL_{name.upper()}",
        name,
        4.0,
        _properties(**mpo),
        tuple(Action(t, None, "INHIBITOR", None) for t in targets),
        known_cause,
        "chembl",
        withdrawn,
        safety,
    )


def _cache(*drugs: Drug) -> DrugCache:
    return DrugCache("2026-11-25T00:00:00+00:00", 1, TARGETS, drugs)


# The hand-worked example. Criteria (impact, coverage, safety penalty, CNS MPO, readiness):
#   alpha  0.8  3  2  6.0  1.0
#   beta   0.6  3  0  3.0  0.5   (best of HTR2A 0.4 and TSPO 0.6; PSSD with PAS and PFS)
#   gamma  0.4  1  1  1.5  1.0
# Column norms 1.0770, 4.3589, 2.2361, 6.8739, 1.5. Weighted (30/25/20/15/10):
#   alpha  .22283 .17206 .17889 .13093 .06667
#   beta   .16713 .17206 0      .06547 .03333
#   gamma  .11142 .05735 .08944 .03273 .06667
# Ideal best .22283 .17206 0 .13093 .06667 (safety is a cost: its lowest), worst the opposite.
# Distances to best / worst: alpha .17889 / .19059, beta .09220 / .22211, gamma .20788 / .09545.
ALPHA = _drug("alpha", ("SRD5A1",), GOOD_MPO, _safety(concerns=2))
BETA = _drug("beta", ("HTR2A", "TSPO"), FAIR_MPO, _safety(generic=False))
GAMMA = _drug("gamma", ("HTR2A",), POOR_MPO, _safety(concerns=1))


def test_the_ranking_matches_the_hand_worked_example():
    ranking = rank(_cache(ALPHA, BETA, GAMMA))
    assert [(r.drug.name, round(r.score, 4)) for r in ranking.ranked] == [
        ("beta", 0.7067),
        ("alpha", 0.5158),
        ("gamma", 0.3147),
    ]


def test_known_causes_are_returned_separately_and_never_ranked():
    finasteride = _drug(
        "finasteride", ("SRD5A1",), GOOD_MPO, _safety(), known_cause="5-alpha reductase inhibitor"
    )
    ranking = rank(_cache(ALPHA, finasteride, BETA, GAMMA))
    assert [r.drug.name for r in ranking.ranked] == ["beta", "alpha", "gamma"]
    assert [d.name for d in ranking.known_causes] == ["finasteride"]
    assert [round(r.score, 4) for r in ranking.ranked] == [0.7067, 0.5158, 0.3147]  # as without it


def test_every_ranked_row_is_a_candidate_for_study():
    assert {r.label for r in rank(_cache(ALPHA, BETA, GAMMA)).ranked} == {CANDIDATE}


def test_a_withdrawn_drug_is_ranked_with_its_withdrawal_as_one_more_penalty():
    # gamma's penalty is 1 + 1: alpha and gamma both 2, safety norm 2.8284, weighted .14142 each.
    withdrawn = _drug("gamma", ("HTR2A",), POOR_MPO, _safety(concerns=1), withdrawn=True)
    ranking = rank(_cache(ALPHA, BETA, withdrawn))
    assert [(r.drug.name, round(r.score, 4)) for r in ranking.ranked] == [
        ("beta", 0.6770),
        ("alpha", 0.5741),
        ("gamma", 0.1242),
    ]
    gamma = ranking.ranked[2]
    assert gamma.criteria.safety == 2
    assert "withdrawn from a market after approval" in gamma.risks


def test_a_row_lists_its_label_concerns_and_listed_risks():
    risky = _drug(
        "alpha", ("SRD5A1",), GOOD_MPO, _safety(concerns=2, risks=[Risk("hepatotoxicity", "PMID:1")])
    )
    (row,) = rank(_cache(risky)).ranked
    assert row.risks == ("sexual dysfunction", "libido", "hepatotoxicity")
    assert row.criteria.safety == 3


def test_an_unknown_safety_penalty_is_taken_as_the_worst_seen_and_marked():
    no_label = _drug("delta", ("TSPO",), FAIR_MPO, _safety(label=False))
    unchecked = _drug("epsilon", ("TSPO",), FAIR_MPO, None)
    rows = {r.drug.name: r for r in rank(_cache(ALPHA, BETA, GAMMA, no_label, unchecked)).ranked}
    assert rows["delta"].criteria.safety == 2  # alpha's, the worst
    assert rows["epsilon"].criteria.safety == 2
    assert "safety penalty" in rows["delta"].assumed
    assert rows["alpha"].assumed == ()


def test_the_worst_seen_penalty_stands_in_before_a_withdrawal_is_added():
    withdrawn = _drug("delta", ("TSPO",), FAIR_MPO, _safety(label=False), withdrawn=True)
    (row, *_) = [r for r in rank(_cache(ALPHA, BETA, withdrawn)).ranked if r.drug.name == "delta"]
    assert row.criteria.safety == 3


def test_an_unknown_generic_is_taken_as_none_marketed():
    unknown = _drug("delta", ("TSPO",), FAIR_MPO, _safety(generic=None))
    (row,) = rank(_cache(unknown)).ranked
    assert row.criteria.readiness == 0.5
    assert row.assumed == ("generic",)


def test_an_unknown_cns_mpo_is_taken_as_the_worst_seen_and_marked():
    no_properties = _drug("delta", ("TSPO",), {}, _safety())
    rows = {r.drug.name: r for r in rank(_cache(ALPHA, GAMMA, no_properties)).ranked}
    assert rows["delta"].criteria.bbb == 1.5  # gamma's, the worst
    assert rows["delta"].assumed == ("CNS MPO",)
    assert rows["alpha"].mpo_terms == 4
    assert rows["delta"].mpo_terms == 0


def test_the_weights_steer_the_ranking():
    only_safety = Weights(impact=0, coverage=0, safety=1, bbb=0, readiness=0)
    assert [r.drug.name for r in rank(_cache(ALPHA, BETA, GAMMA), only_safety).ranked] == [
        "beta",
        "gamma",
        "alpha",
    ]
    only_bbb = Weights(impact=0, coverage=0, safety=0, bbb=1, readiness=0)
    assert [r.drug.name for r in rank(_cache(ALPHA, BETA, GAMMA), only_bbb).ranked] == [
        "alpha",
        "beta",
        "gamma",
    ]


def test_weights_are_shares_of_their_sum():
    doubled = Weights(impact=60, coverage=50, safety=40, bbb=30, readiness=20)
    assert [r.score for r in rank(_cache(ALPHA, BETA, GAMMA), doubled).ranked] == pytest.approx(
        [0.7067, 0.5158, 0.3147], abs=5e-5
    )


@pytest.mark.parametrize(
    "weights",
    [
        dict(impact=-1),
        dict(bbb=float("nan")),
        dict.fromkeys(("impact", "coverage", "safety", "bbb", "readiness"), 0),
    ],
)
def test_negative_or_all_zero_weights_are_refused(weights):
    with pytest.raises(ValueError):
        Weights(**weights)


def test_with_nothing_to_rank_the_ranking_is_empty():
    assert rank(_cache()).ranked == ()


def test_a_drug_ranked_alone_scores_one_half_for_it_is_both_best_and_worst():
    (row,) = rank(_cache(ALPHA)).ranked
    assert row.score == 0.5
