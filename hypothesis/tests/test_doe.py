"""Two-level fractional factorial designs (M6-08, spec D1 to D4).

The expected generators, defining relations and alias chains are the standard ones (Box, Hunter and
Hunter, *Statistics for Experimenters*, 2nd ed., table 6.22; Montgomery, *Design and Analysis of
Experiments*, table 8.14), worked by hand here.
"""

from __future__ import annotations

import csv
import io
from itertools import combinations

import pytest

from hypothesis_engine.doe import LETTERS, SEED_LIMIT, Design, design

FIVE = ["A", "B", "C", "D", "E"]


def _column(d: Design, effect: str) -> list[int]:
    """The ±1 column of a main effect or an interaction, the product of its factors' columns."""
    index = {name: i for i, name in enumerate(d.factors)}
    out = []
    for run in d.matrix:
        x = 1
        for name in effect:
            x *= run[index[name]]
        out.append(x)
    return out


def _dot(a: list[int], b: list[int]) -> int:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_five_factors_is_a_16_run_resolution_v_design_with_2fis_unaliased():
    d = design(FIVE, seed=1)
    assert len(d.matrix) == 16
    assert d.resolution == 5
    assert d.generators == ("E=ABCD",)
    assert d.defining_relation == ("ABCDE",)
    effects = FIVE + ["".join(p) for p in combinations(FIVE, 2)]
    for a, b in combinations(effects, 2):
        assert _dot(_column(d, a), _column(d, b)) == 0, (a, b)
    for e in effects:
        assert sum(_column(d, e)) == 0, e
    assert d.confounded == ()


# The 16-run resolution IV fractions, their words, and their two-factor alias chains, by hand.
RESOLUTION_IV = {
    6: (
        ("E=ABC", "F=BCD"),
        ("ABCE", "ADEF", "BCDF"),
        ("AB=CE", "AC=BE", "AD=EF", "AE=BC=DF", "AF=DE", "BD=CF", "BF=CD"),
    ),
    7: (
        ("E=ABC", "F=BCD", "G=ACD"),
        ("ABCE", "ABFG", "ACDG", "ADEF", "BCDF", "BDEG", "CEFG"),
        ("AB=CE=FG", "AC=BE=DG", "AD=CG=EF", "AE=BC=DF", "AF=BG=DE", "AG=BF=CD", "BD=CF=EG"),
    ),
    8: (
        ("E=BCD", "F=ACD", "G=ABC", "H=ABD"),
        (
            "ABCG",
            "ABDH",
            "ABEF",
            "ACDF",
            "ACEH",
            "ADEG",
            "AFGH",
            "BCDE",
            "BCFH",
            "BDFG",
            "BEGH",
            "CDGH",
            "CEFG",
            "DEFH",
            "ABCDEFGH",
        ),
        (
            "AB=CG=DH=EF",
            "AC=BG=DF=EH",
            "AD=BH=CF=EG",
            "AE=BF=CH=DG",
            "AF=BE=CD=GH",
            "AG=BC=DE=FH",
            "AH=BD=CE=FG",
        ),
    ),
}


@pytest.mark.parametrize("k", sorted(RESOLUTION_IV))
def test_six_to_eight_factors_are_16_run_resolution_iv_with_the_standard_aliasing(k):
    names = list(LETTERS[:k])
    generators, words, chains = RESOLUTION_IV[k]
    d = design(names, seed=1)
    assert len(d.matrix) == 16
    assert d.resolution == 4
    assert d.generators == generators
    assert d.defining_relation == words
    assert tuple("=".join(c) for c in d.confounded) == chains
    # Main effects are balanced, orthogonal to each other and clear of every two-factor interaction.
    pairs = ["".join(p) for p in combinations(names, 2)]
    for a in names:
        assert sum(_column(d, a)) == 0
        for b in names + pairs:
            if b != a:
                assert _dot(_column(d, a), _column(d, b)) == 0, (a, b)
    # Each chain's interactions share one column.
    for chain in d.confounded:
        assert len({tuple(_column(d, p)) for p in chain}) == 1
    assert "confounded" in d.note


@pytest.mark.parametrize("k", [2, 3, 4])
def test_two_to_four_factors_are_the_full_factorial_with_nothing_aliased(k):
    names = list(LETTERS[:k])
    d = design(names, seed=1)
    assert len(d.matrix) == 2**k
    assert len(set(d.matrix)) == 2**k
    assert d.resolution is None
    assert d.generators == d.defining_relation == d.confounded == ()
    assert d.note == "Main effects and two-factor interactions are all separable."


def test_the_standard_order_alternates_the_first_factor_fastest():
    assert design(["A", "B", "C"], seed=1).matrix == (
        (-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1),
    )  # fmt: skip


def test_the_note_names_the_confounded_factors():
    d = design(["sildenafil", "B", "C", "D", "E", "F"], seed=1)
    assert "sildenafil × B = C × E" in d.note


@pytest.mark.parametrize("factors", [[], ["A"], list("ABCDEFGHI")])
def test_fewer_than_two_or_more_than_eight_factors_are_refused(factors):
    with pytest.raises(ValueError, match="2 to 8 factors"):
        design(factors, seed=1)


@pytest.mark.parametrize("factors", [["A", "A"], ["A", ""], ["A", " "]])
def test_factors_must_have_distinct_names(factors):
    with pytest.raises(ValueError, match="name"):
        design(factors, seed=1)


def test_the_run_order_is_a_shuffle_its_seed_reproduces():
    d = design(FIVE, seed=42)
    assert d.seed == 42
    assert sorted(d.run_order) == list(range(1, 17))
    assert design(FIVE, seed=42).run_order == d.run_order
    assert design(FIVE, seed=43).run_order != d.run_order
    assert d.run_order != tuple(range(1, 17))


def test_without_a_seed_one_is_drawn_and_recorded():
    d = design(FIVE)
    assert isinstance(d.seed, int)
    assert design(FIVE, seed=d.seed).run_order == d.run_order


def _rows(text: str) -> tuple[str, list[dict[str, str]]]:
    first, rest = text.split("\n", 1)
    return first, list(csv.DictReader(io.StringIO(rest)))


def test_the_csv_lists_the_runs_in_run_order_under_a_safety_warning():
    d = design(["sildenafil", "TSPO", "C"], seed=7)
    warning, rows = _rows(d.csv())
    assert warning.startswith("#")
    assert "safety review before any wet-lab or animal use" in warning
    assert list(rows[0]) == [
        "run_id",
        "run_order",
        "seed",
        "sildenafil",
        "TSPO",
        "C",
        "replicates",
        "response",
    ]
    assert [r["run_order"] for r in rows] == [str(i) for i in range(1, 9)]
    assert [int(r["run_id"]) for r in rows] == list(d.run_order)
    level = {1: "present", -1: "absent"}
    for r in rows:
        run = d.matrix[int(r["run_id"]) - 1]
        assert [r["sildenafil"], r["TSPO"], r["C"]] == [level[x] for x in run]
        assert r["seed"] == "7"
        assert r["replicates"] == "1"
        assert r["response"] == ""


def test_replicates_and_named_responses_get_one_empty_column_each():
    d = design(["A", "B"], seed=1, replicates=3)
    _, rows = _rows(d.csv(responses=["libido", "anhedonia"]))
    assert list(rows[0])[-7:] == [
        "replicates", "libido_1", "libido_2", "libido_3", "anhedonia_1", "anhedonia_2", "anhedonia_3",
    ]  # fmt: skip
    assert {r["replicates"] for r in rows} == {"3"}
    assert {r["libido_2"] for r in rows} == {""}


@pytest.mark.parametrize("replicates", [0, -1])
def test_at_least_one_replicate(replicates):
    with pytest.raises(ValueError, match="replicate"):
        design(["A", "B"], seed=1, replicates=replicates)


def test_a_response_cannot_share_a_name_with_a_column():
    with pytest.raises(ValueError, match="name"):
        design(["A", "B"], seed=1).csv(responses=["A"])


def test_a_shuffle_may_leave_the_standard_order_rather_than_bias_the_draw():
    # 1 in 24 orders of 4 runs is the standard order; refusing it would make the draw non-uniform.
    orders = {design(["A", "B"], seed=s).run_order for s in range(500)}
    assert (1, 2, 3, 4) in orders
    assert len(orders) == 24


@pytest.mark.parametrize(
    ("k", "label"),
    [
        (3, "Full factorial 2^3, 8 runs"),
        (5, "Fractional factorial 2^(5-1), resolution V, 16 runs"),
        (8, "Fractional factorial 2^(8-4), resolution IV, 16 runs"),
    ],
)
def test_the_label_names_the_design_and_its_resolution(k, label):
    assert design(list(LETTERS[:k]), seed=1).label == label


def test_the_note_gives_the_resolution_in_roman_numerals():
    assert design(list(LETTERS[:6]), seed=1).note.startswith("Resolution IV:")


def test_the_safety_warning_says_candidate_for_study():
    warning = design(["A", "B"], seed=1).csv().split("\n", 1)[0]
    assert "candidate for study" in warning


def test_the_csv_lists_each_factor_as_a_candidate_with_its_known_risks():
    d = design(["betadrug", "COMT"], seed=3)
    text = d.csv(risks={"betadrug": "libido; raises blood pressure", "COMT": "none listed"})
    lines = text.splitlines()
    assert lines[0].startswith("# Safety warning")
    assert lines[1] == "# betadrug (candidate for study), known risks: libido; raises blood pressure"
    assert lines[2] == "# COMT (candidate for study), known risks: none listed"
    assert lines[3].startswith("run_id,")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[3:]))))
    assert len(rows) == 4 and rows[0]["betadrug"] in ("present", "absent")


def test_a_risk_cannot_break_out_of_its_comment_line():
    d = design(["A", "B"], seed=3)
    text = d.csv(risks={"A": "one\nrun_id,forged", "B": "none"})
    assert "# A (candidate for study), known risks: one run_id,forged" in text.splitlines()


def test_given_risks_must_say_something_for_every_factor_and_no_other():
    d = design(["A", "B"], seed=3)
    with pytest.raises(ValueError, match="B"):
        d.csv(risks={"A": "none"})
    with pytest.raises(ValueError, match="C"):
        d.csv(risks={"A": "none", "B": "none", "C": "none"})
    with pytest.raises(ValueError, match="B"):
        d.csv(risks={"A": "none", "B": " "})


def test_the_seed_limit_bounds_the_seeds_drawn():
    assert 0 <= design(["A", "B"]).seed < SEED_LIMIT
