"""The page's TOPSIS (`asclepius/web/topsis.js`) ranks as the Python reference does (M6-07, spec R4):
on the hand-worked example of M6-06, on the app's fixture drugs, and on random drug caches, each under
many weightings. The page is given each drug's criteria as the leaderboard sends them.

Runs the script in node (`js/topsis_parity.cjs`); skipped when node is absent.
"""

import json
import random
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from hypothesis_engine.drugs import Action, Drug, DrugCache, Target
from hypothesis_engine.safety import CONCERNS, Label, Mention, Risk, Safety
from hypothesis_engine.topsis import Weights, rank
from knowledge_fixture import DRUGS, PROPERTIES

SCRIPT = Path(__file__).parent / "js" / "topsis_parity.cjs"

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="needs node")

WEIGHTINGS = [
    Weights(),
    Weights(impact=1, coverage=0, safety=0, bbb=0, readiness=0),
    Weights(impact=0, coverage=0, safety=1, bbb=0, readiness=0),
    Weights(impact=0.1, coverage=0.1, safety=0.6, bbb=0.1, readiness=0.1),
    Weights(impact=0, coverage=0.5, safety=0, bbb=0.5, readiness=0),
    Weights(impact=3, coverage=1, safety=2, bbb=0, readiness=7),
]


def browser_rank(cases: list[tuple[DrugCache, Weights]]) -> list[list[tuple[str, float]]]:
    payload = [
        {
            "rows": [{"name": r.drug.name, "criteria": asdict(r.criteria)} for r in rank(cache).ranked],
            "weights": asdict(weights),
        }
        for cache, weights in cases
    ]
    done = subprocess.run(
        ["node", str(SCRIPT)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return [[(r["name"], r["score"]) for r in ranked] for ranked in json.loads(done.stdout)]


def python_rank(cache: DrugCache, weights: Weights) -> list[tuple[str, float]]:
    return [(r.drug.name, r.score) for r in rank(cache, weights).ranked]


def assert_same(cases: list[tuple[DrugCache, Weights]]) -> None:
    for (cache, weights), browser in zip(cases, browser_rank(cases), strict=True):
        python = python_rank(cache, weights)
        assert [n for n, _ in browser] == [n for n, _ in python], weights
        assert [s for _, s in browser] == pytest.approx([s for _, s in python], abs=1e-12)


def _drug(name, targets, properties, safety, withdrawn=False):
    return Drug(
        f"CHEMBL_{name.upper()}",
        name,
        4.0,
        {k: properties.get(k) for k in PROPERTIES},
        tuple(Action(t, None, "INHIBITOR", None) for t in targets),
        None,
        "chembl",
        withdrawn,
        safety,
    )


def _safety(concerns: int, generic: bool) -> Safety:
    mentions = tuple(Mention(c, "adverse_reactions", c) for c in list(CONCERNS)[:concerns])
    return Safety(Label("set", None, None, mentions), (), ("ANDA1",) if generic else ())


HAND_WORKED = DrugCache(  # M6-06's: beta 0.7067, alpha 0.5158, gamma 0.3147 with the default weights
    "2026-11-25",
    1,
    (
        Target("SRD5A1", 0.8, ("PAS", "PFS", "PSSD"), (), "chembl"),
        Target("HTR2A", 0.4, ("PSSD",), (), "chembl"),
        Target("TSPO", 0.6, ("PAS", "PFS"), (), "chembl"),
    ),
    (
        _drug("alpha", ("SRD5A1",), {"alogp": 2, "mw_freebase": 300, "psa": 50, "hbd": 0}, _safety(2, True)),
        _drug(
            "beta",
            ("HTR2A", "TSPO"),
            {"alogp": 4, "mw_freebase": 430, "psa": 105, "hbd": 2},
            _safety(0, False),
        ),
        _drug("gamma", ("HTR2A",), {"alogp": 5, "mw_freebase": 500, "psa": 30, "hbd": 2}, _safety(1, True)),
    ),
)


def test_the_page_ranks_the_hand_worked_example_as_worked_by_hand():
    (ranked,) = browser_rank([(HAND_WORKED, Weights())])
    assert [(n, round(s, 4)) for n, s in ranked] == [("beta", 0.7067), ("alpha", 0.5158), ("gamma", 0.3147)]


def test_the_page_ranks_the_hand_worked_example_as_python_does_under_any_weights():
    assert_same([(HAND_WORKED, w) for w in WEIGHTINGS])


def test_the_page_ranks_the_app_s_fixture_drugs_as_python_does():
    assert_same([(DRUGS, w) for w in WEIGHTINGS])


def test_identical_drugs_tie_at_one_half_and_go_by_name_in_both():
    twin = HAND_WORKED.drugs[0]
    cache = DrugCache(
        "2026-11-25",
        1,
        HAND_WORKED.targets,
        (twin, *(_drug(n, ("SRD5A1",), twin.properties, twin.safety) for n in ("b", "a"))),
    )
    (ranked,) = browser_rank([(cache, Weights())])
    assert ranked == [("a", 0.5), ("alpha", 0.5), ("b", 0.5)]
    assert_same([(cache, Weights())])


def _random_cache(seed: int) -> DrugCache:
    rng = random.Random(seed)
    symbols = [f"G{i}" for i in range(6)]
    targets = tuple(
        Target(
            s,
            rng.random(),
            tuple(sorted(rng.sample(["PSSD", "PFS", "PAS"], rng.randint(0, 3)))),
            (),
            "chembl",
        )
        for s in symbols
    )
    drugs = []
    for i in range(rng.randint(2, 12)):
        properties = {k: rng.uniform(0, 8) for k in ("alogp", "hbd")} | {
            "mw_freebase": rng.uniform(200, 600),
            "psa": rng.uniform(0, 150),
        }
        if rng.random() < 0.15:
            properties = {}  # an unknown CNS MPO: the worst seen stands in
        safety = None if rng.random() < 0.1 else _safety(rng.randint(0, 4), rng.random() < 0.5)
        if safety and rng.random() < 0.3:
            safety = Safety(safety.label, (Risk("a risk", "PMID:1"),), safety.generic_applications)
        drugs.append(
            _drug(
                f"drug{i:02d}",
                tuple(rng.sample(symbols, rng.randint(1, 3))),
                properties,
                safety,
                rng.random() < 0.1,
            )
        )
    return DrugCache("2026-11-25", 1, targets, tuple(drugs))


def test_the_page_ranks_random_drug_caches_as_python_does():
    rng = random.Random(7)
    cases = [
        (_random_cache(seed), Weights(*(rng.choice([0, rng.random()]) for _ in range(4)), 0.05))
        for seed in range(200)
    ]
    assert_same(cases + [(_random_cache(seed), Weights()) for seed in range(50)])
