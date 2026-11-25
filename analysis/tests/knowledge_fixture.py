"""A small paper library and vocabulary for the knowledge graph and its map (M6-02, M6-04).

BDNF is in a PSSD paper and a PFS paper, so two syndromes link to it; it is co-mentioned with
serotonin in two papers (31000001, 31000003) and with COMT in one (31000002).

A drug cache (`DRUGS`) gives the leaderboard (M6-07) four drugs on BDNF and COMT: alphadrug on BDNF,
safe and generic; betadrug on COMT, its label naming libido and a listed risk; gammadrug on both, with
no US label and withdrawn; and sertraline, a known cause. The cache's target scores are not the
graph's, so a test can tell which the leaderboard used.

`with_androsterone` adds androsterone, in a paper of its own and linked to no syndrome, so the
built-in lab-report example's strong PSSD finding on it adds a finding link (M6-04b).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from hypothesis_engine.drugs import Action, Drug, DrugCache, Target, write_drug_cache
from hypothesis_engine.library import write_build
from hypothesis_engine.pubmed import Paper
from hypothesis_engine.safety import Label, Mention, Risk, Safety

from asclepius.run import Settings
from asclepius.web.graph import READY
from asclepius.web.hypotheses import Engine
from asclepius.web.server import App, make_server

VOCABULARY = """
[[term]]
name = "post-SSRI sexual dysfunction"
kind = "syndrome"
aliases = []
abbreviations = ["PSSD"]
pmid = "31000001"

[[term]]
name = "post-finasteride syndrome"
kind = "syndrome"
aliases = []
abbreviations = ["PFS"]
pmid = "31000002"

[[term]]
name = "BDNF"
kind = "gene"
aliases = []
pmid = "31000001"

[[term]]
name = "COMT"
kind = "gene"
aliases = []
pmid = "31000002"

[[term]]
name = "serotonin"
kind = "hormone"
aliases = []
pmid = "31000001"
"""

PAPERS = [
    Paper(
        "31000001",
        "Post-SSRI sexual dysfunction",
        "BDNF and serotonin fell in PSSD.",
        "10.1/pssd",
        None,
        "J Sex Med",
        2022,
    ),
    Paper(
        "31000002",
        "Post-finasteride syndrome",
        "BDNF and COMT changed in PFS.",
        None,
        None,
        "Andrology",
        2021,
    ),
    Paper(
        "31000003",
        "Serotonin and growth factors",
        "Serotonin and BDNF were measured.",
        None,
        None,
        None,
        2020,
    ),
]
TOPICS = {"31000001": [("PSSD", 1)], "31000002": [("PFS", 1)], "31000003": [("serotonin", 1)]}

ANDROSTERONE = """
[[term]]
name = "androsterone"
kind = "hormone"
aliases = []
pmid = "31000004"
"""
ANDROSTERONE_PAPER = Paper(
    "31000004", "Urinary androgens", "Androsterone was measured in urine.", None, None, None, 2019
)


GOOD_MPO = {"alogp": 2.0, "mw_freebase": 300.0, "psa": 50.0, "hbd": 0.0}  # CNS MPO 6 of 6
FAIR_MPO = {"alogp": 4.0, "mw_freebase": 430.0, "psa": 105.0, "hbd": 2.0}  # 3 of 6
PROPERTIES = ("alogp", "cx_logp", "cx_logd", "cx_most_bpka", "mw_freebase", "psa", "hbd", "hba")
LISTED_RISK = Risk("raises blood pressure", "PMID:31000009")


def _drug(name: str, targets: tuple[str, ...], mpo: dict[str, float], safety: Safety | None, **more) -> Drug:
    return Drug(
        f"CHEMBL_{name.upper()}",
        name,
        4.0,
        {k: mpo.get(k) for k in PROPERTIES},
        tuple(Action(t, None, "INHIBITOR", f"{t} inhibitor") for t in targets),
        more.get("known_cause"),
        "chembl",
        more.get("withdrawn", False),
        safety,
    )


DRUGS = DrugCache(
    "2026-11-25T00:00:00+00:00",
    1,
    (  # scores and syndromes unlike the graph's: BDNF scores 0.85 there, linked to PFS and PSSD
        Target("BDNF", 0.1, ("PAS",), ("CHEMBL1",), "chembl"),
        Target("COMT", 0.9, ("PAS", "PFS", "PSSD"), ("CHEMBL2",), "chembl"),
    ),
    (
        _drug("alphadrug", ("BDNF",), GOOD_MPO, Safety(Label("set-a", None, "NDA1", ()), (), ("ANDA1",))),
        _drug(
            "betadrug",
            ("COMT",),
            FAIR_MPO,
            Safety(
                Label("set-b", None, "NDA2", (Mention("libido", "adverse_reactions", "decreased libido"),)),
                (LISTED_RISK,),
                (),
            ),
        ),
        _drug("gammadrug", ("BDNF", "COMT"), FAIR_MPO, Safety(None, (), ()), withdrawn=True),
        _drug("SERTRALINE", ("BDNF",), GOOD_MPO, Safety(None, (), ()), known_cause="SSRI"),
    ),
)


def knowledge_engine(folder: Path, with_androsterone: bool = False) -> Engine:
    """An engine whose library and vocabulary are the ones above, written into `folder`."""
    papers, topics, vocabulary = PAPERS, TOPICS, VOCABULARY
    if with_androsterone:
        papers = [*PAPERS, ANDROSTERONE_PAPER]
        topics = TOPICS | {"31000004": [("androsterone", 1)]}
        vocabulary = VOCABULARY + ANDROSTERONE
    write_build(folder / "library.sqlite", papers, topics, "sha", "2026-11-24", [])
    (folder / "vocabulary.toml").write_text(vocabulary, encoding="utf-8")
    write_drug_cache(folder / "drugs.json", DRUGS)
    return Engine(
        library=folder / "library.sqlite", vocabulary=folder / "vocabulary.toml", drugs=folder / "drugs.json"
    )


@contextmanager
def serving(folder: Path, with_androsterone: bool = False) -> Iterator[tuple[App, str]]:
    """The app over that library, its graph built, and its address; stopped afterwards. Its runs use
    fast settings."""
    server = make_server(
        folder / "work",
        port=0,
        settings=Settings(permutations=999, bootstrap=200),
        engine=knowledge_engine(folder, with_androsterone),
    )
    server.knowledge.start()
    assert server.knowledge.wait(timeout=30) == READY
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
