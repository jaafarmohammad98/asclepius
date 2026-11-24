"""Re-record the responses `test_build.py` and `test_fulltext.py` replay. Needs the network;
run only by hand:

    uv run python tests/record_responses.py build      # PubMed answers, fixtures/build
    uv run python tests/record_responses.py fulltext   # Europe PMC answers, fixtures/fulltext
    uv run python tests/record_responses.py drugs      # ChEMBL and MyChem.info answers, fixtures/drugs
    uv run python tests/record_responses.py safety     # openFDA answers, fixtures/openfda

Then update the literal values in those tests (and `fulltext_fixture.py`) if the answers changed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import drugs_fixture  # noqa: E402
import fulltext_fixture  # noqa: E402
from replay import ReplayTransport, recording, response_path  # noqa: E402

from hypothesis_engine import drugs, europepmc, safety  # noqa: E402
from hypothesis_engine.build import build_library  # noqa: E402
from hypothesis_engine.fulltext import fetch_full_texts  # noqa: E402
from hypothesis_engine.pubmed import PubMed, RateLimiter, TransientError, Transport, http_transport  # noqa: E402
from hypothesis_engine.topics import load_topics  # noqa: E402

FOLDER = Path(__file__).parent / "fixtures" / "build"
FETCH_BATCH = 4  # small, so the recording covers more than one efetch request


def record_full_text() -> None:
    folder = fulltext_fixture.FOLDER
    folder.mkdir(exist_ok=True)
    for old in folder.glob("*"):
        old.unlink()
    scratch = folder.parent / "_recording.sqlite"
    scratch.unlink(missing_ok=True)
    fulltext_fixture.make_library(scratch)
    lookup = europepmc.EuropePMC(
        recording(europepmc.http_transport(), folder),
        RateLimiter(europepmc.RATE),
        lookup_batch=fulltext_fixture.LOOKUP_BATCH,
    )
    pmc = PubMed(recording(http_transport(), folder), RateLimiter(3))
    print(fetch_full_texts(scratch, lookup, pmc, download_batch=fulltext_fixture.DOWNLOAD_BATCH))
    scratch.unlink()


def record_build() -> None:
    for old in [*FOLDER.glob("esearch-*"), *FOLDER.glob("efetch-*")]:
        old.unlink()
    scratch = Path(__file__).parent / "fixtures" / "_recording.sqlite"
    scratch.unlink(missing_ok=True)
    pubmed = PubMed(recording(http_transport(), FOLDER), RateLimiter(3), fetch_batch=FETCH_BATCH)
    report = build_library(scratch, load_topics(FOLDER / "topics.toml"), pubmed)
    scratch.unlink()
    print(report)


def record_drugs() -> None:
    """ChEMBL often answers HTTP 500 for a while: answers already recorded are kept (`--fresh`
    deletes them first), and a target is asked again until it is answered."""
    folder = drugs_fixture.FOLDER
    folder.mkdir(exist_ok=True)
    if "--fresh" in sys.argv:
        for old in folder.glob("*"):
            old.unlink()

    def kept(transport: Transport) -> Transport:
        def send(path: str, params: dict[str, str]) -> bytes:
            stored = response_path(folder, path, params)
            return stored.read_bytes() if stored.exists() else transport(path, params)

        return send

    def service(base: str) -> drugs.Service:
        return drugs.Service(kept(recording(drugs.http_transport(base), folder)), RateLimiter(drugs.RATE))

    chembl, mychem = drugs.ChEMBL(service(drugs.CHEMBL)), drugs.MyChem(service(drugs.MYCHEM))
    for symbol in drugs_fixture.SYMBOLS:
        for _attempt in range(10):
            try:
                found = chembl.drugs(symbol, chembl.targets(symbol))
                break
            except TransientError as error:
                print(symbol, "ChEMBL failed, asking again:", error)
        else:
            raise SystemExit(f"{symbol}: ChEMBL did not answer")
        print(symbol, "ChEMBL:", sorted(d.name for d in found))
    for symbol in drugs_fixture.MYCHEM_SYMBOLS:
        print(symbol, "MyChem.info:", sorted(d.name for d in mychem.drugs(symbol)))


def _slim(path: str, body: bytes) -> bytes:
    """Only the parts of an openFDA answer `safety.py` reads (its `*_FIELDS`): a label is some
    150 kB otherwise."""
    answer = json.loads(body)
    if "results" not in answer:
        return body

    def only(record: dict, fields: tuple[str, ...]) -> dict:
        return {k: v for k, v in record.items() if k in fields}

    if path == "drug/label.json":
        results = [
            {
                **only(r, safety.LABEL_FIELDS),
                "openfda": only(r.get("openfda", {}), safety.LABEL_OPENFDA_FIELDS),
            }
            for r in answer["results"]
        ]
    else:
        results = [
            {
                **only(r, safety.APPLICATION_FIELDS),
                "products": [
                    {
                        **only(p, safety.PRODUCT_FIELDS),
                        "active_ingredients": [
                            only(a, safety.INGREDIENT_FIELDS) for a in p.get("active_ingredients", [])
                        ],
                    }
                    for p in r.get("products", [])
                ],
            }
            for r in answer["results"]
        ]
    return json.dumps({"meta": {"results": answer["meta"]["results"]}, "results": results}).encode()


def record_safety() -> None:
    """The labels and generics of every drug recorded for `drugs_fixture.SAFETY_SYMBOLS`."""
    folder = drugs_fixture.SAFETY_FOLDER
    folder.mkdir(exist_ok=True)
    for old in folder.glob("*"):
        old.unlink()
    network = drugs.http_transport(safety.OPENFDA, not_found_ok=True)

    def slimmed(path: str, params: dict[str, str]) -> bytes:
        return _slim(path, network(path, params))

    openfda = safety.OpenFDA(drugs.Service(recording(slimmed, folder), RateLimiter(drugs.RATE)), {})
    replayed = drugs.ChEMBL(drugs.Service(ReplayTransport(drugs_fixture.FOLDER), RateLimiter(100)))
    names = sorted(
        {d.name for s in drugs_fixture.SAFETY_SYMBOLS for d in replayed.drugs(s, replayed.targets(s))}
    )
    for name in names:
        found = openfda.safety(name)
        print(
            name,
            found.concerns,
            "generic" if found.generic else "no generic",
            found.label and found.label.set_id,
        )


def main() -> None:
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["build", "fulltext", "drugs", "safety"]
    for name in which:
        {"build": record_build, "fulltext": record_full_text, "drugs": record_drugs, "safety": record_safety}[
            name
        ]()
    shutil.rmtree(Path(__file__).parent / "fixtures" / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
