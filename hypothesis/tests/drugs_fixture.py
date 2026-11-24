"""The targets whose ChEMBL and MyChem.info answers are recorded in fixtures/drugs."""

from __future__ import annotations

from pathlib import Path

from hypothesis_engine.bottlenecks import Bottleneck

FOLDER = Path(__file__).parent / "fixtures" / "drugs"
# SRD5A2: finasteride and dutasteride, known causes. SLC6A4: the SSRIs, and drugs that are not.
# TSPO: alpidem, approved and withdrawn since.
SYMBOLS = ("SRD5A2", "SLC6A4", "TSPO", "AR")  # AR: a short symbol ChEMBL's search refuses
MYCHEM_SYMBOLS = ("SRD5A2",)  # recorded from MyChem.info too, for the fallback
SAFETY_FOLDER = Path(__file__).parent / "fixtures" / "openfda"
SAFETY_SYMBOLS = ("SRD5A2", "SLC6A4", "TSPO")  # the targets whose drugs' openFDA answers are recorded


def target(symbol: str, score: float = 0.5) -> Bottleneck:
    return Bottleneck(symbol, "gene", score, 0.0, (), 0.0)
