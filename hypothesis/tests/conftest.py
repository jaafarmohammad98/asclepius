"""Fixtures shared by the hypothesis and citation tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from test_evidence import built  # noqa: F401  (a fixture, shared from here)
from test_hypotheses import LOCAL, FakeServer, bundle
from test_search import WordEmbedder

from hypothesis_engine.evidence import EvidenceBundle
from hypothesis_engine.hypotheses import Failure, Hypothesis, generate_hypotheses
from hypothesis_engine.library import Library
from hypothesis_engine.llm import connect
from hypothesis_engine.search import Index


@pytest.fixture
def library(built):  # noqa: F811
    """The library built from `test_evidence.PAPERS`, open."""
    with Library.open(built[0]) as opened:
        yield opened


@pytest.fixture
def searcher(built):  # noqa: F811
    """The search index of that library, open."""
    library, path = built
    with Index.open(path, library, WordEmbedder()) as opened:
        yield opened


@pytest.fixture
def generate(library: Library, searcher: Index) -> Callable[..., list[Hypothesis | Failure]]:
    """Hypotheses from `server`'s replies for `bundles` (by default the lab finding's)."""

    def run(
        server: FakeServer, *bundles: EvidenceBundle, interventions: tuple[str, ...] = ()
    ) -> list[Hypothesis | Failure]:
        llm = connect(LOCAL, local_transport=server)
        return generate_hypotheses(
            bundles or (bundle(),), llm, library, searcher, interventions=interventions
        )

    return run
