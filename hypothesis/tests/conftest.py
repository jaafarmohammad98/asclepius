"""Fixtures shared by the hypothesis and citation tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from test_evidence import built  # noqa: F401  (a fixture, shared from here)
from test_hypotheses import LOCAL, FakeServer, bundle

from hypothesis_engine.evidence import EvidenceBundle
from hypothesis_engine.hypotheses import Failure, Hypothesis, generate_hypotheses
from hypothesis_engine.library import Library
from hypothesis_engine.llm import connect


@pytest.fixture
def library(built):  # noqa: F811
    """The library built from `test_evidence.PAPERS`, open."""
    with Library.open(built[0]) as opened:
        yield opened


@pytest.fixture
def generate(library: Library) -> Callable[..., list[Hypothesis | Failure]]:
    """Hypotheses from `server`'s replies for `bundles` (by default the lab finding's)."""

    def run(server: FakeServer, *bundles: EvidenceBundle) -> list[Hypothesis | Failure]:
        return generate_hypotheses(bundles or (bundle(),), connect(LOCAL, local_transport=server), library)

    return run
