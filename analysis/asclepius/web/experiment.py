"""The lab experiment builder in the web app (M6-09, spec D1, D2, D4): a two-level design over factors
picked from the drug repurposing leaderboard (compounds) or the bottleneck table (targets), each at
present and absent.

The design is built here, by `hypothesis_engine.doe.design`, never in the browser: its seed reproduces
the run order only in Python (agreed 2026-11-25). The page shows `Design.label` and `Design.note` as
they are.

Each factor is a candidate for study with its known risks listed (rule 6), beside the design and in
its CSV. They are looked up here, never taken from the request: a drug's are the leaderboard's (its
label unread said as unknown, never as none, and none found said as what was checked); a target has
none of its own, and says so. The server writes each factor's risks as one sentence (`risks_said`)
that the page and the CSV show as they are. A drug that
causes the syndromes is never ranked, so it cannot be a factor; nor can a symptom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hypothesis_engine.doe import LETTERS, MAX_FACTORS, MIN_FACTORS, SEED_LIMIT, design
from hypothesis_engine.hypotheses import CANDIDATE

from asclepius.web.graph import BUILDING, FAILED, READY, KnowledgeGraph
from asclepius.web.leaderboard import leaderboard_view

DRUG, TARGET = "drug", "target"
SYMPTOM = "symptom"
MAX_REPLICATES = 20
MAX_RESPONSES = 10
NONE_FOUND = "none on its US label or listed"  # what was checked: never a bare "none"
TARGET_RISKS = "none listed: a target, not a compound; the risks of the compound used to act on it apply"


class ExperimentError(ValueError):
    """A request the builder refuses, with what to change."""


def experiment_view(request: object, drugs: Path, knowledge: KnowledgeGraph) -> dict[str, Any]:
    """The design for the factors `request` names (`{"factors": [{"kind", "name"}...], "seed",
    "replicates", "responses"}`), with each factor's risks, its runs in run order and its CSV."""
    if not isinstance(request, dict):
        raise ExperimentError("The experiment must be a JSON object naming its factors.")
    picked = request.get("factors")
    if not isinstance(picked, list) or not all(isinstance(f, dict) for f in picked):
        raise ExperimentError("The factors must be a list of {kind, name}.")
    if not MIN_FACTORS <= len(picked) <= MAX_FACTORS:
        raise ExperimentError(f"A design takes {MIN_FACTORS} to {MAX_FACTORS} factors, not {len(picked)}.")
    seed, replicates, responses = _settings(request)
    factors = _Factors(drugs, knowledge)
    resolved = [factors.resolve(f.get("kind"), f.get("name")) for f in picked]
    try:
        built = design([f["name"] for f in resolved], seed=seed, replicates=replicates)
        text = built.csv(responses, risks={f["name"]: f["risks_said"] for f in resolved})
    except ValueError as error:
        raise ExperimentError(str(error)) from error
    return {
        "label": built.label,
        "note": built.note,
        "seed": built.seed,
        "replicates": built.replicates,
        "responses": responses,
        "factors": [{**f, "letter": LETTERS[i]} for i, f in enumerate(resolved)],
        "runs": [
            {"run_id": run_id, "run_order": position, "levels": list(built.matrix[run_id - 1])}
            for position, run_id in enumerate(built.run_order, start=1)
        ],
        "csv": text,
        "filename": f"asclepius-experiment-{built.seed}.csv",
    }


def _settings(request: dict[str, Any]) -> tuple[int | None, int, list[str]]:
    seed = request.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < SEED_LIMIT):
        raise ExperimentError(f"The seed must be a whole number from 0 to {SEED_LIMIT - 1}, or left out.")
    replicates = request.get("replicates", 1)
    if type(replicates) is not int or not 1 <= replicates <= MAX_REPLICATES:
        raise ExperimentError(f"The replicate count must be a whole number from 1 to {MAX_REPLICATES}.")
    responses = request.get("responses", ["response"])
    if (
        not isinstance(responses, list)
        or not 1 <= len(responses) <= MAX_RESPONSES
        or not all(isinstance(r, str) and r.strip() for r in responses)
    ):
        raise ExperimentError(f"Name 1 to {MAX_RESPONSES} response columns, none of them empty.")
    return seed, replicates, [" ".join(r.split()) for r in responses]


class _Factors:
    """Looks factors up in the leaderboard and the knowledge graph, each read once, when first needed."""

    def __init__(self, drugs: Path, knowledge: KnowledgeGraph) -> None:
        self.drugs, self.knowledge = drugs, knowledge
        self._board: dict[str, Any] | None = None

    def resolve(self, kind: object, name: object) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise ExperimentError("Each factor needs a name.")
        name = name.strip()
        if kind == DRUG:
            return self._drug(name)
        if kind == TARGET:
            return self._target(name)
        raise ExperimentError(f"A factor is a drug or a target, not {kind!r}.")

    def _drug(self, name: str) -> dict[str, Any]:
        board = self._board = self._board or leaderboard_view(self.drugs, self.knowledge)
        if board["state"] == BUILDING:
            raise ExperimentError("Wait for the knowledge graph to score the drugs' targets.")
        if board["state"] != READY:
            raise ExperimentError(board["message"])
        if any(d["name"] == name for d in board["known_causes"]):
            raise ExperimentError(f"{name} is a known cause of PSSD, PFS or PAS, never a factor.")
        row = next((r for r in board["rows"] if r["name"] == name), None)
        if row is None:
            raise ExperimentError(f"{name} is not on the leaderboard.")
        said = "; ".join([*row["risks"], *filter(None, [row["unread"]])]) or NONE_FOUND
        return {"kind": DRUG, "name": name, "label": CANDIDATE, "risks": row["risks"], "risks_said": said}

    def _target(self, name: str) -> dict[str, Any]:
        current = self.knowledge.current()
        if current is None:
            if self.knowledge.state == FAILED:
                raise ExperimentError("The knowledge graph could not be built, so no target can be a factor.")
            raise ExperimentError("Wait for the knowledge graph to be built.")
        term = next((b for b in current[1] if b.name == name), None)
        if term is None:
            raise ExperimentError(f"{name} is not a term of the bottleneck table.")
        if term.kind == SYMPTOM:
            raise ExperimentError(f"{name} is a symptom: measure it as a response, not a factor.")
        return {"kind": TARGET, "name": name, "label": CANDIDATE, "risks": [], "risks_said": TARGET_RISKS}
