"""Hypotheses: an evidence bundle turned by the LLM into the fixed five sections (spec H2, H6, H7).

1. The finding in the data: written here from the bundle, never by the LLM, with its strength
   label and its labels ("based on self-reports, no controls" first for a self-report finding).
2. The proposed mechanism: claims, each tied to a passage of the bundle and a quote from it.
3. Supporting papers and 4. contradicting papers: passages of the bundle the LLM judged to
   support or contradict, each with a quote and why.
5. How to test it: tests, each naming the kind of measure it uses. A self-report finding needs
   one whose measure is not a self-report.

The LLM answers in JSON. A reply that is not the JSON asked for, cites a passage the bundle
does not hold, or (self-report findings) tests only by self-reports is sent back with what was
wrong, up to `ATTEMPTS` times in all; then the finding is reported as a `Failure`. Whether a
quote is found word for word in its paper is checked afterwards, in `citations.py` (spec H1).

Only the finding's group statistics and the library's public passages are sent. Private
papers' passages are left out of the prompt, so their text never reaches an LLM here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from hypothesis_engine.evidence import (
    SELF_REPORT,
    EvidenceBundle,
    Passage,
    direction,
    is_report,
    worth_testing,
)
from hypothesis_engine.llm import LLM, Completion, LLMError, Message

PROMPT_VERSION = "hypothesis-1"
ATTEMPTS = 3
SELF_REPORT_LABEL = "based on self-reports, no controls"  # section 1 of a self-report finding (H7)
SELF_REPORT_MEASURE = "self-report"
MEASURES = (
    "lab test",
    "physical examination",
    "nerve or physiological recording",
    "imaging",
    "tissue biopsy",
    "genetic test",
    "animal or cell study",
    SELF_REPORT_MEASURE,
)
SECTIONS = ("mechanism", "supporting", "contradicting", "test")

SYSTEM = f"""You help a researcher turn one finding from patient data into a testable hypothesis.
The finding is about groups of patients, never one person. You are given the finding and
numbered passages from papers in the researcher's library.

Reply with one JSON object and nothing else, with exactly these keys:

- "mechanism": a list of claims that together propose how the finding could come about.
  Each is {{"claim": "...", "passage_id": "...", "quote": "..."}}: one claim, the id of the
  passage that backs it, and a quote copied word for word from that passage.
- "supporting": the passages that support the finding or the mechanism. Each is
  {{"passage_id": "...", "quote": "...", "why": "..."}}, the quote copied word for word.
- "contradicting": the passages that contradict them, in the same form. An empty list when none do.
- "test": a list of ways to test the hypothesis. Each is {{"design": "...", "measure": "..."}},
  where measure is the kind of data the test relies on, one of: {", ".join(MEASURES)}.

Rules:
- Every claim and every entry cites one of the given passage ids; cite no other source.
- A passage offered as "supporting" may in truth contradict, and the other way round: judge
  each one yourself, and file it under one section only. Leave out passages that do neither.
- Quotes are copied exactly from the passage, not reworded.
- Say nothing about treating patients."""

SELF_REPORT_RULE = f"""
- This finding comes from self-reports, with no controls. At least one test must not rely on self-reports:
  its measure must be something other than "{SELF_REPORT_MEASURE}"."""


@dataclass(frozen=True)
class FindingSection:
    """Section 1: the finding in the data, linked to the analysis result it came from."""

    statement: str
    label: str | None  # strong, suggestive
    labels: tuple[str, ...]  # SELF_REPORT_LABEL first when it applies, then the export's
    comparison: str
    marker_id: str
    sex: str | None
    group_a: str | None
    group_b: str | None
    statistics: dict[str, int | float | str | None]
    analysis_tool: str
    dataset_fingerprint: str
    run_date: str | None


@dataclass(frozen=True)
class Claim:
    """One step of the proposed mechanism and the passage behind it."""

    text: str
    passage_id: str
    paper_id: str
    quote: str


@dataclass(frozen=True)
class Citation:
    """A paper's passage judged to support or contradict, with the words that do."""

    passage_id: str
    paper_id: str
    title: str
    quote: str
    why: str


@dataclass(frozen=True)
class ProposedTest:
    """A way to test the hypothesis, and the kind of data it relies on (one of MEASURES)."""

    design: str
    measure: str


@dataclass(frozen=True)
class Hypothesis:
    """One finding's hypothesis in the spec's five sections, and what produced it."""

    finding: FindingSection
    mechanism: tuple[Claim, ...]
    supporting: tuple[Citation, ...]
    contradicting: tuple[Citation, ...]
    tests: tuple[ProposedTest, ...]
    record: dict[str, object]  # what produced it (spec H6)

    def as_dict(self) -> dict[str, object]:
        """The five sections in order, then the record, as plain JSON-ready values."""
        return {
            "finding": asdict(self.finding),
            "mechanism": [asdict(c) for c in self.mechanism],
            "supporting": [asdict(c) for c in self.supporting],
            "contradicting": [asdict(c) for c in self.contradicting],
            "test": [asdict(t) for t in self.tests],
            "record": self.record,
        }


@dataclass(frozen=True)
class Failure:
    """A finding the LLM gave no usable hypothesis for, and why, attempt by attempt."""

    finding: FindingSection
    reasons: tuple[str, ...]
    record: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """The failure as plain JSON-ready values."""
        return {"finding": asdict(self.finding), "reasons": list(self.reasons), "record": self.record}


class Malformed(Exception):
    """A reply that is not the hypothesis asked for; its message is sent back to the LLM."""


def generate_hypotheses(
    bundles: Iterable[EvidenceBundle], llm: LLM, attempts: int = ATTEMPTS
) -> list[Hypothesis | Failure]:
    """One hypothesis, or one failure, per bundle, in order."""
    return [generate_hypothesis(b, llm, attempts) for b in bundles]


def generate_hypothesis(bundle: EvidenceBundle, llm: LLM, attempts: int = ATTEMPTS) -> Hypothesis | Failure:
    """The bundle's hypothesis; a reply that is not usable is sent back with what was wrong,
    and after `attempts` replies (or an LLM error) the finding is reported as a `Failure`."""
    section = finding_section(bundle)
    passages = {p.passage_id: p for p in (*bundle.supporting, *bundle.contradicting) if not p.private}
    record: dict[str, object] = {
        "prompt_version": PROMPT_VERSION,
        "library_version": bundle.library_version,
        "search_model": bundle.search_model,
        "analysis_tool": bundle.finding.tool,
        "backend": llm.settings.backend,
        "model": llm.settings.model,
        "settings": llm.settings.public(),
        "attempts": 0,
    }
    if not worth_testing(bundle.finding):
        return Failure(section, (f"a {bundle.finding.label} finding gets no hypothesis (spec H2)",), record)
    if not passages:
        return Failure(section, ("the library holds no public passage about this finding",), record)

    messages = prompt(bundle, list(passages.values()))
    reasons: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            completion = llm.complete(messages)
        except LLMError as error:
            reasons.append(str(error))
            break
        record |= _answered_by(completion) | {"attempts": attempt}
        try:
            mechanism, supporting, contradicting, tests = parse(completion.text, passages, bundle.self_report)
        except Malformed as error:
            reasons.append(str(error))
            messages = [
                *messages,
                {"role": "assistant", "content": completion.text},
                {
                    "role": "user",
                    "content": f"That reply cannot be used: {error}. Reply again with only the JSON object.",
                },
            ]
            continue
        return Hypothesis(section, mechanism, supporting, contradicting, tests, record)
    return Failure(section, tuple(reasons), record)


def _answered_by(completion: Completion) -> dict[str, object]:
    return {"backend": completion.backend, "model": completion.model, "settings": completion.settings}


def prompt(bundle: EvidenceBundle, passages: Sequence[Passage]) -> list[Message]:
    """The messages asking for the bundle's hypothesis: instructions, then the finding and passages."""
    offered = {p.passage_id: "supporting" for p in bundle.supporting} | {
        p.passage_id: "contradicting" for p in bundle.contradicting
    }
    data = {
        "finding": bundle.finding.as_dict() | {"labels": list(bundle.labels)},
        "passages": [
            {
                "passage_id": p.passage_id,
                "paper_title": p.title,
                "offered_as": offered[p.passage_id],
                "text": p.text,
            }
            for p in passages
        ],
    }
    system = SYSTEM + (SELF_REPORT_RULE if bundle.self_report else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False)},
    ]


def parse(
    reply: str, passages: dict[str, Passage], self_report: bool
) -> tuple[tuple[Claim, ...], tuple[Citation, ...], tuple[Citation, ...], tuple[ProposedTest, ...]]:
    """The four LLM-written sections of a reply; raises `Malformed` saying what is wrong."""
    answer = _json_object(reply)
    if set(answer) != set(SECTIONS):
        missing, extra = set(SECTIONS) - set(answer), set(answer) - set(SECTIONS)
        raise Malformed(
            "the object must have exactly the keys "
            + ", ".join(SECTIONS)
            + (f"; missing {', '.join(sorted(missing))}" if missing else "")
            + (f"; unexpected {', '.join(sorted(extra))}" if extra else "")
        )

    def cited(entry: dict[str, str], where: str) -> Passage:
        passage = passages.get(entry["passage_id"])
        if passage is None:
            raise Malformed(f'{where} cites passage "{entry["passage_id"]}", which was not given')
        return passage

    mechanism = tuple(
        Claim(e["claim"], e["passage_id"], cited(e, "a mechanism claim").paper_id, e["quote"])
        for e in _entries(answer, "mechanism", ("claim", "passage_id", "quote"), required=True)
    )
    filed: dict[str, tuple[Citation, ...]] = {}
    for stance in ("supporting", "contradicting"):
        filed[stance] = tuple(
            Citation(
                e["passage_id"], (p := cited(e, f"a {stance} entry")).paper_id, p.title, e["quote"], e["why"]
            )
            for e in _entries(answer, stance, ("passage_id", "quote", "why"), required=False)
        )
    both = {c.passage_id for c in filed["supporting"]} & {c.passage_id for c in filed["contradicting"]}
    if both:
        raise Malformed(f"passage {', '.join(sorted(both))} is filed as both supporting and contradicting")
    tests = tuple(
        ProposedTest(e["design"], e["measure"])
        for e in _entries(answer, "test", ("design", "measure"), required=True)
    )
    unknown = sorted({t.measure for t in tests} - set(MEASURES))
    if unknown:
        raise Malformed(f"measure {', '.join(unknown)} is not one of: {', '.join(MEASURES)}")
    if self_report and all(t.measure == SELF_REPORT_MEASURE for t in tests):
        raise Malformed(
            "the finding comes from self-reports, so at least one test must not rely on self-reports "
            f'(a measure other than "{SELF_REPORT_MEASURE}")'
        )
    return mechanism, filed["supporting"], filed["contradicting"], tests


def _json_object(reply: str) -> dict[str, Any]:
    """The reply's JSON object, also when the model wrapped it in a code fence or a sentence."""
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end < start:
        raise Malformed("the reply holds no JSON object")
    try:
        answer = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as error:
        raise Malformed(f"the reply is not valid JSON ({error.msg} at character {error.pos})") from None
    if not isinstance(answer, dict):
        raise Malformed("the reply holds no JSON object")
    return answer


def _entries(
    answer: dict[str, Any], section: str, keys: tuple[str, ...], required: bool
) -> list[dict[str, str]]:
    entries = answer[section]
    if not isinstance(entries, list):
        raise Malformed(f'"{section}" must be a list')
    if required and not entries:
        raise Malformed(f'"{section}" must not be empty')
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != set(keys):
            raise Malformed(
                f'each entry of "{section}" must be an object with exactly the keys {", ".join(keys)}'
            )
        empty = [k for k in keys if not isinstance(entry[k], str) or not entry[k].strip()]
        if empty:
            raise Malformed(f'an entry of "{section}" has no text in {", ".join(empty)}')
    return [{k: entry[k].strip() for k in keys} for entry in entries]


# --- section 1 --------------------------------------------------------------------------


def finding_section(bundle: EvidenceBundle) -> FindingSection:
    """Section 1, from the bundle alone."""
    finding = bundle.finding
    # the bundle's labels, "self-report, no controls" (first when it applies) in the spec's words (H7)
    labels = tuple(SELF_REPORT_LABEL if label == SELF_REPORT else label for label in bundle.labels)
    return FindingSection(
        statement=statement(bundle),
        label=finding.label,
        labels=labels,
        comparison=finding.comparison,
        marker_id=finding.marker_id,
        sex=finding.sex,
        group_a=finding.group_a,
        group_b=finding.group_b,
        statistics=dict(finding.statistics),
        analysis_tool=finding.tool,
        dataset_fingerprint=finding.dataset_fingerprint,
        run_date=finding.run_date,
    )


def statement(bundle: EvidenceBundle) -> str:
    """The finding in one line, from its names and statistics: what differs, by how much, how sure."""
    finding, stats = bundle.finding, bundle.finding.statistics
    sign = direction(finding)
    a, b = finding.group_a or "the group", finding.group_b
    sex = f" ({finding.sex})" if finding.sex else ""
    if is_report(finding):
        how = {1: "more", -1: "less", 0: "as"}[sign]
        what = f"{finding.marker_name} reported {how} often after {a}" + (f" than after {b}" if b else "")
        numbers = [f"{_percent(stats.get('share_a'))} vs {_percent(stats.get('share_b'))}"]
        if isinstance(stats.get("odds_ratio"), float):
            numbers.append(
                f"odds ratio {_number(stats['odds_ratio'])}"
                + _interval(stats.get("odds_ratio_low"), stats.get("odds_ratio_high"))
            )
    else:
        how = {1: "higher", -1: "lower", 0: "differs"}[sign]
        what = f"{finding.marker_name} {how} in {a}" + (f" than in {b}" if b else "")
        numbers = []
        if isinstance(stats.get("effect_size"), float):
            numbers.append(
                f"effect size {_number(stats['effect_size'])}"
                + _interval(stats.get("ci_low"), stats.get("ci_high"))
            )
    if isinstance(stats.get("q_value"), float):
        numbers.append(f"q {_number(stats['q_value'])}")
    sizes = [stats.get("n_a"), stats.get("n_b")]
    if all(isinstance(n, int) for n in sizes):
        numbers.append(f"n {sizes[0]} and {sizes[1]}")
    return f"{what}{sex}: " + ", ".join(numbers) + f"; {finding.label}"


def _number(value: object) -> str:
    return f"{value:.2g}" if isinstance(value, float) and abs(value) < 0.1 else f"{value:.2f}"


def _percent(value: object) -> str:
    return f"{value:.0%}" if isinstance(value, float) else "?"


def _interval(low: object, high: object) -> str:
    if isinstance(low, float) and isinstance(high, float):
        return f" (95% CI {_number(low)} to {_number(high)})"
    return ""
