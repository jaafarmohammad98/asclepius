"""Hypotheses: an evidence bundle turned by the LLM into the fixed five sections (spec H2, H6, H7).

1. The finding in the data: written here from the bundle, never by the LLM, with its strength
   label and its labels ("based on self-reports, no controls" first for a self-report finding).
2. The proposed mechanism: claims, each tied to a passage of the bundle and a quote from it.
3. Supporting papers and 4. contradicting papers: passages of the bundle the LLM judged to
   support or contradict, each with a quote and why.
5. How to test it: tests, each naming the kind of measure it uses. A self-report finding needs
   one whose measure is not a self-report.

Treatment candidates (spec H3) are the most dangerous output. Every intervention a hypothesis
names is a `Candidate`, labelled "candidate for study": an intervention finding's own, those the
LLM lists, those of `interventions.toml` or of the findings export found in its text, and those a
second LLM request finds in its text (`mentioned`); a hypothesis that check cannot be run on is a
`Failure`. Each gets its
known risks from the library (`risks_for`), every one with a verified citation, or "risks not
established in the library". Any output holding a candidate or an intervention carries `NOTICE`:
`Hypothesis.as_dict`, `Failure.as_dict` and `export` add it themselves, so it cannot be left out.

The LLM answers in JSON. A reply that is not the JSON asked for, cites a passage the bundle
does not hold, (self-report findings) tests only by self-reports, or quotes words not found
word for word in the cited paper (`citations.py`, spec H1) is sent back with what was wrong,
up to `ATTEMPTS` times in all. Then the best usable reply is kept with its failing citations
removed, so no unverified citation is ever output; with none, the finding is a `Failure`.

Only the finding's group statistics and the library's public passages are sent. Private
papers' passages are left out of the prompt, so their text never reaches an LLM here.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

from hypothesis_engine.citations import Problem, problem, quoted
from hypothesis_engine.evidence import (
    INTERVENTION,
    REPORTED,
    SELF_REPORT,
    EvidenceBundle,
    Passage,
    Searcher,
    direction,
    is_intervention,
    is_report,
    worth_testing,
)
from hypothesis_engine.library import Library
from hypothesis_engine.llm import LLM, Completion, LLMError, Message

PROMPT_VERSION = "hypothesis-3"  # 2: failing quotes are sent back; 3: interventions named are listed
RISK_PROMPT_VERSION = "risks-1"
MENTION_PROMPT_VERSION = "mentions-1"
VOCABULARY = Path(__file__).with_name("interventions.toml")
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
SECTIONS = ("mechanism", "supporting", "contradicting", "test", "candidates")
CANDIDATE = "candidate for study"  # on every intervention a hypothesis names (spec H3)
NO_RISKS = "risks not established in the library"
UNCHECKED = "risks not checked"  # the lookup did not finish: not a finding that there are none
NOTICE = (
    "For researchers only: treatment candidates are ideas for study, not treatment advice. "
    "No one should start, stop or change a treatment because of them."
)
RISK_PASSAGES = 8  # library passages naming an intervention offered for its risks, at most

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
- "candidates": every intervention (a drug, supplement, procedure or therapy) your claims, entries
  or tests name as something that could change the condition, each {{"intervention": "<its name>"}}.
  An empty list when none is named. A drug named only as the cause of the condition is not one.

Rules:
- Every claim and every entry cites one of the given passage ids; cite no other source.
- A passage offered as "supporting" may in truth contradict, and the other way round: judge
  each one yourself, and file it under one section only. Leave out passages that do neither.
- Quotes are copied exactly from the passage, not reworded.
- Name an intervention only as something to study, never as advice to patients or doctors."""

INTERVENTION_RULE = """
- This finding is an intervention's reported outcomes: counts of what patients said happened,
  not evidence that it works. Propose how it could act, and tests of whether it does."""

RISK_SYSTEM = """You list the known risks of one intervention for a researcher, from numbered
passages of papers in their library.

Reply with one JSON object and nothing else: {"risks": [...]}, each risk
{"risk": "...", "passage_id": "...", "quote": "..."}: a harm, side effect, interaction or warning
a passage states for this intervention, the id of that passage, and a quote copied word for word
from it that states the risk. An empty list when no passage states a risk of it.

Rules:
- Cite only the given passages, and add no risk you know of from elsewhere.
- Quotes are copied exactly from the passage, not reworded.
- Say nothing about whether anyone should use it."""

MENTION_SYSTEM = """You check a hypothesis written for a researcher for the interventions it
names. You are given its sentences.

Reply with one JSON object and nothing else: {"interventions": ["...", ...]}: every drug,
supplement, herb, diet, procedure, device or therapy the sentences name as something that could
treat, relieve or change the condition, or that a test would give to patients, each written
exactly as it appears in the sentences. An empty list when none is named.

Leave out a drug named only as the cause of the condition, and hormones, enzymes and other body
chemicals named only as markers or as part of the mechanism."""

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

    @property
    def is_intervention(self) -> bool:
        """An intervention's reported outcomes, not a finding of a test."""
        return self.comparison == INTERVENTION


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
class Risk:
    """A known risk of a candidate, as a library paper states it."""

    text: str
    passage_id: str
    paper_id: str
    title: str
    quote: str


@dataclass(frozen=True)
class Candidate:
    """An intervention a hypothesis names: a candidate for study, never advice (spec H3)."""

    intervention: str
    risks: tuple[Risk, ...]  # each citation verified (spec H1)
    unchecked: str | None  # why the risks could not be looked for, when they could not
    record: dict[str, object]  # the risk prompt's version, attempts, citations removed

    @property
    def label(self) -> str:
        """Always "candidate for study"."""
        return CANDIDATE

    @property
    def risks_note(self) -> str | None:
        """With no risk: "risks not established in the library", or "risks not checked" and why
        when the lookup did not finish."""
        if self.risks:
            return None
        return f"{UNCHECKED}: {self.unchecked}" if self.unchecked else NO_RISKS

    def as_dict(self) -> dict[str, object]:
        """The candidate with the notice, its label and its risks, or the note saying why there are none."""
        return {
            "notice": NOTICE,
            "intervention": self.intervention,
            "label": self.label,
            "risks": [asdict(r) for r in self.risks],
            "risks_note": self.risks_note,
            "record": self.record,
        }


@dataclass(frozen=True)
class Hypothesis:
    """One finding's hypothesis in the spec's five sections, its candidates, and what produced it."""

    finding: FindingSection
    mechanism: tuple[Claim, ...]
    supporting: tuple[Citation, ...]
    contradicting: tuple[Citation, ...]
    tests: tuple[ProposedTest, ...]
    record: dict[str, object]  # what produced it (spec H6)
    candidates: tuple[Candidate, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """The five sections in order, the candidates, then the record, as plain JSON-ready values;
        `NOTICE` first whenever there is a candidate or the finding is an intervention's."""
        return _noticed(
            bool(self.candidates) or self.finding.is_intervention,
            {
                "finding": asdict(self.finding),
                "mechanism": [asdict(c) for c in self.mechanism],
                "supporting": [asdict(c) for c in self.supporting],
                "contradicting": [asdict(c) for c in self.contradicting],
                "test": [asdict(t) for t in self.tests],
                "candidates": [c.as_dict() for c in self.candidates],
                "record": self.record,
            },
        )


@dataclass(frozen=True)
class Failure:
    """A finding the LLM gave no usable hypothesis for, and why, attempt by attempt."""

    finding: FindingSection
    reasons: tuple[str, ...]
    record: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """The failure as plain JSON-ready values; `NOTICE` first when the finding is an intervention's."""
        return _noticed(
            self.finding.is_intervention,
            {"finding": asdict(self.finding), "reasons": list(self.reasons), "record": self.record},
        )


def _noticed(applies: bool, out: dict[str, object]) -> dict[str, object]:
    return {"notice": NOTICE, **out} if applies else out


def export(results: Iterable[Hypothesis | Failure]) -> dict[str, object]:
    """What the `hypotheses` command writes: `{"hypotheses": [...], "failed": [...]}`, with
    `NOTICE` first whenever any of them holds a candidate or an intervention."""
    results = list(results)
    made = [r.as_dict() for r in results if isinstance(r, Hypothesis)]
    failed = [r.as_dict() for r in results if isinstance(r, Failure)]
    applies = any(r.finding.is_intervention or isinstance(r, Hypothesis) and r.candidates for r in results)
    return _noticed(applies, {"hypotheses": made, "failed": failed})


class Malformed(Exception):
    """A reply that is not the hypothesis asked for; its message is sent back to the LLM."""


def generate_hypotheses(
    bundles: Iterable[EvidenceBundle],
    llm: LLM,
    library: Library,
    searcher: Searcher,
    attempts: int = ATTEMPTS,
    interventions: Iterable[str] = (),
) -> list[Hypothesis | Failure]:
    """One hypothesis, or one failure, per bundle, in order; quotes checked against `library`,
    candidates' risks searched for with `searcher`. `interventions`: names of the findings
    export's interventions, each marked a candidate wherever a hypothesis mentions it; the
    bundles' own interventions are added."""
    bundles = list(bundles)
    known = [*vocabulary(), *interventions]
    return [generate_hypothesis(b, llm, library, searcher, attempts, known) for b in bundles]


@cache
def vocabulary() -> tuple[str, ...]:
    """The intervention names of `interventions.toml`, looked for in every hypothesis."""
    names = tomllib.loads(VOCABULARY.read_text(encoding="utf-8"))["names"]
    return tuple(str(n) for n in names)


def generate_hypothesis(
    bundle: EvidenceBundle,
    llm: LLM,
    library: Library,
    searcher: Searcher,
    attempts: int = ATTEMPTS,
    interventions: Iterable[str] = (),
) -> Hypothesis | Failure:
    """The bundle's hypothesis, every citation in it checked against the library (spec H1), and
    every intervention it names a candidate with its risks (spec H3).

    A reply that is not usable, or that has citations whose quote is not in their paper, is sent
    back with what was wrong. After `attempts` replies (or an LLM error) the best usable reply
    is kept: the one with the fewest failing citations, the later one on a tie. Its failing
    citations are removed; left with no mechanism claim, or with no usable reply, the finding
    is reported as a `Failure`."""
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
        what = (
            "an unranked intervention"
            if is_intervention(bundle.finding)
            else f"a {bundle.finding.label} finding"
        )
        return Failure(section, (f"{what} gets no hypothesis (spec H2)",), record)
    if not passages:
        return Failure(section, ("the library holds no public passage about this finding",), record)

    messages = prompt(bundle, list(passages.values()))
    reasons: list[str] = []
    best: _Checked | None = None
    for attempt in range(1, attempts + 1):
        try:
            completion = llm.complete(messages)
        except LLMError as error:
            reasons.append(str(error))
            break
        try:
            sections = parse(completion.text, passages, bundle.self_report)
        except Malformed as error:
            reasons.append(str(error))
            if best is None:
                record |= _answered_by(completion)
            feedback = f"That reply cannot be used: {error}."
        else:
            checked = _check(sections, library, bundle.library_version)
            if best is None or len(checked.problems) <= len(best.problems):
                best = checked
                record |= _answered_by(completion) | {"attempts": attempt}
            if not checked.problems:
                break
            reasons.append("; ".join(str(p) for p in checked.problems))
            feedback = _quotes_not_found(checked.problems)
        record["attempts"] = attempt
        messages = [
            *messages,
            {"role": "assistant", "content": completion.text},
            {"role": "user", "content": f"{feedback} Reply again with only the JSON object."},
        ]
    if best is None:
        return Failure(section, tuple(reasons), record)
    record["citations_removed"] = [str(p) for p in best.problems]
    if not best.mechanism:
        reason = "no mechanism claim is left once unverified citations are removed (spec H1)"
        return Failure(section, (*reasons, reason), record)
    try:
        checked_names = mentioned(_shown(best), llm, attempts, record)
    except LLMError as error:
        reason = f"the check for the interventions it names did not finish ({error}) (spec H3)"
        return Failure(section, (*reasons, reason), record)
    named = candidate_names(bundle, best, (*interventions, *checked_names))
    candidates = tuple(
        risks_for(name, llm, library, searcher, bundle.library_version, attempts) for name in named
    )
    return Hypothesis(
        section, best.mechanism, best.supporting, best.contradicting, best.tests, record, candidates
    )


@dataclass(frozen=True)
class _Checked:
    """A usable reply's sections with the failing citations taken out, and those citations."""

    mechanism: tuple[Claim, ...]
    supporting: tuple[Citation, ...]
    contradicting: tuple[Citation, ...]
    tests: tuple[ProposedTest, ...]
    candidates: tuple[str, ...]  # the interventions the LLM listed
    problems: tuple[Problem, ...]


Sections = tuple[
    tuple[Claim, ...], tuple[Citation, ...], tuple[Citation, ...], tuple[ProposedTest, ...], tuple[str, ...]
]


def _check(sections: Sections, library: Library, version: int) -> _Checked:
    mechanism, supporting, contradicting, tests, candidates = sections
    problems: list[Problem] = []

    def verified[C: (Claim, Citation)](cited: tuple[C, ...], what: str) -> tuple[C, ...]:
        kept = []
        for c in cited:
            found = problem(c, what, library, version)
            if found:
                problems.append(found)
            else:
                kept.append(c)
        return tuple(kept)

    return _Checked(
        verified(mechanism, "a mechanism claim"),
        verified(supporting, "a supporting entry"),
        verified(contradicting, "a contradicting entry"),
        tests,
        candidates,
        tuple(problems),
    )


def _quotes_not_found(problems: Sequence[Problem]) -> str:
    """The message sending failing citations back, quotes included: the LLM wrote them."""
    lines = [f'- {p.what} citing passage {p.passage_id}, quote "{p.quote}": it {p.why}' for p in problems]
    return (
        "These citations cannot be shown:\n"
        + "\n".join(lines)
        + "\nCopy each quote exactly from the text of the passage it cites, or leave the entry out."
    )


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
    system = SYSTEM + (INTERVENTION_RULE if is_intervention(bundle.finding) else "")
    system += SELF_REPORT_RULE if bundle.self_report else ""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False)},
    ]


def parse(reply: str, passages: dict[str, Passage], self_report: bool) -> Sections:
    """The four LLM-written sections of a reply and its candidates; raises `Malformed` saying what is wrong."""
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
    candidates = tuple(
        e["intervention"] for e in _entries(answer, "candidates", ("intervention",), required=False)
    )
    return mechanism, filed["supporting"], filed["contradicting"], tests, candidates


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


# --- treatment candidates (spec H3) ------------------------------------------------------


def _shown(checked: _Checked) -> list[str]:
    """Everything a hypothesis shows beyond section 1, piece by piece."""
    return (
        [piece for c in checked.mechanism for piece in (c.text, c.quote)]
        + [
            piece
            for c in (*checked.supporting, *checked.contradicting)
            for piece in (c.title, c.quote, c.why)
        ]
        + [t.design for t in checked.tests]
    )


def candidate_names(bundle: EvidenceBundle, checked: _Checked, interventions: Iterable[str]) -> list[str]:
    """Every intervention the hypothesis names, once each (case aside), in this order: the
    finding's own when it is an intervention's, those the LLM listed, then those of
    `interventions` found anywhere in what the hypothesis shows but not listed. The finding's
    group names (the drug said to cause the condition) are not looked for."""
    shown = " ".join(_shown(checked))
    groups = {g.casefold() for g in (bundle.finding.group_a, bundle.finding.group_b) if g}
    own = [bundle.finding.marker_name] if is_intervention(bundle.finding) else []
    found = [n for n in interventions if n.casefold() not in groups and _mentions(n, shown)]
    names: dict[str, str] = {}
    for name in (*own, *checked.candidates, *found):
        names.setdefault(name.casefold(), name)
    return list(names.values())


def _mentions(name: str, text: str) -> bool:
    """Whether `text` names the intervention, as a whole word or words, case aside."""
    words = r"[\s_-]+".join(re.escape(w) for w in re.split(r"[\s_-]+", name.strip()) if w)
    return bool(words) and re.search(rf"(?<!\w){words}(?!\w)", text, re.IGNORECASE) is not None


def mentioned(pieces: Sequence[str], llm: LLM, attempts: int, record: dict[str, object]) -> list[str]:
    """The interventions a second LLM request finds named in `pieces` (the hypothesis's text), so
    one neither listed by the first reply nor in the vocabulary is still a candidate. A name not
    found word for word in the text is sent back, then dropped. Raises `LLMError` when no usable
    reply comes, since a hypothesis left unchecked may name an unlabelled intervention."""
    text = " ".join(pieces)
    data = {"sentences": list(pieces)}
    messages: list[Message] = [
        {"role": "system", "content": MENTION_SYSTEM},
        {"role": "user", "content": json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False)},
    ]
    check: dict[str, object] = {"prompt_version": MENTION_PROMPT_VERSION, "attempts": 0, "names_dropped": []}
    record["mention_check"] = check
    best: list[str] | None = None
    for attempt in range(1, attempts + 1):
        check["attempts"] = attempt
        completion = llm.complete(messages)
        try:
            names = _mention_names(completion.text)
        except Malformed as error:
            feedback = f"That reply cannot be used: {error}."
        else:
            best = [n for n in names if _mentions(n, text)]
            missing = [n for n in names if not _mentions(n, text)]
            check["names_dropped"] = missing
            if not missing:
                break
            feedback = (
                f"These names are not found word for word in the sentences: {', '.join(missing)}. "
                "Write each intervention exactly as the sentences do."
            )
        messages = [
            *messages,
            {"role": "assistant", "content": completion.text},
            {"role": "user", "content": f"{feedback} Reply again with only the JSON object."},
        ]
    if best is None:
        raise LLMError("no usable reply listing the interventions named")
    return best


def _mention_names(reply: str) -> list[str]:
    answer = _json_object(reply)
    if set(answer) != {"interventions"}:
        raise Malformed('the object must have exactly the key "interventions"')
    names = answer["interventions"]
    if not isinstance(names, list) or not all(isinstance(n, str) and n.strip() for n in names):
        raise Malformed('"interventions" must be a list of names')
    return [n.strip() for n in names]


def risk_queries(name: str) -> list[str]:
    """The searches for an intervention's known risks: names only."""
    return [
        f"adverse effects of {name}",
        f"{name} side effects",
        f"risks and safety of {name}",
        f"{name} worsened symptoms",
    ]


def risk_passages(name: str, searcher: Searcher, k: int = RISK_PASSAGES) -> list[Passage]:
    """Up to `k` public library passages naming the intervention, found by `risk_queries`, best
    first. A passage that does not name it cannot state its risks; private passages are never sent."""
    best: dict[str, Passage] = {}
    for query in risk_queries(name):
        for hit in searcher.search(query, k):
            if hit.private or hit.text.strip() == hit.title.strip() or not _mentions(name, hit.text):
                continue
            if hit.passage_id not in best or hit.score > best[hit.passage_id].score:
                best[hit.passage_id] = Passage(
                    hit.passage_id, hit.paper_id, hit.private, hit.title, hit.text, query, hit.score
                )
    return sorted(best.values(), key=lambda p: (-p.score, p.passage_id))[:k]


def risks_for(
    name: str, llm: LLM, library: Library, searcher: Searcher, version: int, attempts: int = ATTEMPTS
) -> Candidate:
    """The candidate `name` with the risks library passages state for it, each quote checked
    against its paper (spec H1); retried and cleaned like a hypothesis. With none: no risk, and
    the note "risks not established in the library"."""
    record: dict[str, object] = {"prompt_version": RISK_PROMPT_VERSION, "attempts": 0, "risks_removed": []}
    passages = {p.passage_id: p for p in risk_passages(name, searcher)}
    record["passages_offered"] = list(passages)
    if not passages:
        return Candidate(name, (), None, record)
    data = {
        "intervention": name,
        "passages": [
            {"passage_id": p.passage_id, "paper_title": p.title, "text": p.text} for p in passages.values()
        ],
    }
    messages: list[Message] = [
        {"role": "system", "content": RISK_SYSTEM},
        {"role": "user", "content": json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False)},
    ]
    best: tuple[tuple[Risk, ...], tuple[Problem, ...]] | None = None
    error_reason = None
    for attempt in range(1, attempts + 1):
        record["attempts"] = attempt
        try:
            completion = llm.complete(messages)
        except LLMError as error:
            error_reason = f"the LLM did not answer ({error})"
            break
        try:
            risks = parse_risks(completion.text, passages)
        except Malformed as error:
            feedback = f"That reply cannot be used: {error}."
        else:
            judged = [(r, _risk_problem(r, passages, library, version)) for r in risks]
            checked = (
                tuple(r for r, p in judged if p is None),
                tuple(p for _, p in judged if p is not None),
            )
            if best is None or len(checked[1]) <= len(best[1]):
                best = checked
            if not checked[1]:
                break
            feedback = _quotes_not_found(checked[1])
        messages = [
            *messages,
            {"role": "assistant", "content": completion.text},
            {"role": "user", "content": f"{feedback} Reply again with only the JSON object."},
        ]
    if best is None:
        return Candidate(name, (), error_reason or "no usable reply on its risks", record)
    record["risks_removed"] = [str(p) for p in best[1]]
    return Candidate(name, best[0], None, record)


def _risk_problem(risk: Risk, passages: dict[str, Passage], library: Library, version: int) -> Problem | None:
    """Why the risk cannot be shown: its citation fails (spec H1), or its quote is not in the
    passage offered, which names the intervention (elsewhere in the paper it may be another's risk)."""
    found = problem(risk, "a risk", library, version)
    if found is None and not quoted(risk.quote, passages[risk.passage_id].text):
        why = f"quotes words not found word for word in passage {risk.passage_id}"
        found = Problem("a risk", risk.passage_id, risk.quote, why)
    return found


def parse_risks(reply: str, passages: dict[str, Passage]) -> tuple[Risk, ...]:
    """The risks of a reply to the risk prompt; raises `Malformed` saying what is wrong."""
    answer = _json_object(reply)
    if set(answer) != {"risks"}:
        raise Malformed('the object must have exactly the key "risks"')
    risks = []
    for entry in _entries(answer, "risks", ("risk", "passage_id", "quote"), required=False):
        passage = passages.get(entry["passage_id"])
        if passage is None:
            raise Malformed(f'a risk cites passage "{entry["passage_id"]}", which was not given')
        risks.append(
            Risk(entry["risk"], entry["passage_id"], passage.paper_id, passage.title, entry["quote"])
        )
    return tuple(risks)


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
    if is_intervention(finding):
        return _intervention_statement(finding.marker_name, stats)
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


def _intervention_statement(name: str, stats: dict[str, int | float | str | None]) -> str:
    """An intervention's reported outcomes in one line, hidden values read "hidden"."""
    clear = stats.get("n_clear")
    authors = f"{clear} authors" if isinstance(clear, int) else "the authors"
    improved = _share(stats, "share_improved", "improved_low", "improved_high")
    worse = _share(stats, "share_worse", "worse_low", "worse_high")
    rank = stats.get("rank")
    ranked = f"; ranked {rank}" if isinstance(rank, int) else ""
    return f"{name}: improvement reported by {improved} of {authors} with a clear outcome, worse by {worse}{ranked}; {REPORTED}"


def _share(stats: dict[str, int | float | str | None], share: str, low: str, high: str) -> str:
    value, a, b = stats.get(share), stats.get(low), stats.get(high)
    if not isinstance(value, float):
        return "a hidden share"
    if isinstance(a, float) and isinstance(b, float):
        return f"{value:.0%} (95% CI {a:.0%} to {b:.0%})"
    return f"{value:.0%}"


def _number(value: object) -> str:
    return f"{value:.2g}" if isinstance(value, float) and abs(value) < 0.1 else f"{value:.2f}"


def _percent(value: object) -> str:
    return f"{value:.0%}" if isinstance(value, float) else "?"


def _interval(low: object, high: object) -> str:
    if isinstance(low, float) and isinstance(high, float):
        return f" (95% CI {_number(low)} to {_number(high)})"
    return ""
