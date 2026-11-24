"""Each cached drug's safety and generic availability, from openFDA (spec R2, R3).

`update_drugs` (`drugs.py`) asks openFDA about every drug it found, by its ChEMBL name:

1. its US label (`drug/label.json`): the latest one of a brand-name approval (an NDA), or when
   there is none the latest label of any. Its warnings, precautions and adverse reactions are read
   for the concerns of `CONCERNS`: sexual dysfunction, libido, anhedonia, emotional blunting. Each
   concern found is kept with the label text it was found in, the section and the label's set id
   (its DailyMed page).
2. its Drugs@FDA applications (`drug/drugsfda.json`): it has a generic when an abbreviated
   application (an ANDA) of it has a product still marketed.

A label or application is of the drug when its active ingredients are the drug's: a salt counts
as its parent ("PAROXETINE HYDROCHLORIDE" is paroxetine), and a combination product
("OLANZAPINE AND FLUOXETINE") is not a label of one of its parts.

The risks listed for the drug in the `[risks]` table of `interventions.toml` are merged in
(`load_risks`), each with its cited source. The safety penalty (`Safety.penalty`, a cost criterion
of the ranking, M6-06) counts the label's concerns and those risks.

The only free text sent is the drug's name. openFDA is free and needs no key at this rate
(at most 3 requests a second, well under its 240 a minute); it answers HTTP 404 when nothing
matches, which `drugs.http_transport(..., not_found_ok=True)` hands back as an answer.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

OPENFDA = "https://api.fda.gov/"
LABELS = 5  # labels asked for at a time, latest first
APPLICATIONS = 100  # Drugs@FDA applications a page
INTERVENTIONS = Path(__file__).with_name("interventions.toml")

# The label sections read, in this order; each concern is kept from the first it is found in.
SECTIONS = ("boxed_warning", "warnings_and_cautions", "warnings", "precautions", "adverse_reactions")

# The concerns a label is penalised for (spec R3), each a pattern matched with case aside.
CONCERNS: dict[str, str] = {
    "sexual dysfunction": (
        r"sexual (?:dysfunction|side effects?|adverse (?:events?|reactions?))|erectile dysfunction"
        r"|impotence|anorgasmia|orgasm\w* (?:dysfunction|disorders?|abnormalit\w+)|abnormal orgasm"
        r"|(?:abnormal|delayed|decreased|retrograde) ejaculation"
        r"|ejaculat\w* (?:disorders?|dysfunction|failure|delay\w*)|genital (?:anesthesia|numbness)"
    ),
    "libido": r"libido|sex drive|sexual desire",
    "anhedonia": r"anhedonia",
    "emotional blunting": (
        r"emotional (?:blunting|flattening|numbing|indifference)|(?:blunted|flat|flattened) affect"
        r"|affective (?:blunting|flattening)"
    ),
}
_CONCERNS = {
    concern: re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE) for concern, pattern in CONCERNS.items()
}
EXCERPT = 150  # characters kept on each side of a match, at most, within its sentence
MARKETED = ("Prescription", "Over-the-counter")  # Drugs@FDA's statuses of a product still sold
SOURCE = re.compile(r"PMID:\d+|DOI:10\.\S+|https://\S+")  # a cited source of a risk


class SafetyError(Exception):
    """openFDA answered with something unusable, or the risks table is malformed."""


class Getter(Protocol):
    """A service's requests (`drugs.Service`): JSON answers to a path and its parameters."""

    def get(self, path: str, params: dict[str, str]) -> Any: ...


@dataclass(frozen=True)
class Mention:
    """A concern a label states, with the text it states it in."""

    concern: str  # one of `CONCERNS`
    section: str  # one of `SECTIONS`
    excerpt: str  # the label's words around it


@dataclass(frozen=True)
class Label:
    set_id: str
    effective_time: str | None  # "20260921"
    application: str | None  # "NDA020788"; None for a drug sold without an application
    mentions: tuple[Mention, ...]  # in `CONCERNS` order

    @property
    def url(self) -> str:
        return f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={self.set_id}"


@dataclass(frozen=True)
class Risk:
    """A risk of the drug listed by hand in `interventions.toml`, with the source it is cited from."""

    risk: str
    source: str  # "PMID:123", "DOI:10...", or an https URL


@dataclass(frozen=True)
class Safety:
    label: Label | None  # None when openFDA holds no US label of it
    risks: tuple[Risk, ...]
    generic: bool  # a generic (ANDA) product of it is still marketed in the US
    generic_applications: tuple[str, ...]  # those ANDAs: "ANDA090121"

    @property
    def concerns(self) -> tuple[str, ...]:
        return tuple(m.concern for m in self.label.mentions) if self.label else ()

    @property
    def penalty(self) -> int:
        """The label's concerns and the listed risks: the more, the worse (spec R3)."""
        return len(self.concerns) + len(self.risks)


def _words(name: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", name.casefold()))


def _ingredients(name: str) -> list[frozenset[str]]:
    """The active ingredients a name lists: "OLANZAPINE AND FLUOXETINE HYDROCHLORIDE" is two."""
    parts = re.split(r"\s+and\s+|,|/|;", name.casefold())
    return [w for w in (_words(p) for p in parts) if w]


def same_ingredients(drug: str, ingredients: Iterable[str]) -> bool:
    """Whether `ingredients` (names, maybe each a combination) are the drug's own, a salt or
    hydrate counting as its parent."""
    theirs = [i for name in ingredients for i in _ingredients(name)]
    ours = _ingredients(drug)
    if not ours or len(theirs) != len(ours):
        return False
    left = list(theirs)
    for words in ours:
        match = next((i for i in left if words <= i), None)
        if match is None:
            return False
        left.remove(match)
    return True


def _excerpt(text: str, start: int, end: int) -> str:
    """The words around `text[start:end]`, within its sentence, at most `EXCERPT` either side."""
    before = text.rfind(". ", 0, start)
    sentence_start = before + 2 if before >= 0 else 0
    sentence_end = text.find(". ", end)
    sentence_end = len(text) if sentence_end < 0 else sentence_end + 1
    left, right = max(sentence_start, start - EXCERPT), min(sentence_end, end + EXCERPT)
    if left > sentence_start:
        left = text.find(" ", left, start) + 1 or left
    if right < sentence_end and (space := text.rfind(" ", end, right)) > 0:
        right = space
    excerpt = " ".join(text[left:right].split())
    return f"{'…' if left > sentence_start else ''}{excerpt}{'…' if right < sentence_end else ''}"


def mentions(label: Mapping[str, Any]) -> tuple[Mention, ...]:
    """Each concern the label states, from the first section stating it."""
    found: list[Mention] = []
    for concern, pattern in _CONCERNS.items():
        for section in SECTIONS:
            texts = label.get(section) or ()
            match = next(
                ((t, m) for t in (texts if isinstance(texts, list) else [texts]) if isinstance(t, str)
                 for m in [pattern.search(t)] if m),
                None,
            )  # fmt: skip
            if match:
                text, m = match
                found.append(Mention(concern, section, _excerpt(text, m.start(), m.end())))
                break
    return tuple(found)


def _phrase(name: str) -> str:
    """The drug's name as an openFDA phrase: only letters, digits, spaces and hyphens."""
    return '"' + " ".join(re.sub(r"[^A-Za-z0-9-]", " ", name).split()) + '"'


def _results(answer: object, what: str) -> tuple[list[dict[str, Any]], int]:
    """An openFDA answer's results and their total: none when it found nothing."""
    if isinstance(answer, dict) and isinstance(answer.get("error"), dict):
        if answer["error"].get("code") == "NOT_FOUND":
            return [], 0
        raise SafetyError(f"openFDA {what}: {answer['error'].get('message')}")
    try:
        results, total = answer["results"], int(answer["meta"]["results"]["total"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as error:
        raise SafetyError(f"openFDA {what}: an answer without its results") from error
    if not isinstance(results, list):
        raise SafetyError(f"openFDA {what}: an answer without its results")
    return [r for r in results if isinstance(r, dict)], total


def _first(value: object) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, str) else None


class OpenFDA:
    """Looks a drug's label and generics up in openFDA."""

    def __init__(self, service: Getter, risks: Mapping[str, Sequence[Risk]] | None = None) -> None:
        self._service = service
        self._risks = {name.casefold(): tuple(r) for name, r in (risks or {}).items()}

    def label(self, name: str) -> Label | None:
        """The latest brand-name label of the drug, or else the latest label of any."""
        phrase = _phrase(name)
        for search in (f"openfda.generic_name:{phrase} AND openfda.application_number:NDA*",
                       f"openfda.generic_name:{phrase}"):  # fmt: skip
            answer = self._service.get(
                "drug/label.json", {"search": search, "sort": "effective_time:desc", "limit": str(LABELS)}
            )
            labels, _total = _results(answer, "label")
            for label in labels:
                given = label.get("openfda")
                generic_names = given.get("generic_name") if isinstance(given, dict) else None
                if (
                    isinstance(generic_names, list)
                    and generic_names
                    and same_ingredients(name, generic_names[:1])
                ):
                    return Label(
                        str(label.get("set_id")),
                        _first(label.get("effective_time")),
                        _first(given.get("application_number")) if isinstance(given, dict) else None,
                        mentions(label),
                    )
        return None

    def generic_applications(self, name: str) -> tuple[str, ...]:
        """The drug's ANDAs with a product of it still marketed."""
        found: list[str] = []
        skip = 0
        while True:
            answer = self._service.get(
                "drug/drugsfda.json",
                {
                    "search": f"products.active_ingredients.name:{_phrase(name)} AND application_number:ANDA*",
                    "limit": str(APPLICATIONS),
                    "skip": str(skip),
                },
            )
            applications, total = _results(answer, "drugsfda")
            for application in applications:
                number = application.get("application_number")
                if not isinstance(number, str) or not number.startswith("ANDA"):
                    continue
                if any(
                    isinstance(p, dict)
                    and p.get("marketing_status") in MARKETED
                    and same_ingredients(
                        name,
                        [a.get("name", "") for a in p.get("active_ingredients") or () if isinstance(a, dict)],
                    )
                    for p in application.get("products") or ()
                ):
                    found.append(number)
            skip += APPLICATIONS
            if not applications or skip >= total:
                return tuple(sorted(set(found)))

    def safety(self, name: str) -> Safety:
        generics = self.generic_applications(name)
        return Safety(self.label(name), self._risks.get(name.casefold(), ()), bool(generics), generics)


def load_risks(text: str) -> dict[str, tuple[Risk, ...]]:
    """The `[risks]` table of `interventions.toml`: each drug's risks, every one with a cited
    source (a PMID, a DOI or an https URL). None listed is an empty table."""
    try:
        table = tomllib.loads(text).get("risks", {})
    except tomllib.TOMLDecodeError as error:
        raise SafetyError(f"interventions.toml: {error}") from error
    if not isinstance(table, dict):
        raise SafetyError("interventions.toml: [risks] is not a table")
    risks: dict[str, tuple[Risk, ...]] = {}
    for name, listed in table.items():
        if not isinstance(listed, list):
            raise SafetyError(f"interventions.toml: the risks of {name} are not a list")
        kept = []
        for entry in listed:
            if not isinstance(entry, dict) or set(entry) != {"risk", "source"}:
                raise SafetyError(f"interventions.toml: a risk of {name} is not {{risk, source}}")
            risk, source = entry["risk"], entry["source"]
            if not isinstance(risk, str) or not risk.strip():
                raise SafetyError(f"interventions.toml: a risk of {name} has no text")
            if not isinstance(source, str) or not SOURCE.fullmatch(source):
                raise SafetyError(
                    f"interventions.toml: the risk {risk!r} of {name} cites no PMID, DOI or https URL"
                )
            kept.append(Risk(risk.strip(), source))
        risks[name.casefold()] = tuple(kept)
    return risks


def read_risks(path: Path = INTERVENTIONS) -> dict[str, tuple[Risk, ...]]:
    return load_risks(path.read_text(encoding="utf-8"))
