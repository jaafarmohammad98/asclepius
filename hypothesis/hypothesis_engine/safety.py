"""Each cached drug's safety and generic availability, from openFDA (spec R2, R3).

`update_drugs` (`drugs.py`) asks openFDA about every drug it found, by its ChEMBL name:

1. its US label (`drug/label.json`): the latest one of a brand-name approval (an NDA), or when
   there is none the latest label of any. Its warnings, precautions and adverse reactions are read
   for the concerns of `CONCERNS`: sexual dysfunction, libido, anhedonia, emotional blunting. Each
   concern found is kept with the label text it was found in, the section and the label's set id
   (its DailyMed page).
2. its Drugs@FDA applications (`drug/drugsfda.json`): it has a generic when an abbreviated
   application (an ANDA) of it has a product still marketed.

A concern the label's clause denies ("sexual dysfunction was not observed") is not counted, nor is
more sexual desire ("increased libido", "hypoactive sexual desire disorder").

A label or application is of the drug when its active ingredients are the drug's: a salt or
hydrate counts as its parent ("PAROXETINE HYDROCHLORIDE" is paroxetine, `SALT_WORDS`), another
word makes it another drug ("ETHINYL ESTRADIOL" is not estradiol, "TESTOSTERONE CYPIONATE", an
ester ChEMBL keeps apart, not testosterone), and a combination product ("OLANZAPINE AND
FLUOXETINE") is not a label of one of its parts.

The risks listed for the drug in the `[risks]` table of `interventions.toml` are merged in
(`load_risks`), each with its cited source. The safety penalty (`Safety.penalty`, a cost criterion
of the ranking, M6-06) counts the label's concerns and those risks. It is unknown (None), not
zero, for a drug with no US label, and for one openFDA could not answer for (`Safety.unchecked`
says why): M6-06 decides what stands in for it.

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
from typing import Any, Literal, Protocol, get_args

OPENFDA = "https://api.fda.gov/"
LABEL_PAGE = 5  # labels asked for at a time, latest first
LABEL_PAGES = 4  # pages of labels looked through for one of the drug, at most, in each search
APPLICATION_PAGE = 100  # Drugs@FDA applications asked for at a time
INTERVENTIONS = Path(__file__).with_name("interventions.toml")

Section = Literal["boxed_warning", "warnings_and_cautions", "warnings", "precautions", "adverse_reactions"]
Concern = Literal["sexual dysfunction", "libido", "anhedonia", "emotional blunting"]
# The label sections read, in this order; each concern is kept from the first it is found in.
SECTIONS: tuple[Section, ...] = get_args(Section)

# The fields read of a label, of its `openfda` part, of an application and of its products: all
# `record_responses.py` keeps of a recorded answer.
LABEL_FIELDS = ("set_id", "effective_time", *SECTIONS)
LABEL_OPENFDA_FIELDS = ("generic_name", "application_number")
APPLICATION_FIELDS = ("application_number",)
PRODUCT_FIELDS = ("marketing_status", "active_ingredients")
INGREDIENT_FIELDS = ("name",)

# The concerns a label is penalised for (spec R3), each a pattern matched with case aside.
CONCERNS: dict[Concern, str] = {
    "sexual dysfunction": (
        r"sexual (?:dysfunction|side effects?|adverse (?:events?|reactions?|effects?))"
        r"|erectile dysfunction|impotence|anorgasmia|orgasm\w* (?:dysfunction|disorders?|abnormal\w*)"
        r"|abnormal orgasm|(?:abnormal|delayed|decreased|retrograde) ejaculation"
        r"|ejaculat\w* (?:disorders?|dysfunction|failure|delay\w*)|genital (?:anesthesia|numbness)"
        r"|(?:decreased|impaired|reduced) sexual function|sexual function abnormal"
    ),
    "libido": r"libido|sex drive|sexual desire|(?:loss of|decreased|reduced) sexual interest",
    "anhedonia": r"anhedonia",
    "emotional blunting": (
        r"emotional (?:blunting|flattening|numbing|indifference)|(?:blunted|flat|flattened) affect"
        r"|affective (?:blunting|flattening)"
    ),
}
_CONCERNS = {
    concern: re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE) for concern, pattern in CONCERNS.items()
}
# More sexual desire is not a concern: a match within one of these is not counted.
MORE_DESIRE = re.compile(
    r"(?:increased|increase in|heightened) (?:libido|sex drive|sexual desire)"
    r"|(?:libido|sex drive|sexual desire) increased|hypoactive sexual desire",
    re.IGNORECASE,
)
# A clause denying the concern: before it ("no reports of", "without") or after it ("was not observed").
DENIED_BEFORE = re.compile(r"\b(?:no|not|without|neither|nor|absence of)\b", re.IGNORECASE)
DENIED_AFTER = re.compile(
    r"^[^.;:]{0,40}?\b(?:was|were|has|have|is|are)\s+not\s+(?:been\s+)?(?:observed|reported|seen|found)",
    re.IGNORECASE,
)
CLAUSE = 60  # characters before a match looked through for a denial, at most, within its clause
# The words a salt, hydrate or dosage form adds to an ingredient's name without making it another drug.
SALT_WORDS = frozenset(
    [
        "hydrochloride",
        "hcl",
        "dihydrochloride",
        "hydrobromide",
        "hbr",
        "hydroiodide",
        "sodium",
        "disodium",
        "potassium",
        "dipotassium",
        "calcium",
        "magnesium",
        "zinc",
        "lithium",
        "aluminum",
        "ammonium",
        "maleate",
        "mesylate",
        "dimesylate",
        "methanesulfonate",
        "besylate",
        "besilate",
        "tosylate",
        "citrate",
        "tartrate",
        "bitartrate",
        "succinate",
        "hemisuccinate",
        "fumarate",
        "hemifumarate",
        "sulfate",
        "bisulfate",
        "hemisulfate",
        "phosphate",
        "diphosphate",
        "acetate",
        "lactate",
        "gluconate",
        "malate",
        "oxalate",
        "napsylate",
        "pamoate",
        "mandelate",
        "xinafoate",
        "tromethamine",
        "meglumine",
        "bromide",
        "chloride",
        "iodide",
        "nitrate",
        "hydrate",
        "monohydrate",
        "dihydrate",
        "trihydrate",
        "hemihydrate",
        "sesquihydrate",
        "anhydrous",
        "oral",
        "orally",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "solution",
        "suspension",
        "injection",
        "injectable",
        "intravenous",
        "liquid",
        "concentrate",
        "extended",
        "delayed",
        "release",
        "er",
        "xr",
        "sr",
        "disintegrating",
        "chewable",
        "topical",
        "gel",
        "cream",
        "ointment",
        "transdermal",
        "patch",
        "film",
        "spray",
        "nasal",
        "ophthalmic",
        "for",
    ]
)
EXCERPT = 150  # characters kept on each side of a match, at most, within its sentence
MARKETED = ("Prescription", "Over-the-counter")  # Drugs@FDA's statuses of a product still sold
SOURCE = re.compile(r"PMID:\d+|DOI:10\.\S+|https://\S+")  # a cited source of a risk


class SafetyError(Exception):
    """openFDA answered with something unusable, or the risks table is malformed."""


class JsonService(Protocol):
    """A service's requests (`drugs.Service`): JSON answers to a path and its parameters."""

    def get(self, path: str, params: dict[str, str]) -> Any: ...


@dataclass(frozen=True)
class Mention:
    """A concern a label states, with the text it states it in."""

    concern: Concern
    section: Section
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
    label: Label | None  # None when openFDA holds no US label of it, or was not asked
    risks: tuple[Risk, ...]
    generic_applications: tuple[str, ...] | None  # its ANDAs still marketed: "ANDA090121"; None unchecked
    unchecked: str | None = None  # why openFDA could not answer for it, when it could not

    @property
    def generic(self) -> bool | None:
        """A generic (ANDA) product of it is still marketed in the US; None when unchecked."""
        return None if self.generic_applications is None else bool(self.generic_applications)

    @property
    def concerns(self) -> tuple[Concern, ...]:
        return tuple(m.concern for m in self.label.mentions) if self.label else ()

    @property
    def penalty(self) -> int | None:
        """The label's concerns and the listed risks: the more, the worse (spec R3). None, unknown,
        with no label to read."""
        return None if self.label is None else len(self.concerns) + len(self.risks)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Safety:
        """The safety `dataclasses.asdict` wrote of one."""
        label, applications = data["label"], data["generic_applications"]
        for m in label["mentions"] if label else ():
            if m["concern"] not in CONCERNS or m["section"] not in SECTIONS:
                raise SafetyError(f"not a concern of a label section: {m['concern']!r}, {m['section']!r}")
        return cls(
            None
            if label is None
            else Label(
                label["set_id"],
                label["effective_time"],
                label["application"],
                tuple(Mention(m["concern"], m["section"], m["excerpt"]) for m in label["mentions"]),
            ),
            tuple(Risk(r["risk"], r["source"]) for r in data["risks"]),
            None if applications is None else tuple(applications),
            data["unchecked"],
        )


def _words(name: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", name.casefold()))


def _ingredients(name: str) -> list[frozenset[str]]:
    """The active ingredients a name lists: "OLANZAPINE AND FLUOXETINE HYDROCHLORIDE" is two."""
    parts = re.split(r"\s+and\s+|,|/|;", name.casefold())
    return [w for w in (_words(p) for p in parts) if w]


def same_ingredients(drug: str, ingredients: Iterable[str]) -> bool:
    """Whether `ingredients` (names, maybe each a combination) are the drug's own, a salt or
    hydrate counting as its parent: each may add only `SALT_WORDS` to one of the drug's."""
    theirs = [i for name in ingredients for i in _ingredients(name)]
    ours = _ingredients(drug)
    if not ours or len(theirs) != len(ours):
        return False
    left = list(theirs)
    for words in ours:
        match = next((i for i in left if words <= i and i - words <= SALT_WORDS), None)
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


def _stated(text: str, match: re.Match[str]) -> bool:
    """Whether the label states the concern it matched: not a denial of it, nor more desire."""
    nearby = MORE_DESIRE.finditer(text, max(0, match.start() - 25), match.end() + 25)
    if any(m.start() <= match.start() and match.end() <= m.end() for m in nearby):
        return False
    before = text[max(0, match.start() - CLAUSE) : match.start()]
    clause = re.split(r"[.;:]", before)[-1]
    return not DENIED_BEFORE.search(clause) and not DENIED_AFTER.search(text[match.end() :])


def find_mentions(label: Mapping[str, Any]) -> tuple[Mention, ...]:
    """Each concern the label states, from the first section stating it."""
    found: list[Mention] = []
    for concern, pattern in _CONCERNS.items():
        for section in SECTIONS:
            given = label.get(section) or ()
            texts = [t for t in (given if isinstance(given, list) else [given]) if isinstance(t, str)]
            stated = next(((t, m) for t in texts for m in pattern.finditer(t) if _stated(t, m)), None)
            if stated:
                text, m = stated
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
        return [r for r in results if isinstance(r, dict)], total
    except (KeyError, TypeError, ValueError) as error:  # TypeError too when `results` is no list
        raise SafetyError(f"openFDA {what}: an answer without its results") from error


def _first(value: object) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, str) else None


class OpenFDA:
    """Looks a drug's label and generics up in openFDA."""

    def __init__(self, service: JsonService, risks: Mapping[str, Sequence[Risk]]) -> None:
        self._service = service
        self._risks = {name.casefold(): tuple(r) for name, r in risks.items()}

    def label(self, name: str) -> Label | None:
        """The latest brand-name label of the drug, or else the latest label of any: up to
        `LABEL_PAGES` pages of each search looked through for one of the drug's own."""
        phrase = _phrase(name)
        for search in (f"openfda.generic_name:{phrase} AND openfda.application_number:NDA*",
                       f"openfda.generic_name:{phrase}"):  # fmt: skip
            for page in range(LABEL_PAGES):
                params = {"search": search, "sort": "effective_time:desc", "limit": str(LABEL_PAGE),
                          "skip": str(page * LABEL_PAGE)}  # fmt: skip
                labels, total = _results(self._service.get("drug/label.json", params), "label")
                for label in labels:
                    given = label.get("openfda")
                    given = given if isinstance(given, dict) else {}
                    names = [n for n in given.get("generic_name") or () if isinstance(n, str)]
                    if any(same_ingredients(name, [n]) for n in names):
                        return Label(
                            str(label.get("set_id")),
                            _first(label.get("effective_time")),
                            _first(given.get("application_number")),
                            find_mentions(label),
                        )
                if (page + 1) * LABEL_PAGE >= total:
                    break
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
                    "limit": str(APPLICATION_PAGE),
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
            skip += APPLICATION_PAGE
            if not applications or skip >= total:
                return tuple(sorted(set(found)))

    def risks(self, name: str) -> tuple[Risk, ...]:
        """The risks `interventions.toml` lists for the drug."""
        return self._risks.get(name.casefold(), ())

    def safety(self, name: str) -> Safety:
        generics = self.generic_applications(name)
        return Safety(self.label(name), self.risks(name), generics)

    def unchecked(self, name: str, reason: str) -> Safety:
        """The drug's safety when openFDA could not answer for it: its listed risks, nothing more."""
        return Safety(None, self.risks(name), None, reason)


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
