"""Approved drugs acting on the knowledge graph's top bottleneck targets (spec R1, R2, R6).

`asclepius-library build` ends by fetching them, and `asclepius-library drugs` fetches them again
on their own. The web app only reads the cache this writes (`read_drug_cache`) and never fetches.

The targets are the genes with the highest bottleneck scores (`bottlenecks.bottlenecks` on the
library's graph, `TOP_TARGETS` of them). For each, ChEMBL is asked

1. which human targets have the gene among their components, by its HGNC symbol (exactly, as a
   gene symbol of a component): the protein
   itself, and the families and complexes it is part of (finasteride's 5-alpha reductase is a
   family of SRD5A1 to SRD5A3);
2. which drug mechanisms act on those targets, and so which parent molecules (a salt counts as
   its parent);
3. what those molecules are: name, `max_phase`, molecular properties, whether withdrawn. Only
   approved ones (`max_phase` 4) are kept; one withdrawn since (alpidem, sibutramine) is kept
   and flagged, a risk for the ranking to weigh.

When ChEMBL cannot be used for a target (it is down, or answers with something unreadable),
MyChem.info is asked instead: the molecules whose DrugCentral bioactivities name the gene as a
mechanism of action, with the ChEMBL id, phase and properties MyChem keeps of each.

The only free text sent is the target's gene symbol, and to openFDA the drug's name; everything
else is a ChEMBL id. Every request
goes through the library's `PrivateTextGuard` first all the same (M4-08), so a private paper's
text could not leave in one.

The properties are those ChEMBL gives for a CNS MPO estimate (M6-06): ChEMBL 36 no longer gives
the ChemAxon values (`cx_logp`, `cx_logd`, `cx_most_bpka`), which are kept as None when absent;
`alogp` stands in for the logP then.

The drugs said to cause the syndromes (`KNOWN_CAUSES`: SSRIs, SNRIs, finasteride, dutasteride,
isotretinoin) are kept, tagged with what they are, so they can be listed as "known causes,
excluded" and never ranked.

Then each drug's label concerns, listed risks and generic availability are asked of openFDA
(`safety.py`), by the drug's name.

Nothing is written until every target and drug has been looked up, so a failed fetch leaves the previous
cache as it was. All three services are free, need no account and are sent no email; requests are
spaced at most 3 a second and a transient failure (HTTP 429, 5xx, a dropped connection) is sent
again (`pubmed.send_politely`).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from hypothesis_engine.bottlenecks import Bottleneck, bottlenecks
from hypothesis_engine.graph import library_graph
from hypothesis_engine.home import DATA
from hypothesis_engine.library import Library
from hypothesis_engine.privacy import PrivateTextGuard
from hypothesis_engine.pubmed import RateLimiter, TransientError, Transport, send_politely
from hypothesis_engine.safety import Label, Mention, OpenFDA, Risk, Safety
from hypothesis_engine.vocabulary import Vocabulary

DEFAULT_DRUGS = DATA / "drugs.json"
CACHE_VERSION = 2  # 2: each drug's safety, from openFDA (M6-05b)
TOP_TARGETS = 20  # genes, highest bottleneck score first
APPROVED = 4  # ChEMBL's max_phase of an approved drug
RATE = 3  # requests a second, to each service
PAGE = 1000  # ChEMBL's largest page
MOLECULE_BATCH = 50  # ids per molecule request, so the URL stays short
TIMEOUT = 120.0
USER_AGENT = "asclepius (open-source research tool)"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"
MYCHEM = "https://mychem.info/v1/"
HUMAN = "Homo sapiens"
PROPERTIES = ("alogp", "cx_logp", "cx_logd", "cx_most_bpka", "mw_freebase", "psa", "hbd", "hba")

# The drugs that cause PSSD, PFS and PAS (spec R6), by any word of ChEMBL's name of the parent
# molecule (so "OLANZAPINE AND FLUOXETINE" counts), and any drug of ATC class N06AB (`SSRI_ATC`).
# Dapoxetine is an SSRI, though sold for premature ejaculation; indalpine and zimelidine are SSRIs
# withdrawn long ago, sibutramine a withdrawn serotonin-noradrenaline reuptake inhibitor sold for
# weight loss. Vilazodone and vortioxetine act on the serotonin transporter among other targets and
# are not counted as SSRIs.
KNOWN_CAUSES = {
    **dict.fromkeys(
        (
            "citalopram",
            "escitalopram",
            "fluoxetine",
            "fluvoxamine",
            "paroxetine",
            "sertraline",
            "dapoxetine",
            "indalpine",
            "zimelidine",
            "zimeldine",  # ChEMBL's spelling
        ),  # fmt: skip
        "SSRI",
    ),
    **dict.fromkeys(
        ("venlafaxine", "desvenlafaxine", "duloxetine", "milnacipran", "levomilnacipran", "sibutramine"),
        "SNRI",
    ),
    "finasteride": "5-alpha reductase inhibitor",
    "dutasteride": "5-alpha reductase inhibitor",
    "isotretinoin": "retinoid",
}
SSRI_ATC = "N06AB"  # selective serotonin reuptake inhibitors


Source = Literal["chembl", "mychem"]  # which service answered


class DrugsError(Exception):
    """A service answered with something unusable, or the drug cache cannot be read."""


@dataclass(frozen=True)
class Target:
    """A gene the drugs are looked up for, with the bottleneck score that chose it."""

    symbol: str
    score: float
    syndromes: tuple[str, ...]  # the syndromes linking to it: "PSSD", "PFS", "PAS"
    chembl_ids: tuple[str, ...]  # the ChEMBL targets it is a component of; none when MyChem answered
    source: Source


@dataclass(frozen=True)
class Action:
    """How a drug acts on one target, as ChEMBL's mechanism (or DrugCentral's) records it."""

    target: str  # the gene symbol
    target_chembl_id: str | None
    action_type: str | None  # "INHIBITOR", "AGONIST"...
    mechanism: str | None  # "Steroid 5-alpha-reductase 2 inhibitor"; DrugCentral's target name from MyChem


@dataclass(frozen=True)
class Drug:
    chembl_id: str
    name: str
    max_phase: float
    properties: dict[str, float | None]  # `PROPERTIES`, None where ChEMBL gives none
    actions: tuple[Action, ...]
    known_cause: str | None  # what it is when it causes a syndrome ("SSRI"): never ranked
    source: Source
    withdrawn: bool | None = None  # withdrawn from some market after approval; None when unknown (MyChem)
    safety: Safety | None = None  # from openFDA; None only before `add_safety`

    @property
    def url(self) -> str:
        return f"https://www.ebi.ac.uk/chembl/explore/compound/{self.chembl_id}"


@dataclass(frozen=True)
class DrugCache:
    fetched_at: str
    library_version: int | None
    targets: tuple[Target, ...]
    drugs: tuple[Drug, ...]  # by name

    @property
    def ranked(self) -> list[Drug]:
        return [d for d in self.drugs if d.known_cause is None]

    @property
    def excluded(self) -> list[Drug]:
        return [d for d in self.drugs if d.known_cause is not None]


def known_cause(name: str, atc: Iterable[str] = ()) -> str | None:
    """What a drug is when it causes a syndrome: by any word of its name, with case aside, or by
    its ATC codes."""
    for word in re.findall(r"[a-z]+", name.casefold()):
        if word in KNOWN_CAUSES:
            return KNOWN_CAUSES[word]
    return "SSRI" if any(code.startswith(SSRI_ATC) for code in atc) else None


def top_targets(scores: Iterable[Bottleneck], n: int = TOP_TARGETS) -> list[Bottleneck]:
    """The `n` genes with the highest bottleneck scores, in score order."""
    return [b for b in scores if b.kind == "gene"][:n]


def http_transport(base: str, not_found_ok: bool = False) -> Transport:
    """The real network: a GET to `base` + path, the parameters in the query string. With
    `not_found_ok`, an HTTP 404's body is the answer (openFDA's "no matches")."""

    def send(path: str, params: dict[str, str]) -> bytes:
        url = f"{base}{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body: bytes = response.read()
                return body
        except urllib.error.HTTPError as error:
            if error.code == 404 and not_found_ok:
                found_none: bytes = error.read()
                return found_none
            if error.code == 429 or error.code >= 500:
                raise TransientError(f"{base}{path}: HTTP {error.code}") from error
            raise DrugsError(f"{base}{path}: HTTP {error.code} {error.reason}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise TransientError(f"{base}{path}: {error}") from error

    return send


class Service:
    """One service's requests: each checked by `guard`, then sent within `limiter`'s rate."""

    def __init__(
        self,
        transport: Transport,
        limiter: RateLimiter,
        guard: PrivateTextGuard | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport
        self._limiter = limiter
        self._guard = guard
        self._sleep = sleep

    def get(self, path: str, params: dict[str, str]) -> Any:
        if self._guard is not None:
            self._guard.check(" ".join((path, *params.values())))
        body = send_politely(self._transport, path, params, self._limiter, self._sleep)
        try:
            return json.loads(body)
        except ValueError as error:
            raise DrugsError(f"{path}: an answer that is not JSON") from error


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _properties(raw: object) -> dict[str, float | None]:
    given = raw if isinstance(raw, dict) else {}
    return {name: _number(given.get(name)) for name in PROPERTIES}


class ChEMBL:
    """Looks a gene's approved drugs up in ChEMBL."""

    def __init__(self, service: Service) -> None:
        self._service = service

    def _all(self, path: str, params: dict[str, str], key: str) -> list[dict[str, Any]]:
        """Every page of a list answer."""
        found: list[dict[str, Any]] = []
        offset = 0
        while True:
            answer = self._service.get(path, {**params, "limit": str(PAGE), "offset": str(offset)})
            try:
                page, total = answer[key], int(answer["page_meta"]["total_count"])
            except (KeyError, TypeError, ValueError) as error:
                raise DrugsError(f"ChEMBL {path}: an answer without its {key}") from error
            found.extend(page)
            offset += PAGE
            if not page or offset >= total:
                return found

    def targets(self, symbol: str) -> list[str]:
        """The human ChEMBL targets with the gene among their components."""
        # the exact-synonym filter, not the search: the search fails on short symbols ("AR")
        found = self._all(
            "target.json",
            {
                "target_components__target_component_synonyms__component_synonym__iexact": symbol,
                "organism": HUMAN,
            },
            "targets",
        )
        return sorted(
            {
                t["target_chembl_id"]
                for t in found
                if t.get("organism") == HUMAN
                and any(
                    s.get("syn_type") == "GENE_SYMBOL" and s.get("component_synonym") == symbol
                    for c in t.get("target_components") or ()
                    for s in c.get("target_component_synonyms") or ()
                )
            }
        )

    def drugs(self, symbol: str, target_ids: Sequence[str]) -> list[Drug]:
        """The approved drugs whose mechanisms act on these targets of the gene."""
        if not target_ids:
            return []
        mechanisms = self._all("mechanism.json", {"target_chembl_id__in": ",".join(target_ids)}, "mechanisms")
        actions: dict[str, list[Action]] = {}
        for m in mechanisms:
            parent = m.get("parent_molecule_chembl_id") or m.get("molecule_chembl_id")
            if parent:
                action = Action(
                    symbol, m.get("target_chembl_id"), m.get("action_type"), m.get("mechanism_of_action")
                )
                if action not in actions.setdefault(parent, []):
                    actions[parent].append(action)
        ids = sorted(actions)
        molecules: list[dict[str, Any]] = []
        for start in range(0, len(ids), MOLECULE_BATCH):
            batch = ",".join(ids[start : start + MOLECULE_BATCH])
            molecules.extend(self._all("molecule.json", {"molecule_chembl_id__in": batch}, "molecules"))
        drugs = []
        for molecule in molecules:
            chembl_id, phase = molecule.get("molecule_chembl_id"), _number(molecule.get("max_phase"))
            if chembl_id not in actions or phase is None or phase < APPROVED:
                continue
            name = molecule.get("pref_name") or chembl_id
            drugs.append(
                Drug(
                    chembl_id,
                    name,
                    phase,
                    _properties(molecule.get("molecule_properties")),
                    tuple(actions[chembl_id]),
                    known_cause(name, molecule.get("atc_classifications") or ()),
                    "chembl",
                    bool(molecule.get("withdrawn_flag")),
                )
            )
        return drugs


class MyChem:
    """Looks a gene's approved drugs up in MyChem.info, from DrugCentral's mechanisms of action."""

    FIELDS = "chembl.molecule_chembl_id,chembl.pref_name,chembl.max_phase,chembl.molecule_properties,drugcentral.bioactivity"
    SIZE = 1000  # MyChem's largest page

    def __init__(self, service: Service) -> None:
        self._service = service

    def drugs(self, symbol: str) -> list[Drug]:
        answer = self._service.get(
            "query",
            {
                "q": f"drugcentral.bioactivity.uniprot.gene_symbol:{symbol}",
                "fields": self.FIELDS,
                "size": str(self.SIZE),
            },
        )
        try:
            hits = answer["hits"]
        except (KeyError, TypeError) as error:
            raise DrugsError("MyChem.info query: an answer without its hits") from error
        drugs: dict[str, Drug] = {}
        for hit in hits:
            chembl = hit.get("chembl")
            bioactivity = [  # MyChem gives one record or a list of them, at each level
                b
                for record in _listed(hit.get("drugcentral"))
                if isinstance(record, dict)
                for b in _listed(record.get("bioactivity"))
            ]
            for record in _listed(chembl):
                if not isinstance(record, dict):
                    continue
                chembl_id, phase = record.get("molecule_chembl_id"), _number(record.get("max_phase"))
                if not chembl_id or phase is None or phase < APPROVED:
                    continue
                actions = tuple(
                    dict.fromkeys(  # each distinct action once, in order
                        Action(symbol, None, b.get("action_type"), b.get("target_name"))
                        for b in bioactivity
                        if isinstance(b, dict)
                        and str(b.get("moa")) == "1"
                        and b.get("organism") == HUMAN
                        and any(
                            isinstance(u, dict) and u.get("gene_symbol") == symbol
                            for u in _listed(b.get("uniprot"))
                        )
                    )
                )
                if actions and chembl_id not in drugs:
                    name = record.get("pref_name") or chembl_id
                    drugs[chembl_id] = Drug(
                        chembl_id,
                        name,
                        phase,
                        _properties(record.get("molecule_properties")),
                        actions,
                        known_cause(name),
                        "mychem",
                    )
        return list(drugs.values())


def _listed(value: object) -> list[Any]:
    """MyChem's one-or-many: a list as it is, anything else as a list of it (None as none)."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def fetch_drugs(
    targets: Sequence[Bottleneck],
    chembl: ChEMBL,
    mychem: MyChem,
    progress: Callable[[str], None] = lambda _message: None,
) -> tuple[list[Target], list[Drug]]:
    """The approved drugs acting on each target, merged by drug; MyChem.info for a target ChEMBL fails on."""
    looked_up: list[Target] = []
    merged: dict[str, Drug] = {}
    for bottleneck in targets:
        symbol = bottleneck.name
        try:
            ids = chembl.targets(symbol)
            found = chembl.drugs(symbol, ids)
            looked_up.append(Target(symbol, bottleneck.score, bottleneck.syndromes, tuple(ids), "chembl"))
        except (DrugsError, TransientError) as error:
            progress(f"{symbol}: ChEMBL failed ({error}), asking MyChem.info")
            found = mychem.drugs(symbol)
            looked_up.append(Target(symbol, bottleneck.score, bottleneck.syndromes, (), "mychem"))
        progress(f"{symbol}: {len(found)} approved drugs")
        for drug in found:
            if drug.chembl_id in merged:
                held = merged[drug.chembl_id]
                actions = held.actions + tuple(a for a in drug.actions if a not in held.actions)
                merged[drug.chembl_id] = replace(held, actions=actions)
            else:
                merged[drug.chembl_id] = drug
    return looked_up, sorted(merged.values(), key=lambda d: (d.name.casefold(), d.chembl_id))


def add_safety(
    found: Sequence[Drug], openfda: OpenFDA, progress: Callable[[str], None] = lambda _message: None
) -> list[Drug]:
    """The drugs, each with its safety from openFDA."""
    kept = []
    for drug in found:
        safety = openfda.safety(drug.name)
        progress(
            f"{drug.name}: label concerns {', '.join(safety.concerns) or 'none'}"
            f"{'' if safety.label else ' (no US label)'}, {'a generic' if safety.generic else 'no generic'}"
        )
        kept.append(replace(drug, safety=safety))
    return kept


def update_drugs(
    library_path: Path,
    vocabulary: Vocabulary,
    out: Path,
    chembl_transport: Transport,
    mychem_transport: Transport,
    openfda_transport: Transport,
    risks: dict[str, tuple[Risk, ...]] | None = None,
    limiter: Callable[[], RateLimiter] = lambda: RateLimiter(RATE),
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    top: int = TOP_TARGETS,
    progress: Callable[[str], None] = lambda _message: None,
) -> DrugCache:
    """Score the library's graph, look up its top targets' drugs and their safety, and write them
    to `out`. `risks`: the `[risks]` table of `interventions.toml` (`safety.read_risks`)."""
    with Library.open(library_path) as library:
        version = library.version
        guard = PrivateTextGuard.from_library(library)
        targets = top_targets(bottlenecks(library_graph(library, vocabulary)), top)
    progress(f"looking up approved drugs for {len(targets)} targets: {', '.join(t.name for t in targets)}")
    chembl = ChEMBL(Service(chembl_transport, limiter(), guard, sleep))
    mychem = MyChem(Service(mychem_transport, limiter(), guard, sleep))
    openfda = OpenFDA(Service(openfda_transport, limiter(), guard, sleep), risks)
    looked_up, found = fetch_drugs(targets, chembl, mychem, progress)
    drugs = add_safety(found, openfda, progress)
    cache = DrugCache(now().isoformat(timespec="seconds"), version, tuple(looked_up), tuple(drugs))
    write_drug_cache(out, cache)
    return cache


def write_drug_cache(path: Path, cache: DrugCache) -> None:
    """Write `cache` whole, replacing the file only once it is written."""
    data = {"cache_version": CACHE_VERSION, **asdict(cache)}
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, scratch = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(scratch, path)
    except BaseException:
        Path(scratch).unlink(missing_ok=True)
        raise


def _safety(data: dict[str, Any] | None) -> Safety | None:
    if data is None:
        return None
    label = data["label"]
    return Safety(
        None
        if label is None
        else Label(
            label["set_id"],
            label["effective_time"],
            label["application"],
            tuple(Mention(**m) for m in label["mentions"]),
        ),
        tuple(Risk(**r) for r in data["risks"]),
        data["generic"],
        tuple(data["generic_applications"]),
    )


def read_drug_cache(path: Path = DEFAULT_DRUGS) -> DrugCache:
    """The cache `update_drugs` wrote; no network is needed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DrugsError(f"no drug data at {path}; run `asclepius-library drugs`") from error
    except (OSError, ValueError) as error:
        raise DrugsError(f"{path}: not a drug cache ({error})") from error
    if not isinstance(data, dict) or data.get("cache_version") != CACHE_VERSION:
        raise DrugsError(f"{path}: not a drug cache of this version; run `asclepius-library drugs`")
    try:
        return DrugCache(
            data["fetched_at"],
            data["library_version"],
            tuple(
                Target(t["symbol"], t["score"], tuple(t["syndromes"]), tuple(t["chembl_ids"]), t["source"])
                for t in data["targets"]
            ),
            tuple(
                Drug(
                    d["chembl_id"],
                    d["name"],
                    d["max_phase"],
                    {name: d["properties"].get(name) for name in PROPERTIES},
                    tuple(Action(**a) for a in d["actions"]),
                    d["known_cause"],
                    d["source"],
                    d["withdrawn"],
                    _safety(d["safety"]),
                )
                for d in data["drugs"]
            ),
        )
    except (KeyError, TypeError) as error:
        raise DrugsError(f"{path}: not a drug cache ({error})") from error
