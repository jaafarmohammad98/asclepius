"""Drug targets from ChEMBL, MyChem.info as fallback, cached at build time (M6-05a, spec R1, R2, R6).

The answers are recorded in fixtures/drugs (`record_responses.py drugs`, 2026-11-24); the drug
names below were read from them by hand.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from drugs_fixture import FOLDER, SAFETY_FOLDER, target
from replay import ReplayTransport
from test_rate_limit import FakeClock, _most_in_any_second

from hypothesis_engine import drugs
from hypothesis_engine.bottlenecks import Bottleneck
from hypothesis_engine.drugs import (
    ChEMBL,
    DrugsError,
    MyChem,
    Service,
    fetch_drugs,
    read_drug_cache,
    update_drugs,
)
from hypothesis_engine.library import write_build
from hypothesis_engine.privacy import PrivateTextGuard, PrivateTextLeak
from hypothesis_engine.private import PrivatePaper
from hypothesis_engine.pubmed import Paper, RateLimiter, TransientError
from hypothesis_engine.safety import Risk
from hypothesis_engine.vocabulary import Term, Vocabulary

WHEN = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
PSSD, PFS = "post-SSRI sexual dysfunction", "post-finasteride syndrome"


class Recorded:
    """The replayed services, plus the times their requests were sent."""

    def __init__(self, chembl_down: bool = False, guard: PrivateTextGuard | None = None) -> None:
        self.clock = FakeClock()
        self.replay = ReplayTransport(FOLDER)
        self.openfda = ReplayTransport(SAFETY_FOLDER)
        self.chembl_down = chembl_down
        self.sent_at: list[float] = []
        self.guard = guard

    def _send(self, path: str, params: dict[str, str]) -> bytes:
        self.sent_at.append(self.clock.now)
        return self.replay(path, params)

    def chembl_transport(self, path: str, params: dict[str, str]) -> bytes:
        if self.chembl_down:
            self.sent_at.append(self.clock.now)
            raise TransientError(f"{path}: HTTP 503")
        return self._send(path, params)

    def limiter(self) -> RateLimiter:
        return RateLimiter(drugs.RATE, self.clock.time, self.clock.sleep)

    def services(self) -> tuple[ChEMBL, MyChem]:
        return (
            ChEMBL(Service(self.chembl_transport, self.limiter(), self.guard, self.clock.sleep)),
            MyChem(Service(self._send, self.limiter(), self.guard, self.clock.sleep)),
        )


def _fetch(*symbols: str, recorded: Recorded | None = None):
    recorded = recorded or Recorded()
    return fetch_drugs([target(s) for s in symbols], *recorded.services())


def _names(found) -> set[str]:
    return {d.name for d in found}


def test_the_approved_drugs_acting_on_a_target_and_its_families_are_found():
    _targets, found = _fetch("SRD5A2")
    # finasteride acts on SRD5A2 itself; dutasteride and abiraterone on the 5-alpha reductase family
    assert _names(found) == {"ABIRATERONE", "DUTASTERIDE", "FINASTERIDE"}


def test_drugs_that_cause_the_syndromes_are_tagged_for_exclusion():
    _targets, found = _fetch("SRD5A2", "SLC6A4")
    by_name = {d.name: d for d in found}
    assert (
        by_name["FINASTERIDE"].known_cause
        == by_name["DUTASTERIDE"].known_cause
        == "5-alpha reductase inhibitor"
    )
    assert by_name["PAROXETINE"].known_cause == by_name["SERTRALINE"].known_cause == "SSRI"
    assert by_name["DULOXETINE"].known_cause == "SNRI"
    assert by_name["VORTIOXETINE"].known_cause is None  # acts on the transporter, not counted an SSRI


def test_every_known_cause_is_tagged_however_chembl_names_or_classes_it():
    assert (
        drugs.known_cause("INDALPINE")
        == drugs.known_cause("ZIMELIDINE")
        == drugs.known_cause("ZIMELDINE")
        == "SSRI"
    )
    assert drugs.known_cause("SIBUTRAMINE") == "SNRI"
    assert drugs.known_cause("OLANZAPINE AND FLUOXETINE") == "SSRI"  # a combination product
    assert drugs.known_cause("NEWOXETINE", ["N06AB99"]) == "SSRI"  # by its ATC class
    assert drugs.known_cause("BUPROPION", ["N06AX12"]) is None


def test_the_recorded_withdrawn_reuptake_inhibitors_are_excluded():
    _targets, found = _fetch("SLC6A4")
    by_name = {d.name: d for d in found}
    assert (by_name["INDALPINE"].known_cause, by_name["SIBUTRAMINE"].known_cause) == ("SSRI", "SNRI")


def test_a_target_keeps_the_syndromes_linking_to_it():
    recorded = Recorded()
    found = Bottleneck("SRD5A2", "gene", 0.9, 0.1, ("PFS", "PSSD"), 1.0)
    (srd5a2,), _drugs = fetch_drugs([found], *recorded.services())
    assert srd5a2.syndromes == ("PFS", "PSSD")


def test_a_drug_keeps_its_phase_properties_and_chembl_id():
    _targets, found = _fetch("SRD5A2")
    (finasteride,) = [d for d in found if d.name == "FINASTERIDE"]
    assert finasteride.chembl_id == "CHEMBL710"
    assert finasteride.max_phase == 4
    assert finasteride.url == "https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL710"
    properties = finasteride.properties
    assert (properties["mw_freebase"], properties["psa"], properties["hbd"], properties["alogp"]) == (
        372.55,
        58.2,
        2,
        3.81,
    )
    assert set(properties) == set(drugs.PROPERTIES)
    (action,) = finasteride.actions
    assert (action.target, action.target_chembl_id, action.action_type) == (
        "SRD5A2",
        "CHEMBL1856",
        "INHIBITOR",
    )


def test_only_approved_drugs_are_kept():
    _targets, found = _fetch("SRD5A2", "SLC6A4", "TSPO")
    assert found and all(d.max_phase == 4 for d in found)


def test_a_drug_withdrawn_since_its_approval_is_kept_and_flagged():
    targets, found = _fetch("TSPO")
    assert [(d.name, d.withdrawn) for d in found] == [("ALPIDEM", True)]
    assert [t.symbol for t in targets] == ["TSPO"]
    assert targets[0].chembl_ids and targets[0].source == "chembl"
    _targets, finasteride = _fetch("SRD5A2")
    assert not any(d.withdrawn for d in finasteride)


def test_a_salt_counts_as_its_parent_molecule():
    _targets, found = _fetch("SLC6A4")
    assert not any(
        " " in d.name and d.name.split()[-1] in ("HYDROCHLORIDE", "MALEATE", "MESYLATE") for d in found
    )
    assert len({d.chembl_id for d in found}) == len(found)


def test_a_drug_acting_on_two_targets_is_listed_once_with_both():
    one = Recorded()
    _targets, found = fetch_drugs([target("SRD5A2"), target("SRD5A2")], *one.services())
    (dutasteride,) = [d for d in found if d.name == "DUTASTERIDE"]
    assert [a.target for a in dutasteride.actions] == ["SRD5A2"]  # the same action is not repeated


def test_when_chembl_is_down_mychem_answers_for_the_target():
    recorded = Recorded(chembl_down=True)
    targets, found = _fetch("SRD5A2", recorded=recorded)
    assert targets[0].source == "mychem" and targets[0].chembl_ids == ()
    assert {"FINASTERIDE", "DUTASTERIDE"} <= _names(found)
    assert all(d.source == "mychem" and d.max_phase == 4 and d.actions for d in found)
    assert all(a.target == "SRD5A2" for d in found for a in d.actions)
    by_name = {d.name: d for d in found}
    assert by_name["FINASTERIDE"].known_cause == "5-alpha reductase inhibitor"
    assert by_name["FINASTERIDE"].properties["mw_freebase"] == 372.55


def test_requests_stay_under_three_a_second_to_each_service():
    recorded = Recorded()
    _fetch("SRD5A2", "SLC6A4", "TSPO", recorded=recorded)
    assert len(recorded.sent_at) > 3
    assert _most_in_any_second(recorded.sent_at) <= 3


def test_only_target_names_and_chembl_ids_are_sent():
    recorded = Recorded()
    _fetch("SRD5A2", "SLC6A4", recorded=recorded)
    fixed = {"limit", "offset", "fields", "size", "organism"}  # the same in every request
    for _path, params in recorded.replay.calls:
        for key, value in params.items():
            if key in fixed:
                continue
            for word in value.replace(":", ",").split(","):
                assert word in ("SRD5A2", "SLC6A4", "drugcentral.bioactivity.uniprot.gene_symbol") or (
                    word.startswith("CHEMBL") and word[6:].isdigit()
                ), (key, value)


def test_a_request_holding_private_text_is_refused_before_it_is_sent():
    secret = "the clinic held the whole series back until the second sample was taken in spring"
    guard = PrivateTextGuard(
        [PrivatePaper("private-1", "mine.pdf", "Mine", secret)], max_quote_words=3, max_total_words=3
    )
    recorded = Recorded(guard=guard)
    chembl, _mychem = recorded.services()
    with pytest.raises(PrivateTextLeak):
        chembl.targets(secret)
    assert recorded.replay.calls == []


# --- the cache, written at build time and read by the app ---

VOCABULARY = Vocabulary(
    terms=(
        Term(PSSD, "syndrome", (), "100", abbreviations=("PSSD",)),
        Term(PFS, "syndrome", (), "200", abbreviations=("PFS",)),
        Term("SLC6A4", "gene", (), "100"),
        Term("TSPO", "gene", (), "100"),
        Term("SRD5A2", "gene", (), "200"),
        Term("anhedonia", "symptom", (), "100"),
    )
)


@pytest.fixture
def library(tmp_path):
    papers = [
        Paper("100", "PSSD", "In PSSD, SLC6A4, TSPO and anhedonia. SRD5A2 too.", None, None, None, 2021),
        Paper("200", "PFS", "SRD5A2 in PFS, with SLC6A4.", None, None, None, 2021),
    ]
    found_by = {"100": [("PSSD", 1)], "200": [("PFS", 1)]}
    private = [PrivatePaper("private-1", "mine.pdf", "Mine", "Unpublished notes on SRD5A2 in PFS.")]
    path = tmp_path / "library.sqlite"
    write_build(path, papers, found_by, "sha", "2026-11-24T00:00:00+00:00", private)
    return path


def _update(library, out, recorded: Recorded | None = None, top: int = 3, risks=None):
    recorded = recorded or Recorded()
    return update_drugs(
        library,
        VOCABULARY,
        out,
        recorded.chembl_transport,
        recorded._send,
        recorded.openfda,
        risks,
        limiter=recorded.limiter,
        sleep=recorded.clock.sleep,
        now=lambda: WHEN,
        top=top,
    )


def test_the_top_bottleneck_genes_are_looked_up(library, tmp_path):
    cache = _update(library, tmp_path / "drugs.json")
    assert {t.symbol for t in cache.targets} == {"SRD5A2", "SLC6A4", "TSPO"}
    assert [t.score for t in cache.targets] == sorted((t.score for t in cache.targets), reverse=True)


def test_only_as_many_targets_as_asked_for_are_looked_up(library, tmp_path):
    cache = _update(library, tmp_path / "drugs.json", top=1)
    assert len(cache.targets) == 1


def test_after_a_build_the_cache_reads_back_without_the_network(library, tmp_path):
    out = tmp_path / "drugs.json"
    written = _update(library, out)
    read = read_drug_cache(out)  # no transport: nothing could be fetched
    assert read == written
    assert read.fetched_at == "2026-11-24T12:00:00+00:00" and read.library_version == 1
    assert {"FINASTERIDE", "PAROXETINE"} <= {d.name for d in read.excluded}
    assert read.ranked and all(d.known_cause is None for d in read.ranked)


def test_every_cached_drug_has_its_safety_and_generic_flag(library, tmp_path):
    out = tmp_path / "drugs.json"
    risks = {"vortioxetine": (Risk("nausea", "PMID:1"),)}
    written = _update(library, out, risks=risks)
    read = read_drug_cache(out)
    assert read == written
    assert all(d.safety is not None for d in read.drugs)
    by_name = {d.name: d for d in read.drugs}
    vortioxetine, alpidem = by_name["VORTIOXETINE"].safety, by_name["ALPIDEM"].safety
    assert vortioxetine.concerns == ("sexual dysfunction", "libido") and not vortioxetine.generic
    assert vortioxetine.risks == risks["vortioxetine"] and vortioxetine.penalty == 3
    assert vortioxetine.label.mentions[0].excerpt  # the label text the penalty came from
    assert alpidem.label is None and alpidem.penalty == 0
    assert by_name["FINASTERIDE"].safety.generic


def test_a_cache_of_the_version_before_safety_is_fetched_again(tmp_path):
    out = tmp_path / "drugs.json"
    out.write_text(json.dumps({"cache_version": 1}), encoding="utf-8")
    with pytest.raises(DrugsError, match="asclepius-library drugs"):
        read_drug_cache(out)


def test_a_failed_fetch_leaves_the_previous_cache_as_it_was(library, tmp_path):
    out = tmp_path / "drugs.json"
    _update(library, out)
    before = out.read_bytes()

    class Down(Recorded):
        def _send(self, path, params):
            raise TransientError("HTTP 503")

    with pytest.raises(TransientError):
        _update(library, out, Down(chembl_down=True))
    assert out.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".drugs")] == []


def test_reading_a_missing_cache_names_the_command(tmp_path):
    with pytest.raises(DrugsError, match="asclepius-library drugs"):
        read_drug_cache(tmp_path / "drugs.json")


def test_reading_a_cache_of_another_version_fails(tmp_path):
    out = tmp_path / "drugs.json"
    out.write_text(json.dumps({"cache_version": 99}), encoding="utf-8")
    with pytest.raises(DrugsError, match="version"):
        read_drug_cache(out)


def test_a_short_symbol_finds_its_targets():
    _targets, found = _fetch("AR")
    assert {"BICALUTAMIDE", "ENZALUTAMIDE", "TESTOSTERONE"} <= _names(found)


def test_mychem_answers_holding_lists_at_any_level_are_read():
    # As MyChem.info gives some molecules (AR's, 2026-11-24): drugcentral, bioactivity and uniprot
    # each one record or a list of them.
    answer = {
        "hits": [
            {
                "chembl": [{"molecule_chembl_id": "CHEMBL1", "pref_name": "ONE", "max_phase": 4.0}],
                "drugcentral": [
                    {"bioactivity": {"moa": 1, "organism": drugs.HUMAN, "action_type": "AGONIST",
                                     "uniprot": [{"gene_symbol": "XYZ"}, {"gene_symbol": "AR"}]}},
                    {"bioactivity": [{"moa": "1", "organism": drugs.HUMAN, "action_type": "ANTAGONIST",
                                      "uniprot": {"gene_symbol": "AR"}}]},
                ],
            },
            {"chembl": {"molecule_chembl_id": "CHEMBL2", "max_phase": 4}, "drugcentral": None},
        ]
    }  # fmt: skip
    clock = FakeClock()
    mychem = MyChem(
        Service(lambda _p, _q: json.dumps(answer).encode(), RateLimiter(3, clock.time, clock.sleep))
    )
    (one,) = mychem.drugs("AR")
    assert one.name == "ONE" and [a.action_type for a in one.actions] == ["AGONIST", "ANTAGONIST"]
