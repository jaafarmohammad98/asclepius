"""Each drug's label concerns, listed risks and generics, from openFDA (M6-05b, spec R2, R3).

The answers are recorded in fixtures/openfda (`record_responses.py safety`, 2026-11-25), cut down
to the parts `safety.py` reads; the values below were read from them by hand.
"""

from __future__ import annotations

import json

import pytest
from drugs_fixture import SAFETY_FOLDER
from replay import ReplayTransport
from test_rate_limit import FakeClock

from hypothesis_engine import safety
from hypothesis_engine.drugs import Service
from hypothesis_engine.privacy import PrivateTextGuard, PrivateTextLeak
from hypothesis_engine.private import PrivatePaper
from hypothesis_engine.pubmed import RateLimiter
from hypothesis_engine.safety import OpenFDA, Risk, SafetyError, load_risks


def _openfda(risks=None, guard=None) -> tuple[OpenFDA, ReplayTransport]:
    risks = risks or {}
    clock = FakeClock()
    replay = ReplayTransport(SAFETY_FOLDER)
    return OpenFDA(
        Service(replay, RateLimiter(3, clock.time, clock.sleep), guard, clock.sleep), risks
    ), replay


def test_a_label_stating_sexual_side_effects_is_penalised_with_the_text_it_states_them_in():
    openfda, _replay = _openfda()
    finasteride = openfda.safety("FINASTERIDE")
    assert finasteride.concerns == ("sexual dysfunction", "libido")
    assert finasteride.penalty == 2
    sexual, libido = finasteride.label.mentions
    assert sexual.section == libido.section == "adverse_reactions"
    assert "impotence, decreased libido" in sexual.excerpt
    assert finasteride.label.set_id == "6f988153-fc74-4ca4-b29a-111f750c4a4b"
    assert finasteride.label.url.endswith("setid=6f988153-fc74-4ca4-b29a-111f750c4a4b")


def test_the_brand_name_label_is_read_before_a_generic_one():
    openfda, _replay = _openfda()
    paroxetine = openfda.safety("PAROXETINE")  # PAXIL CR's label, not the latest generic's
    assert paroxetine.label.application == "NDA020936"
    assert paroxetine.label.mentions[0].section == "warnings_and_cautions"
    assert "decreased libido" in paroxetine.label.mentions[1].excerpt


def test_a_label_stating_none_of_the_concerns_has_no_label_penalty():
    openfda, _replay = _openfda()
    lumateperone = openfda.safety("LUMATEPERONE")
    assert lumateperone.label is not None and lumateperone.label.mentions == ()
    assert lumateperone.penalty == 0


def test_a_drug_never_sold_in_the_us_has_no_label_and_no_generic():
    openfda, _replay = _openfda()
    alpidem = openfda.safety("ALPIDEM")  # openFDA answers 404 "No matches found!"
    assert (alpidem.label, alpidem.generic, alpidem.generic_applications) == (None, False, ())
    assert alpidem.penalty is None  # unknown, not safe: M6-06 decides what stands in for it


def test_a_drug_with_a_marketed_anda_has_a_generic():
    openfda, _replay = _openfda()
    finasteride = openfda.safety("FINASTERIDE")
    assert finasteride.generic and "ANDA090121" in finasteride.generic_applications
    assert "ANDA077335" not in finasteride.generic_applications  # discontinued


def test_a_drug_whose_andas_are_not_yet_marketed_has_no_generic():
    openfda, _replay = _openfda()
    vortioxetine = openfda.safety("VORTIOXETINE")  # still under patent: no ANDA product sold
    assert not vortioxetine.generic and vortioxetine.label is not None


def test_a_salt_counts_as_its_parent_and_a_combination_does_not():
    assert safety.same_ingredients("PAROXETINE", ["PAROXETINE HYDROCHLORIDE HEMIHYDRATE"])
    assert not safety.same_ingredients("FLUOXETINE", ["OLANZAPINE AND FLUOXETINE HYDROCHLORIDE"])
    assert safety.same_ingredients("OLANZAPINE AND FLUOXETINE", ["FLUOXETINE HCL", "OLANZAPINE"])
    assert not safety.same_ingredients("TESTOSTERONE", ["METHYLTESTOSTERONE"])
    assert safety.same_ingredients("TESTOSTERONE CYPIONATE", ["TESTOSTERONE CYPIONATE"])
    assert safety.same_ingredients("TRAZODONE", ["TRAZODONE HYDROCHLORIDE ORAL"])  # a dosage form


def test_another_drug_containing_the_name_is_not_the_drug():
    assert not safety.same_ingredients("ESTRADIOL", ["ETHINYL ESTRADIOL"])
    assert not safety.same_ingredients("TESTOSTERONE", ["TESTOSTERONE CYPIONATE"])  # an ester is its own drug


def _concerns(text: str, section: str = "adverse_reactions") -> tuple[str, ...]:
    return tuple(m.concern for m in safety.find_mentions({section: [text]}))


def test_a_concern_the_label_says_was_not_seen_is_not_counted():
    assert _concerns("Sexual dysfunction was not observed in clinical trials.") == ()
    assert _concerns("There were no reports of decreased libido or anhedonia.") == ()
    assert _concerns("Patients reported nausea. Decreased libido occurred in 3%.") == ("libido",)


def test_more_sexual_desire_is_not_a_concern():
    assert _concerns("Increased libido and hypersexuality were reported.") == ()
    assert _concerns("ADDYI is indicated for hypoactive sexual desire disorder (HSDD).") == ()


def test_the_wordings_labels_use_for_the_concerns_are_found():
    for wording in (
        "decreased sexual function",
        "sexual adverse effects",
        "orgasm abnormal",
        "sexual function abnormal",
    ):
        assert _concerns(f"Reported: {wording}.") == ("sexual dysfunction",), wording
    assert _concerns("Reported: loss of sexual interest.") == ("libido",)


def test_a_label_of_the_drug_is_found_behind_a_page_of_combination_products():
    combination = {
        "set_id": "combo",
        "openfda": {"generic_name": ["OLANZAPINE AND FLUOXETINE HYDROCHLORIDE"]},
    }
    own = {
        "set_id": "own",
        "openfda": {"generic_name": ["FLUOXETINE HYDROCHLORIDE"], "application_number": ["NDA018936"]},
        "adverse_reactions": ["Decreased libido."],
    }

    class Pages:
        def get(self, path, params):
            skip = int(params["skip"])
            labels = [combination] * safety.LABEL_PAGE if skip == 0 else [own]
            return {"meta": {"results": {"total": safety.LABEL_PAGE + 1}}, "results": labels}

    label = OpenFDA(Pages(), {}).label("FLUOXETINE")
    assert label is not None and label.set_id == "own" and label.application == "NDA018936"


def test_any_of_a_labels_generic_names_can_be_the_drug():
    label = {"set_id": "x", "openfda": {"generic_name": ["IBUPROFEN AND FAMOTIDINE", "FINASTERIDE"]}}

    class One:
        def get(self, path, params):
            return {"meta": {"results": {"total": 1}}, "results": [label]}

    assert OpenFDA(One(), {}).label("FINASTERIDE").set_id == "x"


def test_risks_listed_for_the_drug_add_to_its_penalty():
    listed = {"finasteride": (Risk("persistent sexual side effects after stopping", "PMID:1"),)}
    openfda, _replay = _openfda(listed | {"alpidem": (Risk("hepatotoxicity", "PMID:2"),)})
    finasteride = openfda.safety("FINASTERIDE")
    assert finasteride.risks == listed["finasteride"]
    assert finasteride.penalty == 3  # two label concerns and the listed risk
    assert openfda.safety("ALPIDEM").risks  # kept and shown, though its penalty is unknown
    assert openfda.safety("PAROXETINE").risks == ()


def test_only_the_drug_name_is_sent():
    openfda, replay = _openfda()
    openfda.safety("FINASTERIDE")
    for _path, params in replay.calls:
        searched = params["search"].replace(" AND ", " ")
        assert set(searched.split()) <= {
            'openfda.generic_name:"FINASTERIDE"',
            "openfda.application_number:NDA*",
            'products.active_ingredients.name:"FINASTERIDE"',
            "application_number:ANDA*",
        }


def test_a_request_holding_private_text_is_refused_before_it_is_sent():
    secret = "the clinic held the whole series back until the second sample was taken in spring"
    guard = PrivateTextGuard(
        [PrivatePaper("private-1", "mine.pdf", "Mine", secret)], max_quote_words=3, max_total_words=3
    )
    openfda, replay = _openfda(guard=guard)
    with pytest.raises(PrivateTextLeak):
        openfda.safety(secret)
    assert replay.calls == []


def test_an_openfda_error_other_than_no_match_fails():
    clock = FakeClock()
    answer = json.dumps({"error": {"code": "BAD_REQUEST", "message": "Syntax error"}}).encode()
    openfda = OpenFDA(Service(lambda _p, _q: answer, RateLimiter(3, clock.time, clock.sleep)), {})
    with pytest.raises(SafetyError, match="Syntax error"):
        openfda.safety("FINASTERIDE")


# --- the [risks] table of interventions.toml ---


def test_the_risks_table_is_read_with_case_aside():
    text = """
names = ["bupropion"]
[risks]
"Bupropion" = [{ risk = "lowers the seizure threshold", source = "PMID:123" }]
"""
    assert load_risks(text) == {"bupropion": (Risk("lowers the seizure threshold", "PMID:123"),)}


def test_a_risk_without_a_cited_source_is_refused():
    text = '[risks]\n"bupropion" = [{ risk = "seizures", source = "a forum post" }]\n'
    with pytest.raises(SafetyError, match="cites no PMID, DOI or https URL"):
        load_risks(text)


def test_a_risk_with_other_fields_is_refused():
    text = '[risks]\n"bupropion" = [{ risk = "seizures", source = "PMID:1", note = "x" }]\n'
    with pytest.raises(SafetyError, match="risk, source"):
        load_risks(text)


def test_the_shipped_interventions_file_reads():
    assert isinstance(safety.read_risks(), dict)
