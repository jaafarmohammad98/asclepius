"""
Clean the extracted cases (no API calls, free, re-runnable).

  - merges drug / treatment name variants (brand -> generic, spelling and suffix variants)
  - groups free-text 'other' symptoms into extra categories by keyword rules
  - forces enum fields back into their allowed values
  - drops malformed or duplicate items

Usage (from harvester/):
  .venv/bin/python clean_cases.py --inp data/cases.jsonl --out data/cases_clean.jsonl --csv data/export

The CSV export is for analysis. It leaves out the Reddit id and the verbatim evidence quotes,
because a quote pasted into a search engine finds the original post and its author.
"""

import argparse, collections, csv, hashlib, json, re
from pathlib import Path

from extract_cases import SYMPTOMS, OUTCOMES

# ---------------------------------------------------------------- names

ALIASES = {
    # brands -> generic
    "zoloft": "sertraline", "lexapro": "escitalopram", "cipralex": "escitalopram", "prozac": "fluoxetine",
    "paxil": "paroxetine", "seroxat": "paroxetine", "celexa": "citalopram", "effexor": "venlafaxine",
    "cymbalta": "duloxetine", "trintellix": "vortioxetine", "brintellix": "vortioxetine",
    "remeron": "mirtazapine", "wellbutrin": "bupropion", "zyban": "bupropion", "luvox": "fluvoxamine",
    "pristiq": "desvenlafaxine", "abilify": "aripiprazole", "rexulti": "brexpiprazole",
    "seroquel": "quetiapine", "zyprexa": "olanzapine", "risperdal": "risperidone", "anafranil": "clomipramine",
    "accutane": "isotretinoin", "propecia": "finasteride", "proscar": "finasteride",
    "cialis": "tadalafil", "viagra": "sildenafil", "levitra": "vardenafil", "buspar": "buspirone",
    "lamictal": "lamotrigine", "ritalin": "methylphenidate", "concerta": "methylphenidate",
    "adderall": "amphetamine", "vyvanse": "lisdexamfetamine", "provigil": "modafinil",
    "dostinex": "cabergoline", "clomid": "clomiphene", "arimidex": "anastrozole", "valdoxan": "agomelatine",
    "nardil": "phenelzine", "parnate": "tranylcypromine", "klonopin": "clonazepam", "xanax": "alprazolam",
    "lyrica": "pregabalin", "neurontin": "gabapentin", "periactin": "cyproheptadine", "xifaxan": "rifaximin",
    "mirapex": "pramipexole", "spravato": "esketamine", "desyrel": "trazodone",
    # class-only spellings
    "ssri": "unspecified_ssri", "ssris": "unspecified_ssri", "snri": "unspecified_snri",
    "antidepressant": "unspecified_antidepressant", "antidepressants": "unspecified_antidepressant",
    "antipsychotic": "unspecified_antipsychotic", "antipsychotics": "unspecified_antipsychotic",
    "antipsychotic_unspecified": "unspecified_antipsychotic", "benzodiazepine": "unspecified_benzodiazepine",
    "benzodiazepines": "unspecified_benzodiazepine", "birth_control_pill": "hormonal_contraceptive",
    "birth_control": "hormonal_contraceptive", "hypericum": "st_johns_wort", "st_john_s_wort": "st_johns_wort",
    # treatment variants
    "testosterone": "testosterone_replacement", "trt": "testosterone_replacement",
    "testosterone_replacement_therapy": "testosterone_replacement",
    "vitamin_d3": "vitamin_d", "d3": "vitamin_d", "vitamin_b12": "vitamin_b12", "b12": "vitamin_b12",
    "probiotic": "probiotics", "psilocybin_mushrooms": "psilocybin", "magic_mushrooms": "psilocybin",
    "mushrooms": "psilocybin", "nac": "n_acetylcysteine", "n_acetyl_cysteine": "n_acetylcysteine",
    "citrulline": "l_citrulline", "arginine": "l_arginine", "tyrosine": "l_tyrosine",
    "ginkgo": "ginkgo_biloba", "omega_3s": "omega_3", "fish_oil": "omega_3", "ldn": "low_dose_naltrexone",
    "fmt": "fecal_microbiota_transplant", "fecal_transplant": "fecal_microbiota_transplant",
    "transcranial_magnetic_stimulation": "tms", "rtms": "tms", "weed": "cannabis", "marijuana": "cannabis",
    "thc": "cannabis", "therapy": "psychotherapy", "cbt": "psychotherapy", "lions_mane_mushroom": "lions_mane",
    "ginseng": "panax_ginseng", "korean_red_ginseng": "panax_ginseng", "mucuna": "mucuna_pruriens",
    "keto": "ketogenic_diet", "keto_diet": "ketogenic_diet", "pelvic_floor_therapy": "pelvic_floor_physical_therapy",
    "pelvic_floor_pt": "pelvic_floor_physical_therapy", "working_out": "exercise", "gym": "exercise",
    "weightlifting": "exercise", "running": "exercise", "no_fap": "nofap",
    "mao_inhibitor_phenelzine": "phenelzine", "maoi": "unspecified_maoi",
}
NOT_A_NAME = {"", "unspecified", "unclear", "unknown", "none", "other", "various", "n_a"}

def norm_name(raw):
    s = str(raw or "").lower().strip()
    s = re.sub(r"\(.*?\)", "", s)                 # "modafinil (sexual function)" -> "modafinil"
    s = s.replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")  # spaces, hyphens, slashes -> _
    s = ALIASES.get(s, s)
    return None if s in NOT_A_NAME else s

# ---------------------------------------------------------------- 'other' symptoms

# first match wins; order matters (specific before general)
OTHER_RULES = [
    (r"pgad|persistent genital arousal|genital arousal disorder", "persistent_genital_arousal"),
    (r"dysautonom|\bpots\b|orthostatic", "dysautonomia_or_temperature"),
    (r"akathisia|restless(ness)?(?! leg)|inner restless|pacing", "akathisia"),
    (r"restless leg|tremor|twitch|dystonia|dyskinesia|hypnic jerk|\btics?\b|spasm|myoclon", "movement_problems"),
    (r"brain zap", "brain_zaps"),
    (r"head pressure|headache|migraine", "headache_or_head_pressure"),
    (r"visual snow|floater|blur|vision|light sensitiv|photophob|eye pain|eyesight", "visual_problems"),
    (r"aphantasia|inner (dialogue|monologue|voice)|imagination|visuali[sz]", "loss_of_inner_imagery_or_monologue"),
    (r"(inability|unable|can.?t|cannot) to cry|crying", "cannot_cry_or_crying_changes"),
    (r"smell|taste", "smell_or_taste_loss"),
    (r"palpitation|tachycard|heart rate|blood pressure|hypertens|chest pain", "heart_or_blood_pressure"),
    (r"dysautonom|\bpots\b|orthostatic|temperature|cold (hands|feet)|heat intoleran|exercise intoleran", "dysautonomia_or_temperature"),
    (r"neuropath|nerve pain|tingling|pins and needles|burning|full body numb|skin sensation|pain sensation|muscle numb", "neuropathy_or_body_numbness"),
    (r"soft glans|hard flaccid|cold genital|glans", "glans_or_flaccid_changes"),
    (r"testic|penile pain|penis pain|genital pain|pelvic|perine|painful erection|varicocele", "genital_or_pelvic_pain"),
    (r"nipple|breast", "nipple_or_breast_changes"),
    (r"delayed ejac|long refractory|refractory period", "delayed_ejaculation_or_long_refractory"),
    (r"weight (gain|loss)|gained weight|lost weight", "weight_change"),
    (r"irritab|agitation|\banger|\brage\b|mood swing|aggress", "irritability_or_mood_swings"),
    (r"motivation|apathy|avolition", "low_motivation"),
    (r"concentrat|focus|executive|blank mind|cognitive|attention", "brain_fog"),   # existing category
    (r"intrusive|obsess|ocd|compuls", "intrusive_or_obsessive_thoughts"),
    (r"nightmare|dream", "dream_changes"),
    (r"suicid", "suicidal_thoughts"),                                              # existing category
    (r"deperson|dereal|dissociat|stranger to (my)?self|unreal", "depersonalization_derealization"),  # existing
    (r"hair (loss|thinning|falling)|losing hair|balding", "hair_loss"),                                                         # existing
    (r"sweat|hot flash", "sweating_or_hot_flashes"),                                # existing
    (r"urinat|bladder|\bpee\b", "urinary_problems"),                                  # existing
    (r"dizz|balance|vertigo", "dizziness_or_balance"),                              # existing
    (r"(loss of|no|low|reduced|decreased|poor|lost) appetite|appetite loss", "appetite_loss"),                                                 # existing
    (r"hallucinat|psychos|paranoi", "psychotic_symptoms"),
    (r"constipat|nausea|bloat|\bibs\b|stomach|digest|abdominal", "gut_issues"),                   # existing
    (r"skin|rash|hives|psoriasis|\bitch|acne", "skin_problems"),
    (r"emotion|feel nothing|numb(ness)? (to|of) feel|connection", "emotional_blunting"),  # existing
    (r"arousal|sexual thought|sex drive|libido", "low_libido"),                     # existing
]
# "sexual dysfunction (unspecified)", "pssd symptoms" etc. name the condition, not a symptom
NOT_A_SYMPTOM = re.compile(r"^(pssd|pfs|psa|post.ssri|sexual (dysfunction|problems|side effects)|unspecified|symptoms?)\b"
                           r"|loss of (relationships|career|job)|unspecified symptoms")
OTHER_RULES = [(re.compile(p), cat) for p, cat in OTHER_RULES]
EXTRA_SYMPTOMS = sorted({cat for _, cat in OTHER_RULES} - set(SYMPTOMS))

def classify_other(desc):
    d = (desc or "").lower().strip()
    if not d or NOT_A_SYMPTOM.search(d):
        return None
    for rx, cat in OTHER_RULES:
        if rx.search(d):
            return cat
    return "other"

# ---------------------------------------------------------------- enums

ENUMS = {
    "condition": {"PSSD", "PFS", "PSA", "multiple", "unclear", "none"},
    "sex": {"male", "female", "other"},
    "onset": {"during_use", "after_stopping", "unclear"},
    "overall_trajectory": {"recovered", "improving", "stable", "worsening", "unclear"},
}
NEAR = {  # values the model invented -> closest allowed value
    "fluctuating": "unclear", "mixed": "unclear", "partial recovery": "improving", "partial": "improving",
    "partial_improvement": "improving", "improved": "improving", "worse": "worsening", "recovering": "improving",
}

def fix_enum(v, allowed):
    if v is None:
        return None
    s = str(v).strip().strip('"\'')
    if s in allowed:
        return s
    s = NEAR.get(s.lower(), s.lower())
    return s if s in allowed else None

def num(v):
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------- clean one case

def clean(c, stats):
    for k, allowed in ENUMS.items():
        before = c.get(k)
        c[k] = fix_enum(before, allowed)
        if before is not None and c[k] != before:
            stats[f"enum fixed: {k}"] += 1

    drugs, seen = [], set()
    for d in c.get("causal_drugs", []):
        n = norm_name(d.get("name"))
        if not n or n in seen:
            continue
        seen.add(n)
        drugs.append({**d, "name": n, "dose_mg": num(d.get("dose_mg")), "duration_months": num(d.get("duration_months"))})
    c["causal_drugs"] = drugs

    syms, seen = [], set()
    for s in c.get("symptoms", []):
        sym = s.get("symptom")
        if sym == "other" or sym not in SYMPTOMS:
            stats["other symptoms in"] += 1
            sym = classify_other(s.get("other_description") or (sym if sym not in SYMPTOMS else ""))
            if sym is None:
                stats["other dropped (not a symptom)"] += 1
                continue
            stats["other still other" if sym == "other" else "other regrouped"] += 1
        if sym in seen and sym != "other":
            continue
        seen.add(sym)
        sev = s.get("severity")
        syms.append({**s, "symptom": sym, "severity": sev if sev in ("mild", "moderate", "severe") else None})
    c["symptoms"] = syms

    ivs, seen = [], set()
    for i in c.get("interventions", []):
        n = norm_name(i.get("name"))
        if not n:
            stats["intervention dropped (no name)"] += 1
            continue
        out = i.get("outcome") if i.get("outcome") in OUTCOMES else "unclear"
        if (n, out) in seen:
            continue
        seen.add((n, out))
        ivs.append({**i, "name": n, "outcome": out, "duration_weeks": num(i.get("duration_weeks"))})
    c["interventions"] = ivs

    c["labs"] = [l for l in c.get("labs", []) if num(l.get("value")) is not None and l.get("test")]
    c["current_meds"] = sorted({n for m in c.get("current_meds") or [] if (n := norm_name(m))})
    return c

# ---------------------------------------------------------------- export

def export_csv(cases, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    cid = lambda c: hashlib.sha256(c["id"].encode()).hexdigest()[:12]   # not reversible to the Reddit id
    def write(name, header, rows):
        with open(outdir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    write("cases.csv",
          ["case_id", "author_key", "year", "kind", "condition", "sex", "age", "onset", "months_since_stopping",
           "overall_trajectory", "evidence_ok"],
          [[cid(c), c["author_key"], c.get("year"), c.get("kind"), c["condition"], c["sex"], c.get("age"),
            c["onset"], c.get("months_since_stopping"), c["overall_trajectory"], c["evidence_ok"]] for c in cases])
    write("causal_drugs.csv", ["case_id", "drug", "dose_mg", "duration_months"],
          [[cid(c), d["name"], d["dose_mg"], d["duration_months"]] for c in cases for d in c["causal_drugs"]])
    write("symptoms.csv", ["case_id", "symptom", "severity"],
          [[cid(c), s["symptom"], s["severity"]] for c in cases for s in c["symptoms"]])
    write("interventions.csv", ["case_id", "intervention", "outcome", "duration_weeks"],
          [[cid(c), i["name"], i["outcome"], i["duration_weeks"]] for c in cases for i in c["interventions"]])
    write("labs.csv", ["case_id", "test", "value", "unit", "ref_low", "ref_high", "flag", "panel"],
          [[cid(c), l["test"], l["value"], l.get("unit"), l.get("ref_low"), l.get("ref_high"), l.get("flag"), l.get("panel")]
           for c in cases for l in c["labs"]])

# ---------------------------------------------------------------- main

if __name__ == "__main__":
    import datetime
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True); p.add_argument("--out", required=True)
    p.add_argument("--csv", help="directory for the analysis CSVs (first-person cases only)")
    a = p.parse_args()

    stats = collections.Counter()
    before_iv = collections.Counter(); before_cd = collections.Counter()
    cases = []
    for line in open(a.inp, encoding="utf-8"):
        c = json.loads(line)
        before_iv.update(i.get("name") for i in c.get("interventions", []))
        before_cd.update(d.get("name") for d in c.get("causal_drugs", []))
        c = clean(c, stats)
        c["year"] = datetime.datetime.fromtimestamp(int(c["created_utc"]), datetime.UTC).year if c.get("created_utc") else None
        cases.append(c)

    with open(a.out, "w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c) + "\n")

    after_iv = collections.Counter(i["name"] for c in cases for i in c["interventions"])
    after_cd = collections.Counter(d["name"] for c in cases for d in c["causal_drugs"])
    print(f"cases: {len(cases)} -> {a.out}")
    print(f"treatment names: {len(before_iv)} -> {len(after_iv)}   causal drug names: {len(before_cd)} -> {len(after_cd)}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    if a.csv:
        # the kind/created_utc of each record live in the candidates file; join them in if present
        cand = Path(a.inp).with_name("candidates.jsonl")
        if cand.exists():
            kind = {json.loads(l)["id"]: json.loads(l)["kind"] for l in open(cand, encoding="utf-8")}
            for c in cases:
                c["kind"] = kind.get(c["id"])
        fp = [c for c in cases if c["is_first_person_case"]]
        export_csv(fp, Path(a.csv))
        print(f"CSV export ({len(fp)} first-person cases, no quotes, no Reddit ids) -> {a.csv}/")
