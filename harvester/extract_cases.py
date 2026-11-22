"""
Turn PSSD / PFS / PSA Reddit posts into structured case records with Claude.

Pipeline:
  0. fetch  - download a subreddit's posts + comments from the Arctic Shift archive API (free, no key)
  1. load   - normalise + pseudonymise into one .jsonl (also accepts Reddit dump .zst files)
  2. filter - keep only texts likely to contain case data (cheap keyword pass)
  3. extract- send each text to Claude, force a tool call that returns schema-shaped JSON  (costs money)
  4. review - dump a random sample to CSV for hand-checking against the source text

Usage (run from harvester/):
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/python extract_cases.py fetch  --subreddit PSSD --out data/PSSD_raw.jsonl
  .venv/bin/python extract_cases.py load   --inputs data/PSSD_raw.jsonl --out data/records.jsonl
  .venv/bin/python extract_cases.py filter --inp data/records.jsonl --out data/candidates.jsonl
  export ANTHROPIC_API_KEY=...
  .venv/bin/python extract_cases.py extract --inp data/candidates.jsonl --out data/cases.jsonl --limit 50
  .venv/bin/python extract_cases.py review --inp data/cases.jsonl --raw data/candidates.jsonl --out data/review.csv --n 200

Everything under data/ is real people's health posts: it is git-ignored and must stay local.
"""

import argparse, csv, hashlib, hmac, io, json, os, random, re, secrets, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

MODEL = "claude-haiku-5-5"   # cheap + good enough for extraction; spot-check vs a larger model
MAX_CHARS = 12000            # truncate very long posts
DATA = Path(__file__).parent / "data"

# ---------------------------------------------------------------- 0. fetch

ARCTIC = "https://arctic-shift.photon-reddit.com/api/{kind}/search"

def cmd_fetch(a):
    """Page through Arctic Shift oldest-first. Resumable: restarts after the last created_utc in --out."""
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    after = {"posts": 0, "comments": 0}
    seen = set()
    if out_path.exists():
        for obj in iter_ndjson(out_path):
            k = "comments" if "body" in obj else "posts"
            after[k] = max(after[k], int(obj["created_utc"]))
            seen.add(obj["id"])
    with open(out_path, "a", encoding="utf-8") as out:
        for kind in ("posts", "comments"):
            n = 0
            while True:
                params = {"subreddit": a.subreddit, "limit": 100, "sort": "asc"}
                if after[kind]:  # the API rejects after=0
                    params["after"] = after[kind]
                req = urllib.request.Request(f"{ARCTIC.format(kind=kind)}?{urllib.parse.urlencode(params)}",
                                             headers={"User-Agent": "asclepius-harvester/0.1"})
                for attempt in range(8):
                    try:
                        with urllib.request.urlopen(req, timeout=60) as r:
                            batch = json.load(r).get("data") or []
                        break
                    except urllib.error.HTTPError as e:
                        # 429 and 422 ("Timeout. Maybe slow down a bit") are the server asking us to back off
                        if 400 <= e.code < 500 and e.code not in (422, 429):
                            sys.exit(f"{kind}: HTTP {e.code}: {e.read()[:300]!r}")
                        wait = 5 * 2 ** attempt
                        print(f"  {kind}: HTTP {e.code}, retry in {wait}s", file=sys.stderr)
                        time.sleep(wait)
                    except Exception as e:
                        wait = 2 ** attempt
                        print(f"  {kind}: {type(e).__name__}, retry in {wait}s", file=sys.stderr)
                        time.sleep(wait)
                else:
                    sys.exit(f"giving up on {kind} after={after[kind]}; rerun to resume")
                if not batch:
                    break
                fresh = 0
                for obj in batch:
                    if obj["id"] not in seen:
                        seen.add(obj["id"])
                        out.write(json.dumps(obj) + "\n"); n += 1; fresh += 1
                newest = max(int(o["created_utc"]) for o in batch)
                # several items can share one second: re-ask from just before the newest one (dupes are
                # dropped via `seen`), but always move forward when a page brought nothing new
                after[kind] = newest - 1 if fresh and newest - 1 > after[kind] else newest
                if n and n % 2000 < 100:
                    print(f"  {kind}: {n} so far")
                time.sleep(a.delay)
            print(f"{kind}: {n} new")
    print(f"-> {out_path}")

# ---------------------------------------------------------------- 1. load

def iter_ndjson(path: Path):
    if path.suffix == ".zst":
        import zstandard
        with open(path, "rb") as fh:
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
            for line in io.TextIOWrapper(reader, encoding="utf-8", errors="ignore"):
                if line.strip():
                    yield json.loads(line)
    else:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

def load_salt() -> bytes:
    """Secret per-install salt. An unsalted hash of a username is reversible by hashing the
    subreddit's public author list, so the key would not actually de-identify anyone."""
    p = DATA / ".author_salt"
    if not p.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_hex(32))
        os.chmod(p, 0o600)
    return bytes.fromhex(p.read_text().strip())

GONE = {"", "[deleted]", "[removed]"}

def author_key(author, salt):
    """Stable pseudonymous key for a username, or None for a deleted account."""
    author = str(author or "")
    return None if author in ("", "[deleted]") else hmac.new(salt, author.encode(), hashlib.sha256).hexdigest()[:16]

def normalize(obj, salt):
    """Submissions have title+selftext, comments have body. Drop usernames (de-identify)."""
    if "body" in obj:
        text = (obj.get("body") or "").strip()
        if text in GONE:
            return None
    else:
        body = (obj.get("selftext") or "").strip()
        text = f"{obj.get('title','')}\n\n{'' if body in GONE else body}".strip()
    if not text:
        return None
    author = obj.get("author")
    return {
        "id": obj.get("name") or obj.get("id"),
        "subreddit": obj.get("subreddit"),
        "created_utc": obj.get("created_utc"),
        "kind": "comment" if "body" in obj else "submission",
        # stable pseudonymous author key so one person's posts can be linked, never the name itself
        "author_key": author_key(author, salt),
        "text": text,
    }

def cmd_load(a):
    salt = load_salt()
    n = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for p in a.inputs:
            for obj in iter_ndjson(Path(p)):
                rec = normalize(obj, salt)
                if rec:
                    out.write(json.dumps(rec) + "\n"); n += 1
    print(f"wrote {n} records -> {a.out}")

# ---------------------------------------------------------------- 2. filter

SIGNAL = re.compile(
    r"\b(dutch|cortisol|cortisone|testosterone|estradiol|e2|dht|shbg|prolactin|progesterone|"
    r"dhea|allopregnanolone|tsh|ng/ml|ng/dl|pg/ml|nmol/l|lab(s)?|blood ?work|results?|"
    r"(helped|worked|improved|recovered|crash(ed)?|worse)\b|mg\b|months?|years?)",
    re.I,
)

def cmd_filter(a):
    kept = total = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for rec in iter_ndjson(Path(a.inp)):
            total += 1
            t = rec["text"]
            # need some length and at least 3 signal hits to be worth an API call
            if len(t) >= 200 and len(SIGNAL.findall(t)) >= 3:
                out.write(json.dumps(rec) + "\n"); kept += 1
    print(f"kept {kept}/{total} -> {a.out}")

# ---------------------------------------------------------------- 3. extract

SYMPTOMS = [
    "genital_numbness", "low_libido", "erectile_dysfunction", "anorgasmia", "pleasureless_orgasm",
    "premature_ejaculation", "reduced_semen", "penile_shrinkage_or_curvature",
    "emotional_blunting", "anhedonia", "depression", "anxiety", "brain_fog", "memory_problems",
    "insomnia", "fatigue", "dry_skin_or_eyes", "muscle_loss", "gynecomastia", "tinnitus",
    "joint_pain", "gut_issues",
    "depersonalization_derealization", "no_morning_erections", "hair_loss", "sweating_or_hot_flashes",
    "urinary_problems", "dizziness_or_balance", "appetite_loss", "suicidal_thoughts", "other",
]
OUTCOMES = ["major_improvement", "partial_improvement", "no_effect", "worse", "unclear"]

def q(desc):  # verbatim evidence field
    return {"type": ["string", "null"], "description": f"Exact quote from the text supporting {desc}"}

CASE_TOOL = {
    "name": "record_case",
    "description": "Record structured case data extracted from one Reddit post/comment.",
    "input_schema": {
        "type": "object",
        "required": ["is_first_person_case", "condition", "causal_drugs", "symptoms", "labs", "interventions"],
        "properties": {
            "is_first_person_case": {"type": "boolean",
                "description": "True only if the author describes their OWN condition."},
            "condition": {"type": "string", "enum": ["PSSD", "PFS", "PSA", "multiple", "unclear", "none"]},
            "sex": {"type": ["string", "null"], "enum": ["male", "female", "other", None]},
            "age": {"type": ["integer", "null"]},
            "causal_drugs": {"type": "array",
                "description": "ONLY the drug(s) the author believes CAUSED their condition. Drugs taken to treat it go in interventions.",
                "items": {"type": "object", "required": ["name"], "properties": {
                "name": {"type": "string", "description": "Generic name, lowercase, e.g. sertraline, finasteride, isotretinoin. "
                         "Brand names -> generic (lexapro -> escitalopram). Only a class given -> unspecified_ssri / unspecified_snri / unspecified_antidepressant"},
                "dose_mg": {"type": ["number", "null"]},
                "duration_months": {"type": ["number", "null"]},
                "evidence": q("this drug"),
            }}},
            "onset": {"type": ["string", "null"], "enum": ["during_use", "after_stopping", "unclear", None]},
            "months_since_stopping": {"type": ["number", "null"]},
            "symptoms": {"type": "array", "items": {"type": "object", "required": ["symptom"], "properties": {
                "symptom": {"type": "string", "enum": SYMPTOMS},
                "other_description": {"type": ["string", "null"], "description": "Short plain name of the symptom, only when symptom is 'other'"},
                "severity": {"type": ["string", "null"], "enum": ["mild", "moderate", "severe", None]},
                "evidence": q("this symptom"),
            }}},
            "labs": {"type": "array", "items": {"type": "object", "required": ["test", "value"], "properties": {
                "test": {"type": "string", "description": "e.g. total_testosterone, cortisol_awakening_response"},
                "value": {"type": "number"},
                "unit": {"type": ["string", "null"]},
                "ref_low": {"type": ["number", "null"]},
                "ref_high": {"type": ["number", "null"]},
                "flag": {"type": ["string", "null"], "enum": ["low", "normal", "high", None]},
                "panel": {"type": ["string", "null"], "description": "e.g. DUTCH Complete, blood"},
                "evidence": q("this value"),
            }}},
            "interventions": {"type": "array",
                "description": "Anything the author took or did to TREAT the condition (drugs, supplements, hormones, therapies), with what it did for them.",
                "items": {"type": "object", "required": ["name", "outcome"], "properties": {
                "name": {"type": "string", "description": "Generic name, lowercase, no qualifiers, e.g. modafinil, testosterone_replacement, hcg, tms. Brand -> generic."},
                "outcome": {"type": "string", "enum": OUTCOMES},
                "duration_weeks": {"type": ["number", "null"]},
                "evidence": q("this outcome"),
            }}},
            "current_meds": {"type": "array", "items": {"type": "string"}},
            "overall_trajectory": {"type": ["string", "null"],
                "enum": ["recovered", "improving", "stable", "worsening", "unclear", None]},
        },
    },
}

SYSTEM = """You extract structured medical case data from patient forum posts about
post-SSRI sexual dysfunction (PSSD), post-finasteride syndrome (PFS) and post-accutane syndrome (PSA).

Rules:
- Record ONLY what the text explicitly states. Never infer or guess. Unknown -> null or omit.
- If the author is describing someone else, or giving general advice, set is_first_person_case=false.
- Every list item needs an 'evidence' field: a short verbatim quote from the text.
- Lab values: copy the number and unit exactly as written. Do not convert units.
- causal_drugs = what caused the condition. interventions = what they tried to fix it. Never put a treatment in causal_drugs.
- Interventions: only include if the author reports what it did for THEM.
- Symptoms: only real bodily/mental symptoms the author has. Use 'other' only if no listed symptom fits, and fill other_description.
Always respond by calling record_case."""

def extract_one(client, rec):
    import anthropic
    text = rec["text"][:MAX_CHARS]
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=4000,
                system=SYSTEM,
                tools=[CASE_TOOL],
                tool_choice={"type": "tool", "name": "record_case"},
                messages=[{"role": "user", "content": f"<post subreddit='{rec['subreddit']}'>\n{text}\n</post>"}],
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            wait = 2 ** attempt
            print(f"  {rec['id']}: {type(e).__name__}, retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        # 400/401/404 etc. are bugs or bad config: let them raise instead of silently skipping every post
        if resp.stop_reason in ("max_tokens", "refusal"):
            print(f"  {rec['id']}: stop_reason={resp.stop_reason}, skipped", file=sys.stderr)
            return None, resp.usage
        block = next((b for b in resp.content if b.type == "tool_use"), None)
        return (block.input if block else None), resp.usage
    return None, None

LIST_KEYS = ("causal_drugs", "symptoms", "labs", "interventions")

def sanitize(case):
    """The tool schema is not enforced (no strict mode), so the model occasionally returns a list
    item as a bare string or a list as something else. Drop those items and flag the record."""
    ok = True
    for key in LIST_KEYS:
        v = case.get(key)
        if v is None:
            case[key] = []
            continue
        if not isinstance(v, list):
            case[key] = []; ok = False
            continue
        good = [i for i in v if isinstance(i, dict)]
        if len(good) != len(v):
            ok = False
        case[key] = good
    if not isinstance(case.get("current_meds", []), list):
        case["current_meds"] = []; ok = False
    return case, ok

def _norm(s):
    s = s.lower().translate(str.maketrans("\u2018\u2019\u201c\u201d\u2013\u2014", "''\"\"--"))
    s = re.sub(r"[*_>`]|&amp;|&gt;|&lt;", "", s)  # markdown / html escapes in Reddit text
    return re.sub(r"\s+", " ", s).strip()

def quote_in(quote, text):
    """True when `quote` appears in `text` (ignoring case, markdown and quote styles). The model often
    joins two real quotes with "..."; each piece must appear."""
    src = _norm(text)
    return all(not p.strip() or p.strip() in src for p in re.split(r"\.\.\.|\u2026", _norm(quote)))

def evidence_ok(case, text):
    """Cheap hallucination check: every evidence quote must actually appear in the source."""
    return all(quote_in(item["evidence"], text) for key in LIST_KEYS for item in case.get(key, [])
               if item.get("evidence"))

def cmd_extract(a):
    import anthropic
    client = anthropic.Anthropic()
    done = set()
    if Path(a.out).exists():  # resumable
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8") if l.strip()}
    from concurrent.futures import ThreadPoolExecutor
    todo = [r for r in iter_ndjson(Path(a.inp)) if r["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} to extract ({len(done)} already done), {a.workers} at a time")
    n = tok_in = tok_out = 0
    with open(a.out, "a", encoding="utf-8") as out, ThreadPoolExecutor(a.workers) as pool:
        # results are written by this thread only, as they finish (order doesn't matter: keyed by id)
        for rec, (case, usage) in zip(todo, pool.map(lambda r: extract_one(client, r), todo)):
            if usage:
                tok_in += usage.input_tokens; tok_out += usage.output_tokens
            if case is None:
                continue
            case, schema_ok = sanitize(case)
            row = {"id": rec["id"], "subreddit": rec["subreddit"], "created_utc": rec["created_utc"],
                   "author_key": rec["author_key"], "schema_ok": schema_ok,
                   "evidence_ok": evidence_ok(case, rec["text"][:MAX_CHARS]), **case}
            out.write(json.dumps(row) + "\n"); out.flush()
            n += 1
            if n % 250 == 0:
                print(f"{n} extracted, {tok_in} in / {tok_out} out tokens")
    # Haiku 5.5, prompts <=100K tokens: $0.10 / $0.50 per MTok
    print(f"done: {n} new -> {a.out}  (~${tok_in / 1e6 * 0.10 + tok_out / 1e6 * 0.50:.3f})")

# ---------------------------------------------------------------- 4. review

def cmd_review(a):
    raw = {r["id"]: r["text"] for r in iter_ndjson(Path(a.raw))}
    cases = [c for c in iter_ndjson(Path(a.inp)) if c.get("is_first_person_case")]
    random.seed(0)
    sample = random.sample(cases, min(a.n, len(cases)))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "source_text", "extracted_json", "evidence_ok",
                    "causal_drugs_correct", "symptoms_correct", "labs_correct", "interventions_correct", "missed_info", "notes"])
        for c in sample:
            w.writerow([c["id"], raw.get(c["id"], ""), json.dumps(c, indent=1), c["evidence_ok"], "", "", "", "", "", ""])
    print(f"wrote {len(sample)} rows for hand review -> {a.out}")

# ---------------------------------------------------------------- cli

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("fetch");   x.add_argument("--subreddit", required=True); x.add_argument("--out", required=True); x.add_argument("--delay", type=float, default=1.0)
    x = s.add_parser("load");    x.add_argument("--inputs", nargs="+", required=True); x.add_argument("--out", required=True)
    x = s.add_parser("filter");  x.add_argument("--inp", required=True); x.add_argument("--out", required=True)
    x = s.add_parser("extract"); x.add_argument("--inp", required=True); x.add_argument("--out", required=True); x.add_argument("--limit", type=int, default=0); x.add_argument("--workers", type=int, default=8)
    x = s.add_parser("review");  x.add_argument("--inp", required=True); x.add_argument("--raw", required=True); x.add_argument("--out", required=True); x.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    {"fetch": cmd_fetch, "load": cmd_load, "filter": cmd_filter, "extract": cmd_extract, "review": cmd_review}[a.cmd](a)
