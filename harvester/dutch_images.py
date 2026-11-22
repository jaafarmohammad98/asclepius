"""DUTCH values from screenshots posted on r/DrWillPowers (plan issue M3-15).

Turns public DUTCH report screenshots into the extractor's lab-report CSV, which the analysis loads
with `load_lab_reports`. Run from harvester/, after `topic_filter.py` and the case extraction:

    .venv/bin/python dutch_images.py fetch    --raw data/DrWillPowers_raw.jsonl data/PSSD_raw.jsonl
    ANTHROPIC_API_KEY=... .venv/bin/python dutch_images.py extract
    ANTHROPIC_API_KEY=... .venv/bin/python dutch_images.py hormones
    .venv/bin/python dutch_images.py export   --cases data/cases_merged_clean.jsonl
    .venv/bin/python dutch_images.py review   --cases data/cases_merged_clean.jsonl
    .venv/bin/python dutch_images.py score    # after filling in data/dutch_review.csv

`fetch` finds posts and comments that mention DUTCH and carry images, and downloads the images one
at a time (resumable, dead links logged). `extract` makes one vision call per image and caches it by
the image's SHA-256, so a re-run never pays twice. `hormones` reads from each report's post text
whether the author was on hormones when tested. `export` writes CSVs with no Reddit ids, URLs,
quotes, image paths or author keys. `review` writes a local page for hand-checking 20 reports, and
`score` turns the filled-in check sheet into a per-marker error rate.

Images can show a name, date of birth and accession number. They stay in data/images/ (git-ignored)
and only the printed result tables are read: no field in the schema can hold an identity.
"""
import argparse, base64, csv, hashlib, html, json, random, re, sys, time, urllib.error, urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from extract_cases import DATA, author_key, iter_ndjson, load_salt, quote_in

MODEL = "claude-haiku-5-5"
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}  # $ / MTok in, out
ITEMS = DATA / "dutch_items.jsonl"        # posts and comments with DUTCH images
IMAGES = DATA / "dutch_images.jsonl"      # one line per download attempt: url -> sha256 or error
IMAGE_DIR = DATA / "images"
PAGES = DATA / "dutch_pages.jsonl"        # vision results, keyed by (sha256, model)
HORMONES = DATA / "dutch_hormones.jsonl"  # hormone step results, keyed by (item id, model)
EXPORT = DATA / "dutch_export"
SAMPLES = Path(__file__).resolve().parent.parent / "samples"

LABELS = ("dutch_page", "other_lab", "not_lab")
SEXES = ("male", "female")
COMPARATORS = ("<", ">")
MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
# the analysis loader's unit pattern (analysis/asclepius/dataset.py UNIT, from extractor/src/parse.ts)
UNIT = re.compile(r"^[a-zµμ]+/[a-zµμ]+$", re.I)
CONDITIONS = {"PSSD": "PSSD", "PFS": "PFS", "PSA": "PAS"}  # case condition -> analysis group
COLUMNS = ("record_id", "sex", "age_years", "collection_month", "report_month", "marker_id", "marker_name",
           "category", "comparator", "value", "unit", "ref_low", "ref_high", "status")
META_COLUMNS = ("record_id", "group", "tier", "drug")


def marker_id(name):
    """The extractor's `slug` (extractor/src/parse.ts), so ids match PDF-extracted reports."""
    return re.sub(r"^_+|_+$", "", re.sub(r"[^a-z0-9]+", "_", name.lower()))


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf") else None


# ---------------------------------------------------------------- page parsing

def parse_page(raw):
    """Check one vision result against the schema. Returns (page, problems): fields that do not fit are
    dropped and named in `problems`, so a misread never reaches the export silently."""
    problems = []
    label = raw.get("label")
    if label not in LABELS:
        problems.append(f"unknown label {label!r}")
        label = "not_lab"
    page = {"label": label, "report_month": None, "collection_month": None, "age_years": None, "sex": None,
            "markers": [], "dropped": []}
    if label != "dutch_page":
        return page, problems
    for key in ("report_month", "collection_month"):
        v = raw.get(key)
        if isinstance(v, str) and MONTH.match(v):
            page[key] = v
        elif v not in (None, ""):
            problems.append(f"{key} {v!r} is not YYYY-MM")
    age = raw.get("age_years")
    if isinstance(age, int) and not isinstance(age, bool) and 0 < age < 130:
        page["age_years"] = age
    elif age is not None:
        problems.append(f"age {age!r} is not an age in years")
    if raw.get("sex") in SEXES:
        page["sex"] = raw["sex"]
    for m in raw.get("markers") or []:
        if not isinstance(m, dict) or not isinstance(m.get("name"), str) or not m["name"].strip() \
                or _num(m.get("value")) is None:
            problems.append(f"unreadable marker row {str(m)[:80]!r}")
            continue
        name = re.sub(r"\s+", " ", m["name"]).strip()
        if str(m.get("unit") or "").strip().lower() == "ratio":
            # derived "Metabolite Ratios" rows: the PDF extractor's unit pattern never reads them either
            page["dropped"].append(name)
            continue
        page["markers"].append({
            "id": marker_id(name), "name": name, "value": float(m["value"]),
            "comparator": m.get("comparator") if m.get("comparator") in COMPARATORS else None,
            "unit": str(m.get("unit") or "").strip(),
            "ref_low": _num(m.get("ref_low")), "ref_high": _num(m.get("ref_high")),
            "status": (m.get("status") or "").strip() or None,
            "category": (m.get("category") or "").strip() or None,
        })
    return page, problems


# ---------------------------------------------------------------- checks

def known_markers(samples=SAMPLES):
    """marker id -> unit, for the markers the PDF extractor reads from the sample reports (committed
    expected.json files, plus local extractor output when present)."""
    known = {}
    for f in sorted(samples.glob("*/expected.json")) + sorted(samples.glob("*/out.json")):
        for m in json.loads(f.read_text()).get("markers", []):
            if m.get("id") and m.get("unit"):
                known.setdefault(m["id"], m["unit"])
    if not known:
        raise SystemExit(f"no marker names found under {samples}: the known-name check needs samples/*/expected.json")
    return known


# a DUTCH report prints totals that are sums of other printed values (extractor/src/checks.ts)
SUM_RULES = [
    ("24hr free cortisol", re.compile(r"^24hr_free_cortisol$"), re.compile(r"^cortisol_[a-e]_"), 4, None),
    ("24hr free cortisone", re.compile(r"^24hr_free_cortisone$"), re.compile(r"^cortisone_[a-e]_"), 4, None),
    ("metabolized cortisol", re.compile(r"^metabolized_cortisol"),
     re.compile(r"^[ab]_tetrahydrocortiso(l|ne)_"), 3, 3),
    ("total estrogen", re.compile(r"^total_estrogen$"), re.compile(
        r"^(estrone_e1|estradiol_e2|estriol_e3|2_oh_e1|4_oh_e1|16_oh_e1|2_methoxy_e1|2_oh_e2|4_oh_e2|2_methoxy_e2)$"),
     10, 10),
]


def check_markers(markers, known):
    """Return one failure message per broken rule: status vs value and range, unit pattern, known name,
    and the report's printed sums."""
    failures = []
    for m in markers:
        if m["id"] not in known:
            failures.append(f"{m['name']}: not a marker name the PDF extractor knows")
        if not UNIT.match(m["unit"]):
            failures.append(f"{m['name']}: unit {m['unit']!r} is not written like ng/mg")
        lo, hi, s = m["ref_low"], m["ref_high"], (m["status"] or "").lower()
        if s and lo is not None and hi is not None and not m["comparator"]:
            # the lab compares unrounded values, so a printed value may equal the limit
            ok = (m["value"] >= hi if s.startswith("above") else m["value"] <= lo if s.startswith("below")
                  else lo <= m["value"] <= hi)
            if not ok:
                failures.append(f"{m['name']}: {m['value']} is marked {m['status']!r} against {lo}-{hi}")
    for name, total_re, part_re, min_parts, count in SUM_RULES:
        total = next((m for m in markers if total_re.search(m["id"])), None)
        parts = [m for m in markers if part_re.search(m["id"])]
        if not total or len(parts) < min_parts or (count and len(parts) != count):
            continue
        s = sum(m["value"] for m in parts)
        if abs(s - total["value"]) > max(0.06 * len(parts), abs(total["value"]) * 0.005):
            failures.append(f"{name}: components add to {round(s, 2)}, report prints {total['value']}")
    return failures


# ---------------------------------------------------------------- reports

def month_of(utc):
    return datetime.fromtimestamp(int(utc), timezone.utc).strftime("%Y-%m")


def author_cases(cases):
    """author_key -> (group, drug, sex) from the author's earliest first-person PSSD / PFS / PAS case.
    The drug is that case's first causal drug."""
    by_author = {}
    for c in sorted(cases, key=lambda c: int(c.get("created_utc") or 0)):
        k = c.get("author_key")
        if not k or k in by_author or not c.get("is_first_person_case") or c.get("condition") not in CONDITIONS:
            continue
        drugs = [x.get("name") for x in c.get("causal_drugs") or [] if isinstance(x, dict) and x.get("name")]
        by_author[k] = (CONDITIONS[c["condition"]], drugs[0] if drugs else None, c.get("sex"))
    return by_author


def _head(pages, key, warnings):
    """A report-level field (month, age, sex) from its pages; the smallest value when pages disagree."""
    vals = {p[key] for p in pages if p[key] is not None}
    if len(vals) > 1:
        warnings.append(f"pages disagree on {key} ({', '.join(map(str, sorted(vals)))}); took the smallest")
    return min(vals) if vals else None


def assemble(items, pages, hormones, cases, known):
    """One report per post or comment, from its DUTCH pages.

    `items`: dicts with id, author_key, created_utc and `images` (SHA-256 list); `pages`: sha256 -> parsed
    page (from `parse_page`, optionally with its `problems`); `hormones`: item id -> hormone answer.
    Authors without a first-person PSSD / PFS / PAS case are left out, and a repost by the same author
    with identical numbers is dropped (the earliest post is kept). Rows with a marker name the PDF
    extractor does not know, or in a unit other than the extractor's for that marker, are left out of the
    report with a warning."""
    joined = author_cases(cases)
    reports, seen = [], set()
    for it in sorted(items, key=lambda i: (int(i["created_utc"]), i["id"])):
        if it.get("author_key") not in joined:
            continue
        dutch = [pages[s] for s in it["images"] if pages.get(s) and pages[s]["label"] == "dutch_page"]
        markers, failures, warnings = {}, [], []
        for p in dutch:
            failures += [f"unreadable on the image: {x}" for x in p.get("problems", [])]
            if p.get("dropped"):
                warnings.append(f"ratio rows left out: {', '.join(p['dropped'])}")
            # newer DUTCH layouts, add-on panels and other labs' tables: rows the PDF extractor cannot read
            # yet, rows with no unit, or a known marker in another lab's unit. The rest is still checked.
            usable = [dict(m, unit=known[m["id"]]) for m in p["markers"]
                      if m["id"] in known and m["unit"].lower() == known[m["id"]].lower()]
            if len(usable) < len(p["markers"]):
                kept = {m["name"] for m in usable}
                left = [f"{m['name']} ({m['unit'] or 'no unit'})" for m in p["markers"] if m["name"] not in kept]
                warnings.append(f"left out, not a marker in a unit the PDF extractor reads: {', '.join(left)}")
            for m in usable:
                prev = markers.setdefault(m["id"], m)
                if prev is not m and (prev["value"], prev["unit"], prev["comparator"]) != (m["value"], m["unit"], m["comparator"]):
                    failures.append(f"{m['name']}: two pages give different values ({prev['value']}, {m['value']})")
        if not markers:
            continue
        signature = (it["author_key"], frozenset((m["id"], m["value"], m["comparator"]) for m in markers.values()))
        if signature in seen:
            continue
        seen.add(signature)
        group, drug, case_sex = joined[it["author_key"]]

        report_month = _head(dutch, "report_month", warnings)
        if report_month is None:
            report_month = month_of(it["created_utc"])
            warnings.append("report month not printed: used the post month")
        sex = _head(dutch, "sex", warnings) or (case_sex if case_sex in SEXES else None)
        h = hormones.get(it["id"])
        if h is None:
            failures.append("hormone step not run for this post")
            h = {"on_hormones_at_test": "unknown", "quote": None}
        marker_list = list(markers.values())
        reports.append({
            "item_id": it["id"], "author_key": it["author_key"], "created_utc": int(it["created_utc"]),
            "images": [s for s in it["images"] if pages.get(s) and pages[s]["label"] == "dutch_page"],
            "group": group, "tier": "C", "drug": drug, "sex": sex, "age_years": _head(dutch, "age_years", warnings),
            "collection_month": _head(dutch, "collection_month", warnings), "report_month": report_month,
            "on_hormones_at_test": h["on_hormones_at_test"], "hormone_quote": h.get("quote"),
            "markers": marker_list, "failures": failures + check_markers(marker_list, known), "warnings": warnings,
        })
    return reports


# ---------------------------------------------------------------- export

BUCKETS = ("default", "hormones", "failing", "retests")


def _order(r):
    return (r["report_month"], r["created_utc"], r["item_id"])


def select(reports):
    """Sort each report into one file. `default`: each author's earliest report that passes every
    check and was not taken on hormones. `hormones`: on hormones when tested. `failing`: a check
    failed. `retests`: an author's later clean reports."""
    buckets = {b: [] for b in BUCKETS}
    authors = set()
    for r in sorted(reports, key=_order):
        if r["on_hormones_at_test"] == "yes":
            buckets["hormones"].append(r)
        elif r["failures"]:
            buckets["failing"].append(r)
        elif r["author_key"] in authors:
            buckets["retests"].append(r)
        else:
            authors.add(r["author_key"])
            buckets["default"].append(r)
    return buckets


def _cell(v):
    if v is None:
        return ""
    return f"{v:g}" if isinstance(v, float) else str(v)


def write_export(buckets, out_dir):
    """Write `dutch_<bucket>.csv` (extractor columns) and `dutch_<bucket>_metadata.csv` for each
    non-empty bucket. Record ids run rd001... across all files, in time order within each file, and
    carry no link to the post: no Reddit id, URL, quote, image or author key is written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("dutch_*.csv"):
        old.unlink()
    n = 0
    written = {}
    for bucket in BUCKETS:
        if not buckets[bucket]:
            continue
        with open(out_dir / f"dutch_{bucket}.csv", "w", newline="", encoding="utf-8") as fh, \
                open(out_dir / f"dutch_{bucket}_metadata.csv", "w", newline="", encoding="utf-8") as mh:
            w, mw = csv.writer(fh), csv.writer(mh)
            w.writerow(COLUMNS); mw.writerow(META_COLUMNS)
            for r in buckets[bucket]:
                n += 1
                rid = f"rd{n:03d}"
                for m in r["markers"]:
                    w.writerow([_cell(v) for v in (rid, r["sex"], r["age_years"], r["collection_month"],
                                r["report_month"], m["id"], m["name"], m["category"], m["comparator"],
                                m["value"], m["unit"], m["ref_low"], m["ref_high"], m["status"])])
                mw.writerow([rid, r["group"], r["tier"], r["drug"] or ""])
        written[bucket] = len(buckets[bucket])
    return written


# ---------------------------------------------------------------- fetch

DUTCH = re.compile(r"\bdutch\b", re.I)
LINK = re.compile(r"https?://(?:i\.redd\.it|preview\.redd\.it|(?:i\.|m\.)?imgur\.com)/[^\s)\]\"'<>]+", re.I)
IMGUR_PAGE = re.compile(r"^https?://(?:m\.)?imgur\.com/(\w+)$", re.I)
UA = "Mozilla/5.0"  # generic: the request names no project or person


def _text(o):
    return o.get("body") if "body" in o else f"{o.get('title', '')}\n{o.get('selftext', '')}"


def image_urls(o):
    """Image URLs of a raw post or comment, in order: uploaded gallery / inline images (full size), a
    direct image link post, then image links in the text. An imgur page link becomes its image."""
    urls = []
    meta = o.get("media_metadata") or {}
    order = [i.get("media_id") for i in (o.get("gallery_data") or {}).get("items") or []]
    for k in order + [k for k in meta if k not in order]:
        v = meta.get(k)
        if isinstance(v, dict) and v.get("e") == "Image" and (v.get("s") or {}).get("u"):
            urls.append(html.unescape(v["s"]["u"]))
    if "body" not in o and re.search(r"//(i\.redd\.it|i\.imgur\.com)/", o.get("url") or ""):
        urls.append(o["url"])
    for u in LINK.findall(html.unescape(_text(o) or "")):
        u = u.rstrip(".,;")
        page = IMGUR_PAGE.match(u)
        urls.append(f"https://i.imgur.com/{page.group(1)}.jpg" if page else u)
    return list(dict.fromkeys(urls))


def dutch_item(o, salt):
    """A post or comment that names DUTCH and carries images, pseudonymised like `extract_cases load`."""
    key = author_key(o.get("author"), salt)
    if not key or not DUTCH.search(_text(o) or ""):
        return None
    urls = image_urls(o)
    if not urls:
        return None
    return {"id": o.get("name") or o.get("id"), "kind": "comment" if "body" in o else "submission",
            "post_id": o.get("link_id") if "body" in o else o.get("name") or o.get("id"),
            "created_utc": int(o.get("created_utc") or 0), "author_key": key, "text": (_text(o) or "").strip(),
            "urls": urls}


MAGIC = {b"\x89PNG": ("image/png", "png"), b"\xff\xd8\xff": ("image/jpeg", "jpg"), b"GIF8": ("image/gif", "gif"),
         b"RIFF": ("image/webp", "webp")}


def image_type(data):
    return next((t for sig, t in MAGIC.items() if data.startswith(sig)), (None, None))


def download(url):
    """-> (status, bytes). status is "ok", "dead" (gone for good) or "error" (retry on the next run)."""
    if re.search(r"imgur\.com/(a|gallery)/", url):
        return "dead", "imgur album: not supported"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            final, data = r.geturl(), r.read()
    except urllib.error.HTTPError as e:
        return ("dead" if e.code in (403, 404, 410) else "error"), f"HTTP {e.code}"
    except Exception as e:
        return "error", type(e).__name__
    if "removed" in final:  # imgur redirects a deleted image to removed.png
        return "dead", "removed"
    if image_type(data)[0] is None:
        return "dead", "not an image"
    return "ok", data


def read_jsonl(path):
    return list(iter_ndjson(path)) if path.exists() else []


def cmd_fetch(a):
    salt = load_salt()
    items = [it for raw in a.raw for o in iter_ndjson(Path(raw)) if (it := dutch_item(o, salt))]
    # a comment's thread post often says whether the author was on hormones: keep it when it is theirs
    wanted = {it["post_id"] for it in items if it["kind"] == "comment"}
    parents = {}
    for raw in a.raw if wanted else []:
        for o in iter_ndjson(Path(raw)):
            name = o.get("name") or f"t3_{o.get('id')}"
            if "body" not in o and name in wanted:
                parents[name] = (author_key(o.get("author"), salt), _text(o).strip())
    for it in items:
        key, text = parents.get(it["post_id"], (None, None)) if it["kind"] == "comment" else (None, None)
        it["parent_text"] = text if key and key == it["author_key"] else None
    with open(ITEMS, "w", encoding="utf-8") as out:
        for it in items:
            out.write(json.dumps(it) + "\n")
    urls = list(dict.fromkeys(u for it in items for u in it["urls"]))
    print(f"{len(items)} posts/comments from {len({i['author_key'] for i in items})} authors, {len(urls)} image URLs")
    done = {r["url"] for r in read_jsonl(IMAGES) if r["status"] in ("ok", "dead")}
    todo = [u for u in urls if u not in done]
    print(f"{len(todo)} to download ({len(done)} already done)")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with open(IMAGES, "a", encoding="utf-8") as log:
        for n, url in enumerate(todo, 1):
            status, data = download(url)
            row = {"url": url, "status": status}
            if status == "ok":
                sha = hashlib.sha256(data).hexdigest()
                path = IMAGE_DIR / f"{sha}.{image_type(data)[1]}"
                if not path.exists():
                    path.write_bytes(data)
                row.update(sha256=sha, file=path.name, bytes=len(data))
            else:
                row["error"] = data
            log.write(json.dumps(row) + "\n"); log.flush()
            counts[status] += 1
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} {dict(counts)}")
            time.sleep(a.delay)
    print(f"done: {dict(counts)} -> {IMAGE_DIR}  (dead links are in {IMAGES.name})")


def image_files():
    """sha256 -> file name in IMAGE_DIR, for every downloaded image"""
    return {r["sha256"]: r["file"] for r in image_index().values() if r["status"] == "ok"}


def image_index():
    """url -> latest download row"""
    rows = {}
    for r in read_jsonl(IMAGES):
        rows[r["url"]] = r
    return rows


# ---------------------------------------------------------------- extract (vision) and hormones

NULLABLE_NUM = {"type": ["number", "null"]}
NULLABLE_STR = {"type": ["string", "null"]}
PAGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["label", "report_month", "collection_month", "age_years", "sex", "markers"],
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "report_month": NULLABLE_STR, "collection_month": NULLABLE_STR,
        "age_years": {"type": ["integer", "null"]},
        "sex": {"type": "string", "enum": ["male", "female", "not_printed"]},
        "markers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "value", "comparator", "unit", "ref_low", "ref_high", "status", "category"],
            "properties": {"name": {"type": "string"}, "value": {"type": "number"},
                           "comparator": {"type": "string", "enum": ["", "<", ">"]}, "unit": {"type": "string"},
                           "ref_low": NULLABLE_NUM, "ref_high": NULLABLE_NUM, "status": NULLABLE_STR,
                           "category": NULLABLE_STR}}},
    },
}

PAGE_PROMPT = """This image was posted on a patient forum. Decide what it shows and, only if it is a page of a
DUTCH hormone test report (Precision Analytical), copy its printed result tables.

label: "dutch_page" for any page of a DUTCH report (including pages with only dials or graphs), "other_lab"
for any other lab result (blood tests, other urine or saliva tests), "not_lab" for anything else.

For a dutch_page only:
- markers: one entry per printed table row that has a number. Copy the test name, result, unit, reference range
  (ref_low, ref_high) and the status words next to it (e.g. "Within range", "Below range") exactly as printed.
  comparator is "<" or ">" only when the result is printed with one, else "". category is the table heading,
  e.g. "Androgens and Metabolites". If a range is printed as "<5", ref_low is 0 and ref_high is 5.
- Read numbers from the tables only. Ignore dials, gauges, bar graphs, charts and text commentary: a page
  with no result table has an empty markers list.
- If a row's test is one of these names, write the name exactly like this: {names}
- Skip any row you cannot read with certainty. Never guess a digit.
- report_month and collection_month: YYYY-MM if printed, else null. age_years: if printed, else null.
  sex: if printed, else "not_printed".
- Never copy the patient's name, date of birth, accession or order number, exact dates or times, the
  practitioner, or any other identifying text. There is no field for them.
For other labels, give an empty markers list and nulls."""

HORMONE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["on_hormones_at_test", "quote"],
    "properties": {"on_hormones_at_test": {"type": "string", "enum": ["yes", "no", "unknown"]},
                   "quote": NULLABLE_STR},
}
HORMONE_PROMPT = """A patient posted DUTCH hormone test results with this text. Was the author taking hormone
therapy or a hormone-changing drug when the test was collected? That includes testosterone, estrogen,
progesterone, DHEA, pregnenolone, hCG, clomiphene or enclomiphene, aromatase inhibitors, anti-androgens,
5-alpha-reductase inhibitors (finasteride, dutasteride), thyroid hormone and glucocorticoids such as hydrocortisone.

on_hormones_at_test: "yes" if the text says so, "no" if it says they were not, "unknown" if it does not say.
quote: a short exact quote from the text that supports the answer, or null for "unknown".

{parent}<post>
{text}
</post>"""


def known_names(samples=SAMPLES):
    names = {}
    for f in sorted(samples.glob("*/out.json")):
        for m in json.loads(f.read_text()).get("markers", []):
            names.setdefault(m["id"], m["name"])
    return sorted(names.values()) or sorted(known_markers(samples))


def ask(client, model, content, schema, label):
    """One structured-output call with retries on transient errors. -> (json or None, error, usage)"""
    import anthropic
    for attempt in range(6):
        try:
            resp = client.messages.create(model=model, max_tokens=16000,
                                          output_config={"format": {"type": "json_schema", "schema": schema}},
                                          messages=[{"role": "user", "content": content}])
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            wait = 2 ** attempt
            print(f"  {label}: {type(e).__name__}, retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
        if resp.stop_reason in ("max_tokens", "refusal"):
            return None, f"stop_reason={resp.stop_reason}", usage
        text = next((b.text for b in resp.content if b.type == "text"), None)
        try:
            return json.loads(text), None, usage
        except (TypeError, json.JSONDecodeError):
            return None, "no JSON in the response", usage
    return None, None, None  # transient: not cached, so the next run retries


def run_calls(jobs, call, out_path, workers, model):
    """Run `call(job)` for each job on a thread pool; write each finished row to `out_path` (append)."""
    from concurrent.futures import ThreadPoolExecutor
    tin = tout = n = 0
    with open(out_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(workers) as pool:
        for row in pool.map(call, jobs):
            if row is None:
                continue
            if row.get("usage"):
                tin += row["usage"]["in"]; tout += row["usage"]["out"]
            out.write(json.dumps(row) + "\n"); out.flush()
            n += 1
            if n % 25 == 0:
                print(f"  {n}/{len(jobs)}")
    pin, pout = PRICES.get(model, (0, 0))
    print(f"done: {n} new -> {out_path.name}  ({tin} in / {tout} out tokens, ~${tin / 1e6 * pin + tout / 1e6 * pout:.2f})")


MAX_IMAGE_BYTES = 5 * 1024 * 1024  # API limit for one base64 image


def cmd_extract(a):
    import anthropic
    client = anthropic.Anthropic()
    done = {r["sha256"] for r in read_jsonl(PAGES) if r["model"] == a.model}
    files = image_files()
    todo = [s for s in files if s not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} images to read with {a.model} ({len(done)} cached)")
    prompt = PAGE_PROMPT.format(names="; ".join(known_names()))

    def call(sha):
        data = (IMAGE_DIR / files[sha]).read_bytes()
        if len(data) > MAX_IMAGE_BYTES:
            return {"sha256": sha, "model": a.model, "raw": None, "error": "image over 5 MB", "usage": None}
        content = [{"type": "image", "source": {"type": "base64", "media_type": image_type(data)[0],
                                                 "data": base64.standard_b64encode(data).decode()}},
                   {"type": "text", "text": prompt}]
        raw, error, usage = ask(client, a.model, content, PAGE_SCHEMA, sha[:12])
        if raw is None and error is None:
            return None
        return {"sha256": sha, "model": a.model, "raw": raw, "error": error, "usage": usage}

    run_calls(todo, call, PAGES, a.workers, a.model)


def load_pages(model):
    """sha256 -> parsed page (with `problems`) for one model; failed calls count as unreadable pages."""
    pages = {}
    for r in read_jsonl(PAGES):
        if r["model"] != model:
            continue
        if r["raw"] is None:
            pages[r["sha256"]] = {"label": "not_lab", "markers": [], "problems": [r["error"]]}
            continue
        page, problems = parse_page(r["raw"])
        page["problems"] = problems
        pages[r["sha256"]] = page
    return pages


def load_items():
    """Items with their URLs resolved to downloaded image hashes, in URL order."""
    index = image_index()
    items = read_jsonl(ITEMS)
    for it in items:
        it["images"] = [index[u]["sha256"] for u in it["urls"] if index.get(u, {}).get("status") == "ok"]
    return items


def cmd_hormones(a):
    import anthropic
    client = anthropic.Anthropic()
    pages = load_pages(a.vision_model)
    done = {r["id"] for r in read_jsonl(HORMONES) if r["model"] == a.model}
    todo = [it for it in load_items() if it["id"] not in done
            and any(pages.get(s, {}).get("label") == "dutch_page" for s in it["images"])]
    print(f"{len(todo)} posts to read for hormone use with {a.model} ({len(done)} cached)")

    def call(it):
        parent = (f"The thread this comment answers, also written by the author:\n<thread>\n"
                  f"{it['parent_text'][:8000]}\n</thread>\n\n" if it.get("parent_text") else "")
        raw, error, usage = ask(client, a.model, HORMONE_PROMPT.format(parent=parent, text=it["text"][:12000]),
                                HORMONE_SCHEMA, it["id"])
        if raw is None and error is None:
            return None
        if raw is None:  # an unanswered post stays "unknown"
            raw = {"on_hormones_at_test": "unknown", "quote": None}
        quote = raw.get("quote") or None
        return {"id": it["id"], "model": a.model, "on_hormones_at_test": raw["on_hormones_at_test"],
                "quote": quote, "quote_ok": bool(quote) and quote_in(quote, f"{it['text']}\n{it.get('parent_text') or ''}"),
                "error": error,
                "usage": usage}

    run_calls(todo, call, HORMONES, a.workers, a.model)


def build_reports(a):
    hormones = {r["id"]: r for r in read_jsonl(HORMONES) if r["model"] == a.hormone_model}
    for h in hormones.values():
        # a "yes" stands even when its quote is not found verbatim: wrongly keeping a hormone user in is worse
        if h["on_hormones_at_test"] == "no" and not h["quote_ok"]:
            h["on_hormones_at_test"] = "unknown"
    return assemble(load_items(), load_pages(a.model), hormones, list(iter_ndjson(Path(a.cases))), known_markers())


def cmd_export(a):
    reports = build_reports(a)
    buckets = select(reports)
    written = write_export(buckets, Path(a.out))
    print(f"{len(reports)} reports from {len({r['author_key'] for r in reports})} authors -> {a.out}")
    for b in BUCKETS:
        print(f"  {b}: {written.get(b, 0)} reports")
    reasons = defaultdict(int)
    for r in buckets["failing"]:
        for f in r["failures"]:
            reasons[re.sub(r"\d+(\.\d+)?", "#", re.sub(r"^[^:]*: ", "", f))[:70]] += 1
    for f, n in sorted(reasons.items(), key=lambda x: -x[1])[:15]:
        print(f"  failure x{n}: {f}")
    warned = sum(1 for r in reports if r["warnings"])
    if warned:
        print(f"  {warned} reports have warnings (e.g. report month taken from the post)")


# ---------------------------------------------------------------- review and score

REVIEW_COLUMNS = ("review_id", "marker_id", "marker_name", "comparator", "value", "unit", "ref_low", "ref_high",
                  "status", "correct", "note")


def cmd_review(a):
    reports = build_reports(a)
    sample = random.Random(a.seed).sample(reports, min(a.n, len(reports)))
    files = image_files()
    e = html.escape
    parts = ["<!doctype html><meta charset=utf-8><title>DUTCH review</title><style>body{font:14px sans-serif;"
             "margin:16px}section{border-top:2px solid #888;padding:12px 0}img{max-width:100%;border:1px solid #ccc}"
             ".cols{display:grid;grid-template-columns:3fr 2fr;gap:16px}table{border-collapse:collapse}"
             "td,th{border:1px solid #ccc;padding:2px 6px;text-align:left}.bad{color:#b00}</style>",
             "<h1>DUTCH values read from Reddit images</h1><p>Local only: these images can identify people. "
             f"Mark each row in <code>{e(str(Path(a.sheet).name))}</code>: correct = y or n, and add a row with "
             "correct = missed for any table value on the image that is not listed.</p>"]
    with open(a.sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(REVIEW_COLUMNS)
        for i, r in enumerate(sample, 1):
            rid = f"rv{i:02d}"
            imgs = "".join(f'<img src="images/{e(files[s])}" loading=lazy>' for s in r["images"] if s in files)
            rows = "".join(f"<tr><td>{e(m['name'])}</td><td>{e(m['comparator'] or '')}{m['value']:g}</td>"
                           f"<td>{e(m['unit'])}</td><td>{_cell(m['ref_low'])}-{_cell(m['ref_high'])}</td>"
                           f"<td>{e(m['status'] or '')}</td></tr>" for m in r["markers"])
            notes = "".join(f"<li class=bad>{e(f)}</li>" for f in r["failures"]) + \
                "".join(f"<li>{e(x)}</li>" for x in r["warnings"])
            parts.append(f"<section><h2>{rid}</h2><p>{e(r['group'])}, report {e(r['report_month'])}, "
                         f"sex {e(r['sex'] or '?')}, age {e(_cell(r['age_years']) or '?')}. On hormones at test: "
                         f"<b>{e(r['on_hormones_at_test'])}</b> {e(repr(r['hormone_quote'] or ''))}</p>"
                         f"<ul>{notes}</ul><div class=cols><div>{imgs}</div><table><tr><th>marker<th>value<th>unit"
                         f"<th>range<th>status</tr>{rows}</table></div></section>")
            for m in r["markers"]:
                w.writerow([rid, m["id"], m["name"], m["comparator"] or "", _cell(m["value"]), m["unit"],
                            _cell(m["ref_low"]), _cell(m["ref_high"]), m["status"] or "", "", ""])
    Path(a.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"{len(sample)} reports -> {a.out} (open in a browser), check sheet -> {a.sheet}")


def score(rows):
    """Per-marker error rate from a filled check sheet: (n wrong + n missed) / n checked."""
    marks = [(r.get("correct") or "").strip().lower() for r in rows]
    right, wrong, missed = marks.count("y"), marks.count("n"), marks.count("missed")
    checked = right + wrong + missed
    return {"checked": checked, "wrong": wrong, "missed": missed, "unmarked": len(marks) - checked,
            "error_rate": (wrong + missed) / checked if checked else None}


def cmd_score(a):
    with open(a.sheet, newline="", encoding="utf-8") as fh:
        s = score(list(csv.DictReader(fh)))
    if s["error_rate"] is None:
        raise SystemExit("no rows marked yet: fill the correct column with y, n or missed")
    print(f"{s['checked']} markers checked: {s['wrong']} wrong, {s['missed']} missed -> per-marker error rate "
          f"{s['error_rate']:.1%}" + (f"  ({s['unmarked']} rows still unmarked)" if s["unmarked"] else ""))
    if s["error_rate"] > 0.02:
        print("over 2%: re-run extract and export with --model claude-sonnet-5-5")


# ---------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("fetch"); x.add_argument("--raw", nargs="+", required=True); x.add_argument("--delay", type=float, default=1.0)
    x = s.add_parser("extract"); x.add_argument("--model", default=MODEL); x.add_argument("--limit", type=int, default=0)
    x.add_argument("--workers", type=int, default=4)
    x = s.add_parser("hormones"); x.add_argument("--model", default=MODEL); x.add_argument("--vision-model", default=MODEL)
    x.add_argument("--workers", type=int, default=4)
    for name in ("export", "review"):
        x = s.add_parser(name)
        x.add_argument("--cases", default=str(DATA / "cases_merged_clean.jsonl"))
        x.add_argument("--model", default=MODEL, help="vision model whose readings to use")
        x.add_argument("--hormone-model", default=MODEL)
    s.choices["export"].add_argument("--out", default=str(EXPORT))
    r = s.choices["review"]
    r.add_argument("--out", default=str(DATA / "dutch_review.html"))
    r.add_argument("--sheet", default=str(DATA / "dutch_review.csv"))
    r.add_argument("--n", type=int, default=20); r.add_argument("--seed", type=int, default=0)
    x = s.add_parser("score"); x.add_argument("--sheet", default=str(DATA / "dutch_review.csv"))
    a = p.parse_args()
    {"fetch": cmd_fetch, "extract": cmd_extract, "hormones": cmd_hormones, "export": cmd_export,
     "review": cmd_review, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
