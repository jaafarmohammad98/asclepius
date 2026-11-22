# harvester

Turns public r/PSSD posts and comments into structured, quote-backed case records. This is
background data for the analysis engine. It is **not** patient-platform data: these people never
agreed to be in a research database.

## Pipeline

| Step | Script | Cost |
|---|---|---|
| `fetch` | download a subreddit from the [Arctic Shift](https://arctic-shift.photon-reddit.com) archive (resumable) | free |
| `load` | normalise, drop deleted posts, replace usernames with a salted pseudonymous key | free |
| `filter` | keyword pass that keeps likely case descriptions | free |
| `extract` | Claude Haiku 5.5 fills a fixed schema; every item carries a verbatim quote | ~$0.0005 / post |
| `review` | CSV sample for hand-checking extraction accuracy | free |
| `clean_cases.py` | merges name variants, regroups free-text symptoms, fixes enums, writes analysis CSVs | free |

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python extract_cases.py fetch  --subreddit PSSD --out data/PSSD_raw.jsonl
.venv/bin/python extract_cases.py load   --inputs data/PSSD_raw.jsonl --out data/records.jsonl
.venv/bin/python extract_cases.py filter --inp data/records.jsonl --out data/candidates.jsonl
ANTHROPIC_API_KEY=... .venv/bin/python extract_cases.py extract --inp data/candidates.jsonl --out data/cases.jsonl
.venv/bin/python extract_cases.py review --inp data/cases.jsonl --raw data/candidates.jsonl --out data/review.csv
.venv/bin/python clean_cases.py --inp data/cases.jsonl --out data/cases_clean.jsonl --csv data/export
```

`extract` is resumable: re-running skips records already in `--out`, so a run that stops (rate
limits, credit) never pays twice.

## Output

`data/export/` holds five tables linked by `case_id`: `cases.csv`, `causal_drugs.csv`,
`symptoms.csv`, `interventions.csv`, `labs.csv`. `causal_drugs` is what the author believes caused
the condition; `interventions` is what they tried to treat it, with the reported outcome.

The export leaves out Reddit ids and the verbatim quotes. A quote pasted into a search engine finds
the original post and its author.

Full r/PSSD run (Oct 2014 to Nov 2026): 408,464 posts and comments, 13,604 after filtering,
10,354 first-person cases from about 3,272 authors, 96% of records with all quotes found in the
source. Extraction cost about $7.50.

## DUTCH values from Reddit images

People on r/DrWillPowers post screenshots of their DUTCH reports. `dutch_images.py` reads the printed
result tables into the extractor's lab-report CSV, which the analysis loads with `load_lab_reports`.
Authors are joined to their cases by `author_key`, so most of them need r/DrWillPowers cases too:
download it, keep the PSSD / PFS / PAS material with `topic_filter.py` (it is mostly about hormone
therapy), and run `load`, `filter`, `extract` and `clean_cases.py` over both subreddits into
`cases_merged_clean.jsonl`, the default `--cases`. `fetch` reads the unfiltered raw files: the author
join decides who is in, and a "my DUTCH results" post often names no condition.

```sh
.venv/bin/python topic_filter.py --inp data/DrWillPowers_raw.jsonl --out data/DrWillPowers_topic_raw.jsonl
.venv/bin/python dutch_images.py fetch --raw data/DrWillPowers_raw.jsonl data/PSSD_raw.jsonl  # resumable
ANTHROPIC_API_KEY=... .venv/bin/python dutch_images.py extract                 # 1 vision call per image
ANTHROPIC_API_KEY=... .venv/bin/python dutch_images.py hormones                # 1 text call per post
.venv/bin/python dutch_images.py export   # -> data/dutch_export/
.venv/bin/python dutch_images.py review   # -> data/dutch_review.html + dutch_review.csv (20 reports)
.venv/bin/python dutch_images.py score    # per-marker error rate from the filled-in review sheet
```

| Step | What it does | Cost |
|---|---|---|
| `fetch` | finds posts and comments naming DUTCH with images (galleries, i.redd.it, preview.redd.it, imgur) and downloads them one at a time; dead links are logged in `data/dutch_images.jsonl` | free |
| `extract` | Claude Haiku 5.5 labels each image `dutch_page`, `other_lab` or `not_lab` and copies a DUTCH page's table rows, report month, collection month, age and sex. Cached by image SHA-256 and model | ~$1–2 (Haiku), ~$5–10 (Sonnet) |
| `hormones` | reads from the post text (and, for a comment, its thread post when the same author wrote it) whether the author was on hormones when tested (`yes` / `no` / `unknown`, with a quote) | cents |
| `export` | one report per post or comment, joined to the author's earliest first-person PSSD / PFS / PAS case (group, first causal drug, tier C), checked, and sorted into files | free |

`export` writes `dutch_<file>.csv` and `dutch_<file>_metadata.csv` (`record_id, group, tier, drug`), with
record ids `rd001`…:

- `default`: each author's earliest report that passes every check and was not taken on hormones
- `hormones`: on hormones at test (`yes`). A `no` whose quote is not in the post counts as `unknown`
- `failing`: a check failed (status disagrees with value and range, a unit not like `ng/mg`, a marker name
  the PDF extractor does not know, a printed sum that does not add up, an unreadable row, two pages
  disagreeing, or no hormone answer). May not load, since a bad unit is one of the failures
- `retests`: an author's later clean reports

A repost by the same author with identical numbers is dropped. A report with no printed report month
takes the post's month, with a warning. No export holds a Reddit id, URL, quote, image path or author
key. Images can show a name, date of birth and accession number: they stay in `data/images/` and no
field in the schema can hold an identity. The hormone quotes appear only on the local review page.

**Accuracy:** not measured yet. Hand-check the 20 reports from `review`, then publish the `score`
result here. If more than 2% of markers are wrong, switch to Sonnet 5.5: `extract --model claude-sonnet-5-5`,
`hormones --vision-model claude-sonnet-5-5` (asks only about posts not yet read), then `export` and
`review` with `--model claude-sonnet-5-5`. Haiku readings stay cached.

Tests: `.venv/bin/python -m unittest discover tests` (run with `../analysis/.venv/bin/python` to also
check that the default export loads in the analysis).

## Limits

These are self-reports from people who chose to post. There is no control group, posting is biased
towards people who got worse or found something that helped, and most posts give no sex, dose or
lab values. Use the data to rank questions, never to say what works.

## Privacy rules

- `data/` is git-ignored and must never be committed or uploaded. It is real people's health posts.
- `data/.author_salt` makes author keys unlinkable to usernames. Losing it means new runs produce
  different keys; leaking it lets anyone with the subreddit's author list reverse them.
- Hand data to a researcher or data holder directly, as the `export/` and `dutch_export/` tables only.
  Never share `images/`, the `dutch_*.jsonl` work files or the DUTCH review page.
