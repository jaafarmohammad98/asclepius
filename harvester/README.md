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

## Limits

These are self-reports from people who chose to post. There is no control group, posting is biased
towards people who got worse or found something that helped, and most posts give no sex, dose or
lab values. Use the data to rank questions, never to say what works.

## Privacy rules

- `data/` is git-ignored and must never be committed or uploaded. It is real people's health posts.
- `data/.author_salt` makes author keys unlinkable to usernames. Losing it means new runs produce
  different keys; leaking it lets anyone with the subreddit's author list reverse them.
- Hand data to a researcher or data holder directly, as the `export/` tables only.
