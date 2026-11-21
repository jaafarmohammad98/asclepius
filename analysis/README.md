# analysis

The analysis engine (milestone 3). So far it loads and checks data, and writes synthetic datasets
with a known answer for testing the statistics.

```sh
cd analysis
uv sync
uv run pytest            # tests
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

## Two kinds of dataset, never pooled

| Loader | Input | Returns |
|---|---|---|
| `asclepius.dataset.load_lab_reports(paths, metadata=None)` | one or more CSV files from `npm run extract -- --csv`, and optionally a metadata file | `LabDataset` |
| `asclepius.selfreport.load_self_reports(dir)` | the harvester's `data/export/` folder (five tables) | `SelfReportDataset` |

Both stop with a `DatasetError` that names the file, line and column of the first problem. A
self-report message never contains a `case_id` or `author_key`.

**Lab reports.** One row per marker per report. Reports are named `<file number>/<record_id>`,
because each extractor file numbers its reports from `r001`. A unit must look like the ones the
extractor reads (`ng/mg`, `µg/mg`), and one marker must use one unit across all files loaded
together.

**Self-reports.** Merged so each author counts once (spec S9). Authors are numbered 1, 2, 3...;
case ids and author keys are dropped. Harvester condition `PSA` becomes the plan's `PAS`;
`unclear`, `multiple` and `none` stay as their own groups. By default the loader leaves out
posts with `evidence_ok` false (`include_unverified=True` keeps them) and posts from deleted
accounts, which cannot be merged. `excluded` counts both. A blank condition counts as `unclear`.

**Groups, tiers and questionnaire data.** `LabDataset.records` has one row per report with its
`group` and confidence `tier` (A, B, C). They come from an optional metadata CSV keyed by
`record_id` (`asclepius.metadata`):

| Column | Required | Values |
|---|---|---|
| `record_id` | yes | `r001`, or `2/r001` when more than one loaded file has an `r001` |
| `group` | yes | PSSD, PFS, PAS, SSRI-exposed control, healthy control (any case); blank means `unassigned` |
| `tier` | no | A, B or C; blank or absent means B |
| `drug` | no | free text |
| `months_since_stopping` | no | a number, 0 or more |
| `score_<name>` | no, any number | a symptom score from the questionnaire |

Any other group name or column is rejected. A report with no metadata row goes to `unassigned`,
tier B. Tier B is an assumption here, not a check: the extractor CSV carries no check results, so
a report is only tier C if the metadata says so. Rows naming a report that was not loaded are left out and listed in
`LabDataset.unmatched`. Two rows for the same report must agree (identical repeats are fine).

Both datasets carry a `fingerprint` (SHA-256 of the input files, metadata included) for the reproducibility record
(spec S7); `SelfReportDataset.settings` records the loader switch.

How an author's posts merge:

| Field | Rule |
|---|---|
| condition | the one named condition (PSSD, PFS, PAS); two different ones, or `multiple`, give `multiple`; else `unclear`, else `none` |
| sex, onset | the value every post that gives one agrees on; disagreement gives missing (sex) or `unclear` (onset) |
| trajectory, age, intervention outcome | from the most recent post that gives a clear value; same-year disagreement gives `unclear` (age: the largest) |
| months since stopping, dose, duration | the largest reported |
| symptoms | once each, at the worst severity reported |
| labs | every distinct value kept |

On the full r/PSSD export (Nov 2026): 3,272 authors with unverified posts kept, matching the
harvester README; 3,153 with the default.

## Synthetic datasets

`asclepius.synthetic` writes datasets of either kind where the true answer is known (spec S8),
so the statistics can be tested against it. The same settings and seed give byte-identical files,
and both kinds load through the loaders above.

```sh
# lab reports: 40 PSSD vs 40 healthy controls, androsterone 40% lower in PSSD
uv run python -m asclepius.synthetic lab out/lab --group PSSD=40 --group "healthy control=40" \
    --plant androsterone:0.6 --seed 1
# self-reports: finasteride users report erectile dysfunction at 3x the odds; tadalafil 60% improved
uv run python -m asclepius.synthetic selfreport out/sr --authors 500 \
    --associate finasteride:erectile_dysfunction:3 --outcome tadalafil:0.6:0.05 --seed 1
# either kind with --null: nothing planted, no group differs
```

**Lab reports** (`write_lab_reports`) give `reports.csv` (the extractor format), `metadata.csv`
(`record_id`, `group`, `tier`; about a quarter tier A, set with `--tier-a-share`) and `truth.json`. Values are log-normal per sex, with each
report's DUTCH reference range at about the 2.5th and 97.5th percentiles, for 26 markers that
include every one the enzyme-activity ratios need. A planted difference multiplies one group's
values of one marker by a fold (optionally in one sex only).

**Self-reports** (`write_self_reports`) give the harvester's five tables and `truth.json`. A
planted association raises the odds of one symptom for one causal drug; a planted outcome sets
the share of authors who tried an intervention reporting improvement and worse. Some authors post
2 to 4 times, with their symptoms and interventions spread over the posts, so the planted
answer only appears once posts are merged per author. A few posts come from deleted accounts,
and a few fail the evidence check; the loader must leave out both (the second by default).
