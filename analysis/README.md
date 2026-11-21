# analysis

The analysis engine (milestone 3). So far it loads and checks data, writes synthetic datasets
with a known answer for testing the statistics, compares markers between two groups, places
values within their own report's reference range, compares enzyme-activity ratios, and corrects
and labels a whole run of tests.

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
together. Within a report, a marker has one row (an exact repeat counts once), and every row gives the
same sex and age or leaves it blank.

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

## Group comparison

`asclepius.compare.compare_groups(data, "PSSD", "healthy control")` compares every marker between
two groups of a `LabDataset`, separately in each sex (spec A1), and returns a findings table, one
row per marker and sex, largest effect first:

| Column | Meaning |
|---|---|
| `comparison` | `group comparison` (the other kinds are below) |
| `marker_id`, `marker_name`, `sex` | what was compared |
| `group_a`, `group_b`, `n_a`, `n_b` | the two groups and how many reports each had for this marker (S3) |
| `n_censored` | how many of those values were printed as a limit (`<x` or `>x`) |
| `age_a`, `age_b` | each group's median age; shown so an age difference is visible, not adjusted for |
| `median_a`, `median_b` | each group's median value (a limit read as its number) |
| `effect_size` | Cliff's delta, P(a > b) - P(a < b): -1 when every value in group a is below every value in group b, 0 for no difference |
| `ci_low`, `ci_high` | 95% percentile bootstrap interval of the effect size (2,000 resamples, each group resampled separately) |
| `p_value` | two-sided permutation test of Gehan's statistic, exact when there are at most 9,999 ways to split the values, else 9,999 random splits (S1); with no limits this is the rank-sum (Mann-Whitney) test |

A value printed as a limit is compared as a range: `<5` is below 5, so it is lower than 5 or 7 but
neither higher nor lower than 4 or `<3`, and such a pair counts as a tie. Reports with no sex are
left out. The same `seed` gives the same table. `compare_values(a, b)` does the same for two lists
of values, where a value is a number or a string like `"<5"`.

The group-size limit (S4) and the false discovery rate correction (S2) are applied by
`run_analysis` (below), not here, so a row on its own is not a finding. On synthetic data with groups of 40 (12 women each),
chance differences among the women often have a larger effect size than a planted 40% difference;
with groups of 100 the planted marker comes first every time (30 of 30 seeds).

## Position within the reference range

Without controls, the only comparison is the range printed with each value (spec A1, A8).
`asclepius.refrange.lab_range_findings(data)` does it for a `LabDataset` and
`self_report_range_findings(data)` for the typed-in values of a `SelfReportDataset`. Each returns:

- `patients`: one row per report and marker (self-reports: per author and test) with its `zone`,
  `below`, `within` or `above` the range (ends included), and its `position`, 0 at the range's low
  end and 1 at its high end (missing for a limit or a range with one end). One row per person, so
  it is working data, never shown or exported (S11).
- `rows`: findings-table rows, one per marker, sex and group, with the share of each group below,
  within and above, each with a 95% Wilson interval (`share_below`, `below_low`, `below_high`, ...).
  `n_a` counts the values placed; `n_unclear` the limits that could fall either side of a range
  end (`<15` against 10 to 20). `median_position` (of the values placed) and `age_a` (median age)
  are also given. There is one group here, so no effect size: the shares and their intervals are
  the result (S3), and `effect_size` stays empty in the combined table.
- Rows are per sex because ranges differ by sex; reports or authors with no sex get rows of their
  own (sex empty), since each value is still placed in its own range.
- `skipped`: markers with values but no range, and how many values were left out.

A range with one end (`<15` printed as the range) places values above it or within, never below.
A lab value printed exactly at a range end may have been rounded, so there the report's own
status (`Above range`, `Below range`) decides; two public sample values need this. A typed-in
value at a range end follows its `flag` (`high`, `low`) the same way.

Self-report rows are labelled `reference range, self-report`. A test name is matched lower-cased
with spaces tidied; units may differ, since each value is placed in its own range. An author counts
once per test, as `unclear` when their values fall in different zones. Groups are the authors'
conditions. On the r/PSSD export (Nov 2026): 346 author-test pairs placed, 1,360 values skipped
for having no range.

`combine_findings(compare_groups(...), lab_range_findings(...).rows, ...)` stacks them into one
findings table; the `comparison` column (`group comparison`, `enzyme-activity ratio`, `reference
range`, `reference range, self-report`) tells them apart, and columns one kind lacks are left empty. Reference-range
rows are not tests, so the correction skips them and they get no label. The count limit S11 is
not applied yet (M3-12), so most self-report rows, with fewer than 5 authors, must not be shown as
they are.

## Enzyme-activity ratios

The leading hypothesis is about enzyme activity, which shows up as a ratio between metabolites
(spec A2). `asclepius.ratios.RATIOS` holds the list as data; the table below is generated from it
(`uv run python -m asclepius.ratios`), and a test checks that it matches.

Ratio list: provisional, needs expert review.

| Ratio | Numerator | Denominator | Proxy for |
|---|---|---|---|
| Androsterone / etiocholanolone | `androsterone` | `etiocholanolone` | 5-alpha vs 5-beta reductase (androgens) |
| 5a-androstanediol / 5b-androstanediol | `5a_androstanediol` | `5b_androstanediol` | 5-alpha vs 5-beta reductase (androgens) |
| a-THF / b-THF | `a_tetrahydrocortisol_a_thf` | `b_tetrahydrocortisol_b_thf` | 5-alpha vs 5-beta reductase (cortisol) |
| a-pregnanediol / b-pregnanediol | `a_pregnanediol` | `b_pregnanediol` | 5-alpha vs 5-beta reductase (progesterone) |
| Free cortisol / free cortisone | `24hr_free_cortisol` | `24hr_free_cortisone` | 11-beta-HSD balance |
| 2-methoxy-E1 / 2-OH-E1 | `2_methoxy_e1` | `2_oh_e1` | Methylation (COMT) activity |
| 2-OH-E1 / 16-OH-E1 | `2_oh_e1` | `16_oh_e1` | Estrogen metabolism pathway preference |
| 2-OH-E1 / 4-OH-E1 | `2_oh_e1` | `4_oh_e1` | Estrogen metabolism pathway preference |

`compute_ratios(data)` gives `data`, a `LabDataset` whose markers are the ratios (`marker_id` like
`androsterone/etiocholanolone`, unit `ratio`, no reference range), and `skipped`: per ratio and
reason, how many reports had none. A ratio is skipped for a report when a component is missing or
blank (`component missing`), when either is zero (`component zero`), or when both are limits the
same way, as `<10` over `<2` could be anything (`both components limits`). One limit gives a limit
on the ratio: `<10` over 2 is `<5`, 10 over `<2` is `>5`.

`compare_ratios(data, "PSSD", "healthy control")` runs them through `compare_groups` and returns
`rows` and the same `skipped`. The rows have its columns and rules (within sex, Cliff's delta,
permutation p-value), with `comparison` set to `enzyme-activity ratio` and a `status` column,
`provisional, needs expert review` until the clinician review (M3-11). All eight ratios are
computed for the three public sample reports. With 100 per group and androsterone halved in PSSD,
androsterone / etiocholanolone is the top row in both sexes in 19 of 20 seeds.
With androsterone ×0.8 and etiocholanolone ×1.25 (a small enzyme-like shift), the ratio's effect
is larger than androsterone's alone in 39 of 40 sex-seed pairs, but not reliably larger than
etiocholanolone's: synthetic markers are drawn independently, so the ratio carries both markers'
noise, where real related metabolites rise and fall together.

Ratios are tested in a call of their own, apart from the raw markers; `run_analysis` corrects
both together, as one run (S2).

## A run: correction, limits and labels

`asclepius.run.run_analysis(data, "PSSD", "healthy control", settings)` is the entry point for a
lab-report dataset. It runs `compare_groups` and `compare_ratios`, stacks them with
`combine_findings`, and corrects the combined table once: one Benjamini-Hochberg false discovery
rate correction over raw markers, ratios and both sexes together (S2), 68 tests on the synthetic
data. A run is one dataset; lab reports and self-reports are corrected in separate runs. It returns:

- `findings`: the tested rows, with `q_value` (the corrected p-value) and `label`.
- `refused`: rows not tested because a group had fewer than 10 values in that sex, with a `reason`.
  They do not count in the correction.
- `skipped`: as in `compare_ratios`.
- `statement`: `nothing reliable yet` when no finding is strong or suggestive (S8), else the counts.
- `settings`: every threshold the run used (`Settings.record()`), marked `provisional, needs expert
  review` until M3-11.

A group with fewer than 10 reports is refused outright with `GroupTooSmall`, whose message names the
group and its size (S4). The labels (S5):

| Label | Rule (defaults in `Settings`) |
|---|---|
| strong | q at most 0.05, Cliff's delta at least 0.33 either way (Romano's "medium"), and holds in Tier A |
| suggestive | q at most 0.10, and not strong |
| noise-level | everything else |

Holding in Tier A is the `holds_in_tier_a` column, which the Tier A re-run (M3-07) will set. Until
then no row has it, so every lab finding "holds only in the full dataset" and is at most
suggestive. This is needed, not just pedantic: on null synthetic data, 3 of 60 seeds give a chance
finding with q at most 0.05 and a medium effect, as a 5% false discovery rate allows; and with a
real difference planted, an unrelated marker can pass too. Only the Tier A condition can keep
those out of strong. With q at most 0.10, about 1 null run in 10 has a suggestive finding.

`label_findings(table, settings, self_report=False)` does the correction and labelling for any
findings table; pass the whole run's table. Self-report analyses (M3-12, M3-14) call it with
`self_report=True`: no row can be strong (S10), and every row gets the caveat `self-report, no
controls`.
