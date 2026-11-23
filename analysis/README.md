# analysis

The analysis engine (milestone 3). So far it loads and checks data, writes synthetic datasets
with a known answer for testing the statistics, compares markers between two groups, places
values within their own report's reference range, compares enzyme-activity ratios, corrects
and labels a whole run of tests, re-runs each finding on Tier A records, records and exports
each run so it can be reproduced, looks for subgroups of patients, compares symptoms between the
drugs self-reporting authors took, and has a local web app so
researchers need no command line after start-up.

```sh
cd analysis
uv sync
uv run asclepius         # the web app: opens http://127.0.0.1:8765/ in the browser
uv run pytest            # tests
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

The app also serves the extractor's browser page at `/extractor/` (from `extractor/dist/` once
`npm run build` has run there; the installed app carries its own copy), and loads a built-in example
of either kind with one click (`asclepius.examples`: invented by `asclepius.synthetic` with fixed
settings and seeds; the lab example gives strong findings, the self-report one suggestive ones).
Installing it as one tool is in the [top-level README](../README.md#install-the-researcher-tool).

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
the share of authors who tried an intervention reporting improvement and worse. A planted symptom
subgroup (`PlantedSubgroup`, Python only) is a share of authors who report its symptoms often and
the other subgroups' symptoms rarely, optionally with more onsets during use. Planted reporting
(`PlantedReporting`, Python only) is a share of authors who report every symptom at higher odds:
more symptoms, and no subgroup. Some authors post
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
rows are not tests, so the correction skips them and they get no label. The rows themselves do
not apply the count limit S11; the web app's findings view does (below), so most self-report rows,
with fewer than 5 authors, must not be shown as they come from here.

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

- `findings`: the tested rows, with `q_value` (the corrected p-value), the Tier A re-run (below) and `label`.
- `refused`: rows not tested because a group had fewer than 10 values in that sex, with a `reason`.
  They do not count in the correction.
- `skipped`: as in `compare_ratios`.
- `statement`: `nothing reliable yet` when no finding is strong or suggestive (S8), else the counts.
- `settings`: every threshold the run used (`Settings.record()`), marked `provisional, needs expert
  review` until M3-11.
- `record`: the run record (below).

A group with fewer than 10 reports is refused outright with `GroupTooSmall`, whose message names the
group and its size (S4). The labels (S5):

| Label | Rule (defaults in `Settings`) |
|---|---|
| strong | q at most 0.05, Cliff's delta at least 0.33 either way (Romano's "medium"), and holds in Tier A |
| suggestive | q at most 0.10, and not strong |
| noise-level | everything else |

### Tier A re-run (S6)

A finding that disappears among clinician-confirmed patients should not be trusted, so
`run_analysis` re-runs every finding on Tier A records only, with the same tests and seed, and
puts the result beside the full-dataset one:

| Column | Meaning |
|---|---|
| `tier_a_n_a`, `tier_a_n_b` | how many Tier A values each group has for this marker and sex |
| `tier_a_effect_size`, `tier_a_ci_low`, `tier_a_ci_high` | Cliff's delta and its interval, on Tier A only |
| `tier_a_p_value` | the permutation p-value on Tier A only, not corrected again (it checks findings the corrected run found) |
| `tier_a_result` | `holds`, `does not hold`, or `Tier A too small to test` |
| `holds_in_tier_a` | True when `tier_a_result` is `holds` |

A finding holds when Tier A has at least 10 values in each group (the S4 limit), its effect goes the
same way and is at least medium (0.33), and its Tier A p-value is at most 0.05 (`Settings.tier_a_p`,
provisional). With fewer than 10 the comparison is refused as in the full run: only the sizes are
shown, and the finding is at most suggestive. With 100 per group and a quarter tier A, as in the
synthetic default, that is usually the women; at 40 per group it is almost every row.

This matters: on null synthetic data, 3 of 60 seeds (40 per group) give a chance finding with q at
most 0.05 and a medium effect, as a 5% false discovery rate allows. With 100 per group and half
tier A, 2 of 60 do; the Tier A re-run rejects one and confirms the other (seed 48, b-pregnanediol in
men), which is labelled strong. Tier A records are part of the full dataset, so a chance pattern
can carry into them: the check makes a chance strong finding rarer, not impossible. With a
real difference planted, an unrelated marker can pass too. With q at most 0.10, about 1 null run
in 10 has a suggestive finding.

### Run record and export (S7)

`RunResult.record` is the run record (`asclepius.record.run_record`):

| Key | Value |
|---|---|
| `tool` | `asclepius` and its version |
| `code_sha256` | a hash of the package's source files, since code can change without a version bump |
| `date` | when the run was made (UTC) |
| `dataset` | `kind`, `fingerprint`, and `files`: each input file's `role` (`lab reports`, `metadata`; for self-reports the table name) and `sha256`, in load order; not its name, which could name a patient. Self-reports add the `loader` switch |
| `groups` | the two groups compared |
| `settings` | `Settings.record()` |
| `status` | `provisional, needs expert review` |

`export_run(result, folder)` writes `findings.csv`, `refused.csv`, `skipped.csv` and
`run_record.json` into an empty folder; the written record adds `outputs`, the SHA-256 of each CSV,
so every file can be matched to its record. `reproduce(record, paths, metadata=...)` runs a record
again: it refuses (`ReproductionError`, naming the file) when an input file is missing, extra or
changed, or when the tool version or code differs, and otherwise gives identical findings and
byte-identical CSV exports; only the date changes.

`label_findings(table, settings, self_report=False)` does the correction and labelling for any
findings table; pass the whole run's table. Self-report analyses (M3-12, M3-14) call it with
`self_report=True`: no row can be strong (S10), and every row gets the caveat `self-report, no
controls`.

## Subgroups

PSSD may be several conditions (spec A3). `asclepius.cluster.cluster_patients(data, groups=["PSSD"])`
looks for subgroups among the reports of the groups named; the groups themselves are not used.
`groups` is required: clustering patients with controls mostly finds the patient/control split,
not subgroups, though naming both is allowed as a deliberate check. An empty list or an unknown
group name is refused. `tiers=["A"]` limits it to Tier A reports, for a Tier A re-run
(S6). Fewer than 30 patients is refused with `TooFewPatients` (S4); reports with no sex are left
out and counted in the message.

1. **Features.** Every raw marker and every enzyme-activity ratio. Each becomes a normal score
   within sex: the normal quantile of its rank / (n + 1), ties sharing a rank, limits read as their
   number. Within sex because ranges differ by sex; ranks so that one extreme value or a skewed
   marker cannot dominate. A feature missing for more than 20% of patients is left out
   (`dropped`); other missing values get the middle score, 0 (`filled` counts them).
2. **Clusters.** k-means, best of 10 k-means++ starts, for 2 to 6 clusters (fewer when 10 per
   cluster would not fit).
3. **Stability chooses the number.** The patients are clustered again in 50 random subsamples of
   80%, and the adjusted Rand index says how well each matches the full clustering (1 the same, about 0
   chance). The number of clusters with the highest mean, among those whose every cluster has at
   least 10 patients (so M3-08b can compare them, S4), is chosen. It must reach 0.8.
4. **Better than one group.** k-means splits one stretched cloud of patients the same way every
   time: when every marker of a patient shares one factor (one person makes more of everything),
   the high-vs-low split reaches stability 0.8 in most seeds. So the chosen clustering must also be tighter,
   in the space its centres span, than k-means finds in all but 5% of 99 datasets drawn from one
   normal distribution with the patients' own covariance and scored within sex the same way (the
   idea of SigClust, Liu et al. 2008). Without that within-sex scoring the test gave clusters in 5 of 20
   one-factor datasets, because evenly spaced scores split more cleanly than a normal sample.

   **An exception to S1.** This check compares against normal data, which S1 rules out. Each
   feature is already a normal score, so what it assumes is that features relate to each other in
   straight lines. No permutation does the job: shuffling each feature alone removes the shared
   factor the check is there to catch, and shuffling along the data's main directions keeps the
   gap between real subgroups, so the check would never find them. The settings record it
   (`one_group_check`), and it is listed for the clinician review (M3-11).

`ClusterSettings` holds every threshold above (provisional, needs expert review). The result:

| Field | Meaning |
|---|---|
| `statement` | `2 stable clusters` (or 3, ...) or `no stable clusters` (S8) |
| `reason` | why: the stability and one-group p-value, or which test failed (or fewer than two usable markers) |
| `k`, `null_p` | the number of clusters (None when none) and the one-group test's p-value |
| `assignments` | `record` and `cluster` (1 the largest): working data for M3-08b, never shown or exported (S11) |
| `sizes` | patients per cluster and sex; empty where fewer than 5 (S11) |
| `profile` | per cluster and feature, the median normal score (0 the middle of that sex, about ±1 one standard deviation), to describe each cluster |
| `stability` | per number of clusters: `stability`, `smallest` cluster, `eligible` |
| `features`, `dropped`, `filled`, `no_sex` | what went in; a count below 5 is None or empty (S11) |
| `settings`, `record` | `ClusterSettings.record()` plus the `tiers` clustered, and the run record (S7, as for `run_analysis`) |

On synthetic data, with PSSD and PFS standing in for two hidden subgroups and four 5-alpha
reduced markers at 40% in one, from 20 seeds each:

| Data | Clusters found | Patients in their subgroup's cluster |
|---|---|---|
| 50 + 50, planted | 20 of 20 | 93% to 99% |
| 30 + 30, planted | 16 of 20 (others: stability just under 0.8, or not better than one group) | 90% to 98% |
| 30 + 30, four markers at 30% | 20 of 20 | |
| 30 + 30, four markers at 70% | 0 of 20: a milder subgroup is not found at this size | |
| 60 or 100, nothing planted | 0 of 20 (stability 0.2 to 0.6) | |
| 60, one shared factor (log-sd 0.3 or 0.6) | 1 of 20 each, as a 5% test allows | |

A run takes under a second for 100 patients.

### Linking subgroups to symptoms and drugs

A cluster only matters if it corresponds to something clinical (spec A3).
`asclepius.subgroups.link_subgroups(data, clusters, settings)` tests whether the clusters differ by
each questionnaire symptom score (`score_<name>` in the metadata) and by drug taken (`drug`):

- With 2 clusters, cluster 1 against cluster 2; with more, each cluster against all the others.
- A score is compared with the same test as a marker (permutation test on ranks, Cliff's delta,
  bootstrap interval), both sexes together: the clusters were already made within sex, and a
  questionnaire score has no sex-specific range. `median_a`, `median_b` are the scores' medians.
- A drug is a yes/no value per patient, compared the same way. On yes/no values Cliff's delta is
  the difference between the shares who took it, `share_a` minus `share_b`. Drug names are
  matched ignoring case. Patients with no drug recorded are left out of drug tests.
- Each row is re-run on the Tier A patients with the same clusters (S6): does the link hold among
  clinician-confirmed patients? Tier A is not clustered again; whether the clusters themselves
  hold in Tier A is `cluster_patients(..., tiers=["A"])`.
- A side with fewer than 10 values, or a drug fewer than 10 clustered patients took, is not tested
  and goes to `refused` (S4). Both shares are hidden when, on either side, fewer than 5 took the
  drug or fewer than 5 did not: with one shown, the effect (their difference) would give the other
  back, and with the side's size, the count. Tier A sizes and refused counts below 5 are hidden
  too (S11). The effect and its interval are still shown, so an effect near ±1 with the sizes
  still narrows a small count down; for review in M3-11.
- With no stable clusters there is nothing to link, and the statement says so (S8); the run's
  statement ends with it.

The rows are not corrected or labelled on their own. Pass them to the run, which corrects them with
every marker and ratio test and labels them like any other finding (S2, S5):

```python
clusters = cluster_patients(data, groups=["PSSD"])
links = link_subgroups(data, clusters, settings)
result = run_analysis(data, "PSSD", "healthy control", settings, subgroups=links)
```

Subgroup rows have `comparison` `subgroup: symptom score` or `subgroup: drug taken`, `marker_id`
`score_<name>` or `drug:<name>`, `sex` `all`, `group_a` `cluster 1` (and so on), and
`subgroups_of`, the groups clustered. The run refuses links made from other data or other
settings. Its record gains `subgroups` (the clustering's statement, reason, groups and settings),
and `reproduce` clusters and links again from it.

`write_lab_reports(scores=[...], planted_scores=[PlantedScore("libido", 3.0, group="PFS")],
drugs={"PSSD": [...]})` writes synthetic symptom scores (whole numbers 0 to 10, mean 4 and sd 2
before rounding) and drugs into the metadata, without changing the reports. From 20 seeds, with
PSSD and PFS as two hidden subgroups as above, a `libido` score shifted in PFS and a `fatigue`
score with nothing planted:

| Data | Libido found (suggestive or strong) | Libido strong | Fatigue found |
|---|---|---|---|
| 50 + 50, shift 3 points, a quarter tier A | 20 of 20 | 11 of 20 (the others suggestive: about 12 Tier A patients a side, near the limit of 10) | 1 of 20 |
| 50 + 50, shift 1 point | 8 of 20 | 1 of 20 | 1 of 20 |
| 100 + 100, shift 3 points, half tier A | 20 of 20 | 20 of 20 | 1 of 20 |
| 50 + 50, nothing planted | 1 of 20 | 0 of 20 | 1 of 20 |

## Cross-syndrome comparison

Do PSSD, PFS and PAS share a mechanism (spec A4)? If they do, a marker or ratio that is off in one
should be off the same way in the others. `asclepius.syndromes.compare_syndromes(data, reference,
settings)` compares every raw marker and enzyme-activity ratio, within sex, in two kinds of test:

- **Each syndrome against the reference** (`comparison` `cross-syndrome: vs reference`), which
  gives each syndrome's direction. The reference is a control group, `healthy control` by default
  or `SSRI-exposed control`, compared with the same test as any group comparison.
- **Each pair of syndromes against each other** (`cross-syndrome: between syndromes`), which says
  where they differ.

The syndromes compared are those with reports in the data. Every test is corrected together in one
false discovery rate correction, re-run on Tier A records and labelled exactly as in `run_analysis`
(S2, S5, S6): 408 tests with all three syndromes and controls. A `measure` column says whether a
row is a `raw marker` or an `enzyme-activity ratio`.

**Without controls**, `reference="reference range"` compares each syndrome with the reports' own
reference ranges instead. Of the values placed with certainty (as in the reference-range rows), the
effect is the share above the range minus the share below it, from -1 (all below) to 1 (all above),
0 when as many fall on each side, as in healthy people, for whom the range is about the middle 95%.
The p-value is an exact two-sided sign test of above against below, the interval a percentile
bootstrap. This is not a Cliff's delta, but the same 0.33 threshold labels it: a net third of a
group outside its range is a large shift, so the threshold is stricter here than for controls. Both
the test and the threshold are new and listed for the clinician review (M3-11).
It assumes that healthy values fall outside the range equally often on each side, which a real
lab's range may not do. Ratios have no range, so they are only compared between syndromes.
`share_above` and `share_below` are both hidden when fewer than 5 fall on either side (S11). When
none fall on one side and 1 to 4 on the other, the effect and its interval are hidden too, since
the effect times `n_a` would be that count; with at most 4 values outside, the p-value is at least
0.125, so such a row is never a finding.

Limits (S4): a syndrome with fewer than 10 reports is left out and listed in `refused`; fewer than
two syndromes left, or a reference group with fewer than 10, is refused with `GroupTooSmall`. A
test where either side has fewer than 10 values in that sex is refused as in a run. No count below
5 is shown in the findings, the refusals or the summary (S11): sizes, limits (`n_censored`),
unclear values and Tier A sizes are hidden below 5.

The result is a `RunResult` (`findings`, `refused`, `skipped`, `statement`, `settings`, `record`)
plus `summary`, one row per measure, marker and sex:

| Column | Meaning |
|---|---|
| `n_PSSD`, `n_PFS`, `n_PAS`, `n_reference` | values per syndrome and in the control group (none for reference ranges) |
| `effect_PSSD`, ... | each syndrome's effect against the reference |
| `direction_PSSD`, ... | `higher` or `lower` when that test is strong or suggestive, else `no clear shift`; `not tested` when refused |
| `differ` | the pairs whose test is strong or suggestive, as `PFS lower than PSSD` |
| `pattern` | see below |
| `shared_by` | the syndromes shifted the same way, for `shared` and `same direction, different size` |
| `label` | how strong the pattern is: the weakest label among the syndromes' shifts that make it; for `differs` found only by pair tests, the strongest of those |

| Pattern | When |
|---|---|
| `shared` | two or more syndromes are shifted the same way, none the other way, and those do not differ from each other |
| `same direction, different size` | as `shared`, but two of them also differ from each other |
| `differs` | syndromes are shifted opposite ways, or a pair differs |
| `one syndrome only` | one syndrome is shifted, and no pair differs |
| `no clear shift` | nothing is strong or suggestive |
| `not tested` | no syndrome could be tested against the reference |

A syndrome with `no clear shift` beside two that are shifted does not stop a `shared` pattern; the
`differ` column says whether it was found to differ from them. The statement is the run's (S8),
followed by how many marker-sex rows are `shared`, `same direction, different size` or `differs`.
The record gains `analysis` (`cross-syndrome`) and `reference`; `groups` lists the syndromes
compared and the reference. `export_run` adds `summary.csv`, and `reproduce` runs it again.

On synthetic data with 100 per syndrome (and 100 controls), 20 seeds each:

| Data | Planted marker, men | Planted marker, women | Other markers |
|---|---|---|---|
| androsterone ×0.5 in all three, half tier A | `shared` 20 of 20 (17 strong) | `shared` 19 of 20 (11 strong) | never `shared`; a chance `differs` or `one syndrome only` in most runs, almost all suggestive |
| androsterone ×0.5 in all three, a quarter tier A | `shared` 20 of 20 (16 strong) | `shared` 19 of 20 (all suggestive: Tier A too small) | as above |
| androsterone ×0.5 in PFS only, half tier A | `differs` 20 of 20 (17 strong) | `differs` 18 of 20 | as above |
| androsterone ×0.3 in all three, no controls | `shared` 20 of 20, strong | `shared` 20 of 20 (16 strong) | never `shared` |
| androsterone ×0.5 in all three, no controls | `shared` 20 of 20, suggestive | `no clear shift` 18 of 20 | |
| nothing planted, 40 per group | 20 of 20 seeds say `nothing reliable yet` | | |

Against reference ranges, a shift must push values past the range to show, so it finds only large
shifts: halving androsterone moves too few women's values out of range. With 40 per group, a
shared shift in women is mostly missed. The chance findings on other markers are what a 10% false
discovery rate allows once real findings exist: one false finding in ten is expected among the
suggestive ones.

## Drug and symptom profile

Different drugs may leave different symptom patterns, which would point at mechanisms worth
testing (spec A6). `asclepius.profile.profile_drugs(data, settings, top_drugs=8)` takes a
`SelfReportDataset` and compares, for every symptom, how often authors report it between drug
classes and between the most common single drugs, and the same for onset and trajectory.
`author_profiles(data)` is its working data: one row per author with their `drugs`, `classes`,
`onset`, `overall_trajectory` and one True/False column per symptom; it is one row per person, so
never shown or exported (S11).

**Drug classes.** `DRUG_CLASSES` maps each causal drug, as the harvester names it, to one of six
classes. A drug not listed is in no class and only appears in the single-drug comparisons;
`unspecified_antidepressant` is not listed, since it could be any class. Brand names and
misspellings seen in the r/PSSD export are listed where the generic is clear. Provisional, for the
clinician review (M3-11).

| Class | Drugs |
|---|---|
| SSRI | sertraline, escitalopram, fluoxetine, paroxetine, citalopram, fluvoxamine, dapoxetine, `unspecified_ssri` |
| SNRI | venlafaxine, desvenlafaxine, duloxetine, milnacipran, levomilnacipran, `unspecified_snri` |
| tricyclic | clomipramine, amitriptyline, nortriptyline, imipramine, desipramine, doxepin, dosulepin, melitracen, unspecified tricyclics |
| other antidepressant | bupropion, vortioxetine, vilazodone, mirtazapine, trazodone, nefazodone, mianserin, agomelatine, moclobemide, phenelzine, tianeptine |
| finasteride | finasteride |
| isotretinoin | isotretinoin |

**An author with drugs in several classes counts in each.** The result's `note` and the run
record say so, and `summary` gives how many authors that is (352 on the r/PSSD export), how many
took only drugs in no class, and how many recorded no causal drug.

**What is compared.** Every row is one yes/no outcome per author, compared between two sides:

| Level | Side a | Side b |
|---|---|---|
| `class` | authors with a drug in the class | the other authors with a drug in some class |
| `drug` | authors who took the drug | the other authors with a causal drug |

The single drugs are the `top_drugs` most common named ones (no `unspecified_...`), leaving out a
drug taken by exactly the authors of its class (finasteride, isotretinoin, a lone SNRI), whose class
row is the same test. The outcomes are each symptom reported by at least 10 authors (absence of a
report counts as not reporting it), onset during use (of authors whose onset is during use or after
stopping), and each trajectory: recovered, improving, stable, worsening (of authors with a clear one).

| Column | Meaning |
|---|---|
| `comparison`, `marker_id`, `marker_name` | `drug profile: symptom` (`symptom:<name>`), `drug profile: onset` (`onset:during_use`) or `drug profile: trajectory` (`trajectory:<value>`) |
| `level`, `group_a`, `group_b` | as above; `group_b` is `other drug classes` or `other causal drugs`; `sex` is `all` |
| `n_a`, `n_b` | authors on each side with a value for this outcome (S3) |
| `reported_a`, `reported_b`, `share_a`, `share_b` | how many, and what share, of each side have the outcome |
| `effect_size`, `ci_low`, `ci_high` | share a minus share b, with Newcombe's 95% interval (from each share's Wilson interval); on yes/no values this is Cliff's delta |
| `odds_ratio`, `odds_ratio_low`, `odds_ratio_high` | the odds ratio and its 95% Woolf interval; with an empty cell, 0.5 is added to every cell |
| `p_value` | two-sided Fisher's exact test (S1) |
| `q_value`, `label`, `caveat` | from `label_findings(..., self_report=True)`, over every row of the run together (S2): never strong, at most suggestive, `self-report, no controls` (S10) |

A group, or a side of one outcome, with fewer than 10 authors is refused (S4), as is a symptom
fewer than 10 authors report; both go to `refused`. No count below 5 is shown (S11): sizes below 5
are hidden, and a row where a side has fewer than 5 with the outcome, or fewer than 5 without it,
hides its counts, shares, effect, odds ratio, p-value and q-value, since with the sizes any of
them gives the count back (the p-value with the total of reporters another row of the same
outcome shows: on the r/PSSD export that recovered every such count before it was hidden); it keeps
its sizes and label. A size below 5, which a `min_group_size` under 5 would let through, is hidden
too. A group's four trajectory rows add up to its size, so when one hides its cells, all four
do. The label still says a small row's q-value is at most 0.10; for review in M3-11.

The result has `findings`, `refused`, `summary`, `skipped` (empty), `note`, `statement` and
`record` (with `analysis` `drug and symptom profile` and the settings, `top_drugs` included).
`export_run` writes it like any run, `summary.csv` included.

On synthetic data, 20 seeds each:

| Data | Planted row found (suggestive) | Runs with another suggestive row |
|---|---|---|
| 1,000 authors, finasteride × erectile dysfunction at odds ratio 3 | 20 of 20, never strong | 12 of 20; 10 of them the mirror below |
| 500 authors, the same at odds ratio 4 (about 60 on finasteride) | 17 of 20 | 7 of 20, all the mirror |
| nothing planted, 500 authors | | 1 of 40 seeds (seed 1) |
| nothing planted, 1,000 authors | | 1 of 40 seeds |

Most other suggestive rows are the planted one's mirror: SSRI authors (and sertraline's) report
the symptom less than the rest, which includes finasteride. Each group is compared with the
others, so one group's real difference shows, reversed, in the groups it is compared against:
read a lower share in a large class alongside the small classes that make up its rest. On the r/PSSD export (Nov 2026): 798
tests, 36 suggestive, none strong; under a second.

## Symptom subgroups in self-reports

The lab-value clustering waits for real reports, but symptom profiles from thousands of authors
can already show whether PSSD splits into subtypes (spec A5).
`asclepius.symptom_subgroups.cluster_symptoms(data, settings, cluster_settings)` takes a
`SelfReportDataset`, clusters its authors by which symptoms they report, describes each cluster
and tests whether the clusters differ by drug class, onset or trajectory.

**Steps.**

1. One row per author (`author_profiles`, S9) and one yes/no feature per symptom at least
   `min_reporters` (10) authors report. The catch-all `other` is not a feature. Authors with fewer
   than `min_symptoms` (2) of the features are left out; `left_out` says how many (692 on the
   r/PSSD export).
2. The distance between two authors is the Jaccard distance: of the symptoms either reports, the
   share only one of them does. On yes/no data not reporting a symptom says less than reporting it,
   so two authors are not alike for both leaving out the same 40 symptoms, as they would be under a
   Euclidean distance.
3. k-medoids on that distance (k-means++-style starts, best of `restarts`) splits the authors into
   2 to `k_max` clusters. The number is chosen by the same subsample stability as the lab
   clustering (`asclepius.cluster.choose_k`): mean adjusted Rand index at least `min_stability`
   (0.8), every cluster at least `min_cluster_size` (30) authors.
4. The clustering must also be tighter (total distance to the medoids, as a share of the distance
   to one medoid of everyone) than in all but `null_alpha` (5%) of `null_samples` (49) copies of
   the data with the links between symptoms broken. The copies come from swap randomisation
   (curveball, Strona et al. 2014), which keeps how many symptoms each author reports and how many
   authors report each symptom. It is a permutation test (S1), unlike the lab clustering's check
   against normal data, and it rules out clusters that only split authors who report many symptoms
   from those who report few. Failing step 3 or 4 gives `no stable clusters` (S8), with the
   `reason`.
5. `summary`: each cluster's `top_symptoms` (5) most over-represented symptoms: `share_in` the
   cluster, `share_out` among the other authors, and the `difference`. A symptom with fewer than 5
   reporters or non-reporters in any cluster is not listed for any (S11): with 3 or more clusters,
   the other clusters' rows would give that count back.
6. `findings`: each cluster against the other (with 2 clusters, cluster 1 against cluster 2) or
   against the other clusters (with more), on each drug class (`level` `class`, of authors with a
   drug in some class), each of the `top_drugs` (8) most common single causal drugs (`level`
   `drug`, of authors with a causal drug, chosen as in the drug profile), onset during use, and
   each trajectory. The columns, tests, hiding of small cells and
   labels are the drug profile's (`level` is `cluster` for onset and trajectory; `comparison` is
   `symptom subgroup: drug class`, `symptom subgroup: causal drug`, `symptom subgroup: onset` or
   `symptom subgroup: trajectory`). All rows are corrected
   together, never strong, `self-report, no controls` (S2, S10). Symptoms are not tested: the
   clusters were made from them. A class fewer than 10 clustered authors took, or a side with
   fewer than 10, goes to `refused` (S4). When one row hides its cells, every row of the same
   outcome does, and every trajectory row when one does: with 3 or more clusters each row shows
   the outcome's total and the other clusters' rows the rest, and the four trajectories add up to
   the side's size. The record carries the profile's note that an author with drugs in several
   classes counts in each.

Clustering is refused below `min_authors` (30) authors left (`TooFewAuthors`, S4). The result has
`clusters`, `reason`, `k`, `sizes`, `summary`, `stability`, `null_p`, `symptoms`, `left_out`,
`findings`, `refused`, `statement` (the clusters' statement, then the findings') and `record`
(`analysis` `symptom subgroups`, with the clustering settings under `settings.clusters`).
`export_run` writes it like any run. Thresholds are provisional, for review in M3-11.

`left_out` is hidden below 5, but when clusters are found and 1 to 4 authors are left out, the
loader's author count minus the cluster sizes gives it back. It says only that a few authors
reported fewer than 2 of the symptoms used, nothing about who; it is accepted for now and listed
for the M3-11 review. (On the r/PSSD export it is 692.)

On synthetic data (`synthetic.PlantedSubgroup`), 400 authors, 20 seeds each:

| Data | Result |
|---|---|
| two subgroups, sexual and emotional symptoms, onset during use 50% and 15% | 2 clusters, described by their planted symptoms, in 20 of 20; onset suggestive in 20 of 20, another suggestive row in 2 |
| nothing planted | no stable clusters in 20 of 20 (stability about 0.2 to 0.3) |
| 30% of authors report every symptom at 6 times the odds (`PlantedReporting`): more symptoms, no pattern | no stable clusters in 20 of 20; with the stability check off, the swap test alone rejects the split in 20 of 20 (p 0.06 to 0.96) |

On the r/PSSD export (Nov 2026): 2,461 authors clustered on 51 symptoms; no stable clusters (the
most stable split, 2 clusters, has stability 0.45); about 6 seconds.

## Intervention outcomes

Patients have tried many things and reported the results. Ranked honestly, that tells researchers
which interventions are worth a proper study; it does not tell anyone what works (spec A7).
`asclepius.interventions.rank_interventions(data, settings, worse_signal=0.2)` takes a
`SelfReportDataset` and counts, per intervention, the authors reporting each outcome.

- **One author, one outcome.** The loader merges posts per author (S9): an author who reported
  the same intervention twice counts once, with their latest clear outcome. "Latest" is by year;
  posts of the same year that disagree give `unclear`, and a newer `unclear` does not replace an
  older clear outcome.
- **Unclear apart.** `unclear` outcomes are left out of the shares and counted in `unclear`;
  `n_clear` is the authors with a clear outcome, `n_authors` all of them.
- **Shares.** `share_improved` (major plus partial improvement) and `share_worse`, of `n_clear`,
  each with its 95% Wilson interval (`improved_low`, `improved_high`, `worse_low`, `worse_high`,
  S3). An intervention with fewer than `min_group_size` (10) authors with a clear outcome goes to
  `refused` (S4). The refused list names only interventions at least 5 authors reported; the
  rarer ones are counted in one row (`fewer than 5 authors`), since a rare name could point at one
  author's case (S11).
- **Ranking.** `rank` 1 has the highest lower bound of the share improved. `safety` lists,
  separately, the interventions whose share worse has a lower bound of at least `worse_signal`
  (20%, provisional, for review in M3-11), highest first; `safety_signal` marks them in
  `findings`. The flag is judged on the true share worse even when that share is hidden (below), so
  the cases most reported as worse are not dropped; a yes/no flag gives no count back.
- **Wording.** Every table has a `caveat` column: "reported outcome, not evidence that it works;
  self-report, no controls", and so does the statement. Nothing is worded as advice (overview
  rule 5); a test checks every text for "recommend", "effective" and the like.
- **Counts below 5 (S11).** No count from 1 to 4 is shown; 0, which is no one, is. A share with
  its interval gives its counts back, so `share_improved` is shown only when the improved and the
  other clear counts are 0 or at least 5, and `share_worse` likewise; when both are shown, the no
  effect count must be too, or only the share improved is shown. The outcome counts are then
  hidden within the groups the shown shares leave unknown (the major/partial split, for one), as
  the reference-range zones are: each count from 1 to 4, and the next smallest until the hidden
  ones add up to 0 or 5 or more. A row whose share improved is hidden is not ranked, and
  `not_ranked` says why; the page and the HTML report write a hidden count as "hidden", since it
  may be 5 or more, hidden beside one that is not. `n_authors`
  and `unclear` are shown only when `unclear` is 0 or at least 5.

The result has `findings` (ranked), `safety`, `refused`, `skipped` (empty), `statement` and
`record` (`analysis` `intervention outcomes`, the settings with `worse_signal`). `export_run`
writes it like any run, with `safety.csv`.

On synthetic data (`synthetic.PlantedOutcome`), 1,000 authors: tadalafil planted at 60% improved,
5% worse, and cyproheptadine at 10% and 45%. Over 60 seeds each planted share lies inside its
interval in 90% to 97% of runs (95% nominal); tadalafil ranks first and cyproheptadine is a safety
signal. On the r/PSSD export (Nov 2026): 75 interventions ranked (29 more with a hidden share),
11 safety signals (4 with their share hidden), 168 refused and named for fewer than 10 authors
with a clear outcome, and 1,645 reported by fewer than 5 authors counted, not named; under a
second.

## Web app

`uv run asclepius-app` starts the app and opens its page (`--port` picks another port,
`--no-browser` skips opening it). It runs on this computer only: the server listens on
127.0.0.1 and refuses requests that name another host or come from another site's page, so no
other web page open in the browser can read the data or start a run. Files picked are copied
into a temporary folder that is deleted when the app stops (Ctrl+C).

1. **Load.** Pick the extractor CSV files and, optionally, the metadata file; or the harvester's
   `data/export` folder. The two kinds are never loaded together. A loading problem is shown
   as the loader's message, naming the file, line and column.
2. **Review.** Lab reports: reports per group, and per group, sex and tier, and how many metadata
   rows named no loaded report. Self-reports: authors per condition, and the posts left out for
   failing the evidence check (`evidence_ok` false) or coming from deleted accounts. No count below 5
   is shown (S11); the page says "fewer than 5". Totals are shown, so when the counts hidden beside a
   total add up to 1 to 4 (which the total minus the others would give back), the next smallest
   counts are hidden too, until the hidden ones add up to 5 or more.
3. **Run.** For lab reports, pick the two groups to compare (PSSD and a control by default). The
   run then goes through its steps with a progress bar, each ending `done`, `refused` (with
   the reason, such as a group under 10), `failed` (an unexpected error, shown on the page) or
   `not built yet`:

| Lab reports | Self-reports |
|---|---|
| Look for subgroups of the patient group picked, the first if both are (`cluster_patients`, `link_subgroups`); refused with two control groups | Typed-in lab values within reference ranges |
| Compare the two groups (`run_analysis`, with the subgroup links in one correction) | Drug and symptom profile (`profile_drugs`) |
| Compare PSSD, PFS and PAS (`compare_syndromes`; the reference is the control picked, else the reference ranges) | Symptom subgroups (`cluster_symptoms`) |
| Positions within reference ranges (`lab_range_findings`) | Intervention outcomes (`rank_interventions`) |

Each step's one-line statement is shown, and when the run finishes, its findings
(`GET /api/findings`, built by `asclepius.web.findings`), one section per step:

4. **Findings.** Each finding shows its sample sizes (values in each group), its effect size
   (Cliff's delta) with its 95% interval, its q-value, its label (strong, suggestive,
   noise-level) and its Tier A result (holds, does not hold, or too small to test, with the Tier A
   sizes), so a weak finding cannot pass for a strong one. Tables sort by strength (then effect
   size) or by effect size. A run, or a step, with no strong or suggestive finding says
   **Nothing reliable yet** in a box above it (S8). Each finding has a plot:
   - group comparisons, ratios and subgroup symptom scores: each group's 10th, 25th, 50th, 75th
     and 90th percentiles as a box plot, for groups of 10 or more; no single value is drawn (S11);
   - drug and symptom profile and symptom subgroups: the share of each side with the outcome
     (reports the symptom, took a drug or class, onset during use, a trajectory), hidden with the
     odds ratio and effect when a cell is below 5;
   - subgroup drugs: the share of each cluster that took the drug (hidden when fewer than 5 on a
     side took it, or did not, and then the effect too: on yes/no values it is the difference of
     the shares, which with the sizes would give the count back; the Tier A effect of a drug link
     is always hidden);
   - reference-range positions (lab and self-report) and syndromes against the reference ranges:
     how many values fall below, within and above the range, or are unclear. Counts only; a
     count of 1 to 4 is hidden, and so is the next smallest when a total would give it back.
     These rows describe and are not tests, so their label is "description, not a test". A
     syndrome with 1 to 4 values within the range has its shares, effect and plot hidden.

   The subgroup section shows the clusters and their sizes per sex (a cluster with a sex hidden
   hides the other too, since the links show its total), and their links to symptom scores and
   drugs, labelled in the run's one correction; when the comparison does not finish, the links
   have no label and are not shown. The cross-syndrome section adds a summary of
   the markers with a pattern (shared, same direction with a different size, differs, one
   syndrome only), then every test. Every self-report section carries "self-report, no
   controls"; the drug and symptom profile adds authors per drug class and drug, and that an
   author counts in each of their classes. Symptom subgroups show the clusters and their sizes
   (or "no stable clusters" and why), how many authors were left out, each cluster's most
   over-represented symptoms, and the clusters compared by drug class, single drug, onset and
   trajectory. Intervention outcomes show the ranked table and the safety signals, with how many
   interventions were too few to rank (the export lists them). Every count below 5 is hidden in the view (0, which
   is no one, is shown).

5. **Export.** Once a run has finished, two downloads take it into a paper or to a colleague:
   - **Findings (CSV)** (`GET /api/export/findings.csv`): one row per finding the page shows
     (`row` is `finding`), every section in run order (`section` names it, `statement` gives its
     statement, "nothing reliable yet" when it is), with the same columns and the same hidden
     counts. The intervention outcomes ranking has one row per intervention (`row` is
     `intervention`, `comparison` `intervention outcome`, `marker_id` `intervention:<name>`):
     `rank` (empty when not ranked), the outcome counts, the shares improved and worse with their
     intervals, `safety_signal` (`true` or `false`) and `not_ranked`, as shown; the hypothesis
     engine reads them (M4-07). A section with none of these, or that did not finish, has one
     `statement` row saying why. Reference-range counts are `zone_below`, `zone_within`, `zone_above` and
     `zone_unclear`. Every row names the `tool` version, the `dataset_fingerprint`, the
     `run_date` and the run's `settings` (as JSON), and its `labels`: "provisional, needs expert
     review" always, "self-report, no controls" on self-report rows. An empty cell is a hidden or missing value. Statistics are written exactly,
     so `asclepius.web.export.read_findings_csv` reads the file back to the values shown: counts as
     whole numbers, statistics as floats, empty cells as None.
   - **Report (HTML)** (`GET /api/export/report.html`): one self-contained file (no links to the
     internet) with the tool and dataset versions, the labels, every section's statement,
     findings table and plots, and the run record. The record is shown, and embedded as JSON in
     `<script type="application/json" id="run-record">` for a program to read back. It is the
     run's `run_record` (dataset fingerprint and file hashes, tool version and code hash, the
     groups, every setting, the date) plus `steps`: each step's name, state and message, and the
     record of the analysis it ran (the comparison's and the cross-syndrome comparison's, the
     clustering's for subgroups; the reference-range positions, being descriptions, have none).
     Print it from the browser for a PDF.

   Nothing can be exported before a run has finished. A failed step's error text can name a
   file, so the exports say only "Unexpected error; the details are shown in the app only." The
   refusal messages say "fewer than 5" for a group or patient count below 5 (S11), on the page and
   in the exports.

6. **Hypotheses** (milestone 4). Once a run has finished, **Write hypotheses** runs the hypothesis
   engine (`../hypothesis`, an editable dependency of this project) on the run's findings export,
   in the background, with a progress bar: evidence from the paper library for each strong or
   suggestive finding and each ranked intervention, then the LLM set in **Settings**, every request
   checked by the private-text guard first (`hypothesis_engine.pipeline`, which the
   `asclepius-library hypotheses` command also uses). It needs the library and its index
   (`asclepius-library build`, then `index`, in `../hypothesis`); the page says so when they are
   missing. Each hypothesis is shown in the five sections: the finding with its strength label and
   labels, the proposed mechanism, supporting and contradicting papers, and how to test it, with
   any treatment candidates labelled "candidate for study" with their known risks under the
   researchers-only notice. A finding row with a hypothesis links down to it, and the hypothesis
   links back up (`key`, the same text for both). Every citation shows its quote, its paper (title,
   journal, year) with links to PubMed, the open-access full text and the DOI, and the passage it is
   from with the quote marked (`hypothesis_engine.citations.quote_span`). The findings that got no
   hypothesis are listed with why, and the short private-paper passages sent, if any, are shown
   on the page only. **Export hypotheses (HTML)** (`GET /api/export/hypotheses.html`) is one file
   with every hypothesis and its sources, the failures, and the run records, shown and embedded as
   JSON (`id="run-record"`): the analysis run's record, what wrote the hypotheses (library version
   and build date, search model, LLM settings without the key, the date) and the engine's own
   output with each hypothesis's record (prompt version, attempts, citations removed). A new load
   or run drops the hypotheses, which were of the previous one.

   **Settings** (at the top of the page) chooses the LLM: Claude with the researcher's own
   Anthropic API key (model and effort optional), or a local model server (its model name and
   address). It writes the engine's settings file, `hypothesis/data/llm.toml`, readable by its
   owner only, so the app and the command share it; it refuses settings the engine would refuse,
   leaving the file as it was. The key is never sent back to the page: a blank key keeps the
   stored one, and it can be forgotten. Environment variables (`ANTHROPIC_API_KEY`,
   `ASCLEPIUS_LLM_*`) still override the file, and the screen names any that do. **Check** asks
   the model one short question, holding nothing from the dataset, and shows which model answered.

7. **Systems bottleneck map** (milestone 6), at the foot of the page, needs no dataset: the
   knowledge graph is built from the paper library as the app opens (`asclepius.web.graph`,
   `GET /api/graph`), and until then, or without a library, the section says so.
   - **Network.** The top terms by score (40 by default) and the co-mentions among them in at
     least 10 papers (both changeable), drawn with Cytoscape.js, shipped in the app
     (`asclepius/web/vendor/`) so nothing is fetched from the internet. Each syndrome sits at a
     corner of a triangle and each term among the syndromes linking to it: all three in the
     middle, two between their corners, one near its corner, none below. A term's colour says
     how many syndromes link to it (grey none, then light to dark blue); its size, its score. A
     line's width is its papers; a syndrome's literature links are dashed (finding links, solid,
     appear once the app adds a run's findings to the graph).
     Hovering names a term or line ("co-mentioned in N papers").
   - **Clicking a line** shows what it was counted from (`GET /api/graph/edge?a=&b=`): each
     passage (up to 50, then how many more) with its paper's title, journal and year and links to
     PubMed, the full text and the DOI, and a finding link's findings.
   - **Ranked bottlenecks.** Every term's score beside its raw metrics (the syndromes linking to
     it, cascade, betweenness), the top 50 or all; a row picks its term on the network.
   - **Weights.** Three sliders, overlap, cascade and betweenness (45, 35, 20 by default, each
     counting as its share of their sum), re-score and re-rank in the page at once, by the same
     formula as `hypothesis_engine.bottlenecks`; a browser test checks both rank alike.

A run of 100 PSSD, 40 PFS and 100 controls takes about 7 seconds; the full
r/PSSD export loads (3,153 authors) and runs in about 3. The logic is
`asclepius.web.workspace.Workspace` (files, review, run), with the findings view in
`asclepius.web.findings`, the exports in `asclepius.web.export`, the hypotheses in
`asclepius.web.hypotheses`, the settings screen in `asclepius.web.llm_settings` and the bottleneck map in `asclepius.web.graph`; `asclepius.web.server` only serves them.