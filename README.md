# Asclepius

Open-source tools for researchers studying post-SSRI sexual dysfunction (PSSD) and related post-drug syndromes. The goal is to turn patients' lab reports into a clean, anonymised dataset, find statistically honest patterns in it, and produce cited, testable hypotheses.

Asclepius is a research tool. It does not diagnose and does not give treatment advice.

## Status

| Part | What it does | State |
|---|---|---|
| `extractor/` | Reads DUTCH hormone report PDFs into anonymised records | Working on the classic report layout; see limits below |
| Analysis engine | Group comparisons, enzyme-activity ratios, subgroups | Not started |
| Hypothesis engine | Literature search plus LLM reasoning, with verified citations | Not started |
| Patient platform | Consented upload and symptom questionnaire | Not started |

## Extractor

Reads a DUTCH report (Precision Analytical) with ordinary code. No AI model is involved, so it cannot invent a number.

```sh
cd extractor
npm install
npm run extract -- path/to/report.pdf            # JSON
npm run extract -- --csv a.pdf b.pdf > data.csv   # one row per marker
```

It also runs in the browser. `npm run build` writes a static page to `extractor/dist/` (`npm run dev` serves it while developing). The page reads the PDF on the user's device: the pdf.js worker is bundled with it, and it makes no network request once it has loaded.

**What it keeps:** sex, age in years, the month of collection, the month of the report, and every result row (marker, value, unit, reference range, the lab's verdict).

**What it never keeps:** name, date of birth, address, accession number, ordering physician, exact dates.

**How it checks itself:** each record lists consistency checks. The report's printed totals must equal the sum of their components (for example, 24-hour free cortisol against the individual samples), and each value must agree with the lab's own verdict beside it. A misread or edited number fails a check and appears under `warnings`.

### Known limits

- Tested only on three public sample reports dated 2018 to 2021. Accuracy on real patient reports has not been measured yet.
- The 2025 layout (report version 1.6.0) is not supported. In the one public sample of it, the result tables are not present as text, so the extractor reports that it found no tables.
- Scanned or photographed reports are not supported.
- DUTCH Plus, Cycle Mapping and other variants are untested.

### Development

```sh
npm run samples     # download the public sample reports (not committed; the lab owns them)
npm test            # includes a check of the built page in Google Chrome, skipped if Chrome is absent
npm run typecheck
npm run accuracy    # compare the extractor with human readings; prints an overall percentage
```

`npm run accuracy` looks under `samples/` for every folder that holds both `report.pdf` and `expected.json`, at any depth. `expected.json` lists what a person read from the pages, one entry per result row:

```json
{ "markers": [
  { "id": "testosterone", "value": 46.8, "unit": "ng/mg", "range": "25 - 115" }
] }
```

Use the printed `name` instead of `id` if you prefer, `"range": null` when no range is printed, and `"comparator": "<"` when the lab printed one. A marker printed in two tables is listed once. A value counts as matched only when its number, unit and range all equal the reading; the script also lists values the extractor missed, got wrong, or added.

## Licence

AGPL-3.0-or-later. Anyone who modifies and deploys this code must keep their version open.
