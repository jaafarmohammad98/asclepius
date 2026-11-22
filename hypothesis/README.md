# hypothesis

The hypothesis engine (milestone 4). So far it builds the local library of papers the engine
is allowed to cite: paper records (title, abstract, identifiers) downloaded from PubMed for
the topics in the plan, the full text of those with an open-access copy, and the
researcher's own private PDFs. Hypotheses come in later issues.

```sh
cd hypothesis
uv sync
uv run asclepius-library build    # writes data/library.sqlite: records, then full text; prints coverage
uv run asclepius-library fulltext # only the full-text step; continues an interrupted download
uv run asclepius-library refresh  # monthly: new papers and the private folder, then their full text
uv run pytest                     # tests; they replay recorded PubMed answers, no network
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

`build` writes to `hypothesis/data/` from any folder. It takes `--library PATH` for another file and `--topics PATH` for another topic list
(`refresh` too), and `--private PATH` for another private folder.
It needs no account. With an NCBI API key in `NCBI_API_KEY` (free, from an NCBI account) it
sends 10 requests a second instead of 3. No email or other detail about the researcher is sent.

## Full text

After the records, `build` asks [Europe PMC](https://europepmc.org/RestfulWebService) which
papers have a copy in the open-access subset (papers whose licence allows reuse), 100 PMIDs
per request, and stores each copy's PMCID and licence as Europe PMC gives it (`cc by`,
`cc by-nc-nd`...). It then downloads those copies' JATS XML from PubMed Central (NCBI
efetch, 20 per request, at the PubMed rate limit) and keeps the readable text: title,
abstract, then the body by section, one paragraph per blank-line-separated block, with
tables, formulas, the reference list and footnotes left out. Papers that are only free to
read, not open access, are not downloaded. When Europe PMC gives no licence, the one the
XML states is kept (for old BioMed Central papers, their copyright statement).

Why two services: NCBI's own open-access lookup no longer answers (HTTP 404), and Europe
PMC's full-text download failed with HTTP 500 for most open-access copies on 2026-11-22.

It ends with a line like `full text for 732 of 3784 papers (19%); 733 have an open-access
copy, 1 of them without readable text` (the real numbers on 2026-11-22). A copy is "without
readable text" when PMC sends it without a body: some publishers do not allow the XML.
The text a run downloads joins the library as one new version at the end of the run.

Formulas keep their MathML symbols in the text (not their TeX copies), so a sentence reads as
in the paper; a list or figure inside a paragraph splits it where it stands. Only the text is
stored, not the XML: changing how text is extracted means downloading again.

Every lookup and every batch of downloads is stored as it arrives, so an interrupted run
(Ctrl-C, a crash, lost network) loses nothing: `asclepius-library fulltext` continues with
what is left. A request that fails for a moment is retried three times. A download batch
that still fails, or a copy missing from PMC's answer, is left for the next run, and three
batches failing in a row stop the run; a lookup that still fails stops it at once. A paper
found without a copy is looked up again after 30 days, since papers become open access
later. Europe PMC gets at
most 3 requests a second (it publishes no limit). No request carries any detail about the
researcher. The full-text step takes about 10 minutes for the shipped topics, most of it
the Europe PMC lookups.

## Monthly refresh

`refresh` runs the topic searches again and fetches only the papers the library has no
record of; stored records are not downloaded again. It reads the private folder, stores
what is new as one new library version, then downloads the full text still missing. A
paper a search no longer finds stays in the library. When nothing is new it adds no
version, so running it twice in a row adds nothing the second time:
`nothing new for data/library.sqlite: no new papers, 2 private unchanged`.

## Private papers

Put your own PDFs (unpublished results, preprints, papers you have a copy of) in
`hypothesis/data/private/` and run `refresh` (or `build`). Their text is extracted on this
machine with [pypdf](https://pypdf.readthedocs.io) and stored in the library marked as
private. Git ignores the folder and every PDF.

- A paper's id is `private-` plus 16 hex digits of the SHA-256 of its text: renaming the
  file keeps it, changing the words gives a new one. Its title is the PDF's title, or its
  first line.
- A scanned PDF with no text layer, or one pypdf cannot read, is skipped and named in the
  output with the reason. There is no OCR.
- A PDF removed from the folder leaves the next version; versions that held it can still
  read it. The same text under two names is one paper.
- Private papers are never looked up or downloaded anywhere: requests to PubMed and Europe
  PMC carry PMIDs only, and private papers have none.
- Anything the engine sends to an LLM (from M4-04) must first pass
  `hypothesis_engine.privacy.PrivateTextGuard.check(request)`. It finds the runs of 8 or
  more words the request shares with a private paper's title or text, refuses the request when one run is
  over 60 words or all of them together are over 200, and otherwise returns those passages
  so they can be shown to the researcher before sending.

## Topics

`hypothesis_engine/topics.toml` holds one PubMed search per topic, and how many of its most
relevant results to keep: PSSD, PFS, PAS, neurosteroids, steroid metabolism enzymes,
serotonin and sexual function, small-fibre neuropathy, epigenetic effects of drugs. No
acronym is searched alone: in PubMed "PFS" mostly means progression-free survival and "PSSD"
also names a nerve-testing device.

## The library file

One SQLite file. Each build is written in one transaction and becomes the next **library
version** (1, 2, 3...). A build that fails part-way changes nothing. The version goes up on
every change: a build, a refresh that found new papers or a change in the private folder,
and a full-text run that stored new text. Papers are never
deleted: `paper_topics` records which papers each version held, so the papers an older
output was allowed to cite can still be read (`library.paper(pmid, version=1)`). A paper's
record is the latest one fetched.

| Table | Holds |
|---|---|
| `builds` | `version`, `built_at` (ISO 8601, UTC), `topics_sha256` (the exact searches run), `papers` |
| `papers` | `pmid`, `title`, `abstract`, `has_abstract`, `doi`, `pmcid`, `journal`, `year`, `first_version` |
| `paper_topics` | `version`, `pmid`, `topic`, `rank` (1 = that search's most relevant result) |
| `open_access` | `pmid`, `checked_at`, `pmcid` (NULL: no open-access copy), `licence`, `fetched_at` (NULL: not downloaded yet), `text` (NULL after a download: no readable text), `version` (the first version holding the text) |
| `private_papers` | `id`, `file_name`, `title`, `text`, `first_version` |
| `private_versions` | `version`, `id` |

Every paper has a PMID and a title. `abstract` is empty (NULL) when PubMed has none, which
is common for letters and news items, and `has_abstract` is 0 for those. A structured
abstract keeps its section labels (`METHODS: ...`), one section per line. Book chapters
(StatPearls and others) are kept, with the book's title as `journal`.

Code reads it through `hypothesis_engine.library.Library`:

```python
with Library.open(Path("data/library.sqlite")) as library:
    library.version, library.built_at, library.count(), library.lacking_abstract()
    library.paper("32317131")  # Paper or None
    library.topics("32317131")  # topics whose search found it
    library.full_text("32317131")  # FullText(pmid, pmcid, licence, text) or None
    library.coverage()  # Coverage(papers, checked, open_access, with_text, unreadable, to_download)
    library.private_papers()  # [PrivatePaper(id, file_name, title, text)]
    library.documents()  # every paper as Document(id, title, text, private); text: full text or abstract
    library.find("allopregnanolone was lower")  # ids of papers holding that passage word for word
```

## Recorded responses

`tests/fixtures/build/` holds real PubMed answers for a three-topic search, and
`tests/fixtures/fulltext/` real Europe PMC and PubMed Central answers for five papers
(`tests/fulltext_fixture.py`; the three articles are CC BY or CC BY-NC-ND, kept unchanged),
both recorded 2026-11-22. To record them again (needs the network):
`uv run python tests/record_responses.py build` or `... fulltext`, then update the values in `tests/test_build.py`,
`tests/test_fulltext.py` and `tests/fulltext_fixture.py`.

An older library (schema 1 or 2) is upgraded in place by `build`, `fulltext` or `refresh`;
text downloaded before versions covered it belongs to the version it was downloaded under.
