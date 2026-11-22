# hypothesis

The hypothesis engine (milestone 4). So far it builds the local library of papers the engine
is allowed to cite: paper records (title, abstract, identifiers) downloaded from PubMed for
the topics in the plan. Hypotheses come in later issues.

```sh
cd hypothesis
uv sync
uv run asclepius-library build    # writes data/library.sqlite (about 2 minutes, ~3,800 papers)
uv run pytest                     # tests; they replay recorded PubMed answers, no network
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

`build` writes to `hypothesis/data/` from any folder. It takes `--library PATH` for another file and `--topics PATH` for another topic list.
It needs no account. With an NCBI API key in `NCBI_API_KEY` (free, from an NCBI account) it
sends 10 requests a second instead of 3. No email or other detail about the researcher is sent.

## Topics

`hypothesis_engine/topics.toml` holds one PubMed search per topic, and how many of its most
relevant results to keep: PSSD, PFS, PAS, neurosteroids, steroid metabolism enzymes,
serotonin and sexual function, small-fibre neuropathy, epigenetic effects of drugs. No
acronym is searched alone: in PubMed "PFS" mostly means progression-free survival and "PSSD"
also names a nerve-testing device.

## The library file

One SQLite file. Each build is written in one transaction and becomes the next **library
version** (1, 2, 3...). A build that fails part-way changes nothing. Papers are never
deleted: `paper_topics` records which papers each version held, so the papers an older
output was allowed to cite can still be read (`library.paper(pmid, version=1)`). A paper's
record is the latest one fetched.

| Table | Holds |
|---|---|
| `builds` | `version`, `built_at` (ISO 8601, UTC), `topics_sha256` (the exact searches run), `papers` |
| `papers` | `pmid`, `title`, `abstract`, `has_abstract`, `doi`, `pmcid`, `journal`, `year`, `first_version` |
| `paper_topics` | `version`, `pmid`, `topic`, `rank` (1 = that search's most relevant result) |

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
```

## Recorded responses

`tests/fixtures/build/` holds real PubMed answers for a three-topic search, recorded
2026-11-22. To record them again (needs the network): `uv run python tests/record_responses.py`,
then update the PMIDs at the top of `tests/test_build.py`.
