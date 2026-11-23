# hypothesis

The hypothesis engine (milestone 4). So far it builds the local library of papers the engine
is allowed to cite: paper records (title, abstract, identifiers) downloaded from PubMed for
the topics in the plan, the full text of those with an open-access copy, and the
researcher's own private PDFs, and a local search index over it, and gathers the passages
it finds for each analysis finding. Hypotheses come in later issues.

```sh
cd hypothesis
uv sync
uv run asclepius-library build    # writes data/library.sqlite: records, then full text; prints coverage
uv run asclepius-library fulltext # only the full-text step; continues an interrupted download
uv run asclepius-library refresh  # monthly: new papers and the private folder, then their full text
uv run asclepius-library index    # after each of those: the search index (downloads the model once)
uv run asclepius-library search "allopregnanolone in cerebrospinal fluid" -k 5
uv run asclepius-library evidence findings.csv --out evidence.json  # passages for each finding
uv run asclepius-library llm-check # checks your LLM key or local model with one short question
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
  `hypothesis_engine.privacy.PrivateTextGuard.check(request)`, one guard per run. It finds
  the passages the request shares with a private paper's title or text (8 or more words in
  a row, or the whole of a shorter paper; runs at most 3 words apart are one passage). It
  refuses the request when one passage is over 60 words, all of them together are over 200,
  or more than 400 different words of one paper would have been sent during the run.
  Otherwise it returns the passages so they can be shown to the researcher before sending.
  It catches private text slipping into a request by mistake, not text disguised on purpose.

## Search

`index` splits every paper of the library's current version into passages and indexes them
twice in `hypothesis/data/index.sqlite`: by keyword (SQLite FTS5, BM25, English stemming)
and by meaning (vectors from a local embedding model). `search` ranks the passages both ways
and merges the rankings (reciprocal rank fusion), so a passage both find comes first.

```python
from hypothesis_engine.search import search

search("lower neurosteroids after finasteride", k=10)
# [Hit(passage_id, paper_id, private, title, text, score, keyword_rank, meaning_rank)]
```

To search many times, open an `Index` once (`Index.open(index, library, LocalEmbedder())`);
`index.passage("32317131:4")` reads one passage back.

- **Passages.** Each line of a paper's text (a full-text paragraph, a section of a structured
  abstract) starts one; one over 200 words is cut at sentence ends, and one under 25 words (a
  heading) joins the next. A passage is a slice of the text character for character, so a quote
  from it is found word for word in the paper. Its id is the paper's id and its number,
  `32317131:4`; the same text always gives the same ids. A paper with no abstract is one
  passage, its title. Private papers are indexed too, and their hits are marked `private`.
- **The model.** [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT
  licence, 64 MB), run on the CPU with ONNX Runtime through
  [fastembed](https://github.com/qdrant/fastembed). `index` downloads it from Hugging Face into
  `hypothesis/data/models/` the first time; after that nothing goes over the network, and
  `search` never downloads it. It embeds about 28 passages a second on a laptop CPU, so the
  first index of the full library (some 40,000 passages) takes about 25 minutes. Rebuilds embed
  only passages not embedded before, so after a refresh it is quick.
- **Versions.** The index records the library version and model it was built from, and
  `search` refuses an index of another version: run `index` again after `build`, `fulltext`
  or `refresh`. A new index replaces the old one only when complete.
- **Relevance test.** `tests/test_relevance.py` asks eight questions, worded unlike the
  papers, of twelve real papers (the recorded fixtures) with the network cut off, and checks
  that each finds its known paper in the top three. It is skipped until the model is
  downloaded.

## Evidence for each finding

`evidence` takes the analysis web app's findings export (`findings.csv`, from a lab-report or
a self-report run) and writes one evidence bundle per finding worth testing, before any LLM is
asked anything:

```python
from hypothesis_engine.evidence import gather_evidence, read_findings

bundles = gather_evidence(read_findings("findings.csv"), index, k=5)  # index: an open Index
bundles[0].as_dict()  # finding, labels, queries, supporting, contradicting, versions
```

- **Which findings.** Strong and suggestive ones only. Noise-level findings get no bundle
  (spec H2), and neither do reference-range descriptions, which test nothing. Intervention
  outcomes are counts, not tests, and are not in the findings export, so they get none either.
- **Group statistics only.** A bundle holds the finding's names (marker, groups, sex), its
  group sizes, medians or shares, effect size and interval, p and q values and Tier A
  re-run, exactly as the export shows them (counts below 5 already hidden there). No other
  column of the file is read, so a record id or an individual value cannot reach a bundle.
- **Labels.** The export's labels are kept. A self-report finding's bundle has
  `self_report` set and "self-report, no controls" first among its labels (spec H7).
- **Queries.** Built from names and directions, never numbers: the marker on its own and in
  the syndrome (`lower Androsterone in post-SSRI sexual dysfunction than in healthy controls`),
  after the syndrome's drug, the enzyme a ratio stands for (`lower 5-alpha reductase
  activity in ...`; free cortisol / free cortisone is read as 11-beta-HSD2 the other way),
  and for self-report findings the drug and the symptom, onset or course (`depression more
  often after paroxetine than after other drugs`, `persistent depression after stopping
  paroxetine`; an onset or course reads `symptoms recovered more often after ...`, with no
  "persistent" search). Group names that name no one ("cluster 2", "other causal drugs") are left out.
- **Supporting and contradicting.** Every query has a stance: the finding's direction is
  supporting, the opposite direction or "no difference" contradicting. A passage is filed
  under the stance of the search that ranked it best (on a tie, contradicting), and offered
  once. These are candidates: the stance is a guess from the wording of a search, and the
  next step (M4-05b) has the LLM judge each passage. Each passage carries its id, paper id,
  title, text, whether it is private, and the search that found it. A paper known only by
  its title is not offered, since it has nothing to quote.
- **Versions.** Each bundle records the analysis tool, the dataset fingerprint and run
  date, the library version and the search model.

## The LLM

The engine reaches an LLM only through `hypothesis_engine.llm`, and the researcher pays for
it: the maintainer runs no server and holds no key. One call, two back ends:

```python
from hypothesis_engine.llm import connect

llm = connect()  # settings from the environment, then data/llm.toml
answer = llm.complete([{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
answer.text, answer.record()  # record: backend, model that answered, settings, stop reason, tokens
```

- **Anthropic (Claude), with your own key.** Set `ANTHROPIC_API_KEY`, or put `api_key`
  under `[anthropic]` in `hypothesis/data/llm.toml` (git ignores `data/`). The model is
  `claude-opus-5-5` unless you name another. Each request asks for Anthropic's default
  server-side fallback: a request the model declines for policy reasons is answered by
  another model in the same call (models without fallbacks just answer alone). The record
  names the model that actually answered. A request still declined raises `LLMError`.
- **A local model.** Run a server that speaks the OpenAI-style chat protocol
  (`/v1/chat/completions`): [Ollama](https://ollama.com), llama.cpp's `llama-server`,
  LM Studio. Nothing then leaves the machine.

Each back end keeps its own table, so switching is a change of `backend` alone.
`hypothesis/data/llm.toml`:

```toml
backend = "local"      # or "anthropic" (the default when a key is set)
max_tokens = 16000     # the default

[anthropic]
api_key = "sk-ant-..." # or ANTHROPIC_API_KEY
model = "claude-opus-5-5"
effort = "high"        # optional: low, medium, high, xhigh, max

[local]
model = "llama3.1:8b"  # the name your server knows the model by
url = "http://localhost:11434/v1"  # Ollama's address, the default
```

Environment variables override the file: `ASCLEPIUS_LLM_BACKEND`, `ASCLEPIUS_LLM_MODEL` (the
chosen back end's model), `ASCLEPIUS_LLM_URL` and `ANTHROPIC_API_KEY`. With neither a key
nor a local back end set, `connect()` raises `LLMNotConfigured`, which says how to set one
up; a misspelt setting or an unknown `effort` is refused the same way.

- **The key** is never printed or stored by the engine: it is left out of the settings'
  repr, of `answer.record()`, of recorded requests and of error messages (which also drop
  the SDK's exception, since it holds the request headers).
- **The request hook.** `connect(on_request=f)` calls `f(OutgoingRequest(backend, url, body))`
  with a copy of every request just before it is sent; raising stops the request. Tests use it to see
  exactly what would leave the machine, and the privacy checks (`PrivateTextGuard`, M4-08)
  hook in there. The body is the JSON sent; headers, and so the key, are not part of it.

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
