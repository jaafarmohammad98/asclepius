# hypothesis

The hypothesis engine (milestone 4). So far it builds the local library of papers the engine
is allowed to cite: paper records (title, abstract, identifiers) downloaded from PubMed for
the topics in the plan, the full text of those with an open-access copy, and the
researcher's own private PDFs, and a local search index over it, gathers the passages
it finds for each analysis finding, has the LLM turn them into hypotheses in a fixed
format, checks every quote against the paper it cites, and marks every intervention a
hypothesis names as a candidate for study with its known risks. The analysis web app
(`../analysis`, `uv run asclepius-app`) shows the hypotheses beside the findings and has a
settings screen for the LLM.

```sh
cd hypothesis
uv sync
uv run asclepius-library build    # writes data/library.sqlite: records, then full text, then drug data
uv run asclepius-library drugs    # only the drug step: approved drugs on the top bottleneck targets
uv run asclepius-library fulltext # only the full-text step; continues an interrupted download
uv run asclepius-library refresh  # monthly: new papers and the private folder, then their full text
uv run asclepius-library index    # after each of those: the search index (downloads the model once)
uv run asclepius-library search "allopregnanolone in cerebrospinal fluid" -k 5
uv run asclepius-library evidence findings.csv --out evidence.json  # passages for each finding
uv run asclepius-library hypotheses findings.csv --out hypotheses.json  # evidence, then the LLM
uv run asclepius-library llm-check # checks your LLM key or local model with one short question
uv run pytest                     # tests; they replay recorded PubMed answers, no network
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

`build` writes to the data folder from any folder: `hypothesis/data/` in a source checkout,
a folder of the researcher's own once installed (`hypothesis_engine.home`; `ASCLEPIUS_DATA` names another). It takes `--library PATH` for another file and `--topics PATH` for another topic list
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
  The `hypotheses` command checks every request with one guard: a refused request stops the
  run before it is sent and nothing is written; the passages that were sent are printed at the end.

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
  (spec H2), and neither do reference-range descriptions, which test nothing.
- **Interventions.** The export has one `intervention` row per intervention of the outcomes
  ranking (analysis spec A7): authors per reported outcome, the shares improved and worse with
  their intervals, the rank and the safety flag. These are counts, not tests, so they have no
  strength label; an intervention gets a bundle when the analysis ranked it (at least 10 authors
  with a clear outcome and a share improved that can be shown). Its queries are its name and
  the syndrome (`bupropion improved post-SSRI sexual dysfunction`, `... had no effect on ...`,
  `... worsened sexual dysfunction`), and its bundle carries "reported outcome, not evidence
  that it works" after "self-report, no controls", even if the export lost it.
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

## Hypotheses

`hypotheses` gathers the evidence as `evidence` does, then asks the configured LLM (below)
to turn each bundle into a hypothesis, and writes `{"hypotheses": [...], "failed": [...]}`:

```python
from hypothesis_engine.hypotheses import generate_hypotheses
from hypothesis_engine.library import Library

with Library.open(path) as library:  # every quote is checked against it (below)
    results = generate_hypotheses(bundles, llm, library, index)  # a Hypothesis or a Failure per bundle
results[0].as_dict()  # finding, mechanism, supporting, contradicting, test, candidates, record
```

Every hypothesis has the five sections of the spec:

1. **finding**: written by the engine from the bundle, never by the LLM: one line
   (`Androsterone lower in PSSD than in healthy control (male): effect size -0.90 (95% CI
   -0.98 to -0.79), q 0.023, n 28 and 28; suggestive`), its strength label, its labels, its
   group statistics and the analysis tool, dataset fingerprint and run date it came from.
   A self-report finding's labels start with "based on self-reports, no controls" (spec H7).
2. **mechanism**: claims, each with the id of the bundle passage behind it and a quote.
3. **supporting** and 4. **contradicting**: bundle passages the LLM judged to support or
   contradict, with paper id, title, quote and why. The bundle's stance is only offered: the
   LLM files each passage itself, in one section at most. Contradicting may be empty.
5. **test**: ways to test it, each naming its kind of measure (`lab test`, `physical
   examination`, `imaging`, ..., `self-report`). For a self-report finding at least one
   test must use a measure other than `self-report`.

- **The reply.** The LLM answers in JSON. A reply that is not that JSON, leaves a section
  out or empty, cites a passage not in the bundle, files a passage both ways, names an
  unknown measure, or tests a self-report finding by self-reports only is sent back with
  what was wrong. So is one whose citations fail the check below: it is told which
  citations, their quotes and why, and asked to copy quotes exactly. A reply that passes is
  not sent back. After 3 replies, or on an LLM error, the best usable reply is kept (fewest
  failing citations; the later on a tie). With none, the finding is reported as failed, with
  the reason of each attempt.
- **Citation check** (spec H1, `citations.py`). Every mechanism claim and every supporting or
  contradicting entry must carry a paper id and a quote; the paper must be in the library
  version the hypothesis was built from, and the quote must be found word for word in its
  title or in its text (full text when that version has it, else the abstract). Case, white
  space and hyphenation do not count (`reduc-\ntase`, `post finasteride` or
  `postfinasteride` for `post-finasteride`), but words must be split where the paper splits
  them (`in creased` is not `increased`) and the quote must start and end on whole words.
  Citations of the kept reply that still fail are removed, and `record.citations_removed`
  says why (never with the quote), so no unverified citation is ever written. A hypothesis
  left with no mechanism claim is reported as failed; one left with empty supporting or
  contradicting sections is kept, since every claim it still makes is verified.
- **The record** (spec H6): the model that answered, the back end and its settings (no
  key), the library version, the search model, the analysis tool, `prompt_version`
  (`hypothesis-3`; it changes whenever the prompt does) and how many attempts it took.
- **Treatment candidates** (spec H3). The reply also lists in `candidates` every intervention
  (drug, supplement, procedure, therapy) it names. The engine adds an intervention finding's own
  intervention, and any name it finds in the hypothesis's text (whole words, case aside; the
  finding's causal drug is not looked for) from the export's interventions or from
  `hypothesis_engine/interventions.toml`: about 200 drugs, classes, supplements and therapies
  from the r/PSSD harvest, cleaned by hand. That list leaves out the drugs that cause the
  syndromes, hormones and body chemicals named as markers (estradiol, GABA, ...) and words with
  another meaning ("time", "diet"), which would be matched where no treatment is named. Then a
  second request (prompt `mentions-1`) gives the LLM the hypothesis's sentences and asks for
  every intervention they name as something that could treat or change the condition; a name
  not found word for word in them is sent back, then dropped (`record.mention_check`). A
  hypothesis this check gets no usable answer for is reported as failed, since it may name an
  intervention left unlabelled. Each is written with
  `"label": "candidate for study"` and its **known risks**: the library is searched for
  passages naming it (`adverse effects of bupropion`, `bupropion side effects`, ...; public
  passages only, at most 8), and the LLM lists the risks they state, each with a passage id
  and a quote (prompt `risks-1`). Every quote is checked like any citation, and must also be in
  the passage it cites (the one naming the intervention: elsewhere in the paper it may be
  another drug's risk); failing ones are sent back, then removed (`record.risks_removed`). With
  no risk left, or no passage naming it, `risks_note` is "risks not established in the library";
  when the LLM gave no usable answer it is "risks not checked: ..." with why. An intervention
  in neither list that both LLM requests miss is still not recognised. The prompts say an intervention may be named only as something to study.
- **The notice.** Every candidate, any hypothesis with one, any intervention's hypothesis or
  failure, and the file holding them start with `"notice"`: "For researchers only: treatment candidates
  are ideas for study, not treatment advice. No one should start, stop or change a treatment
  because of them." `Candidate.as_dict`, `Hypothesis.as_dict`, `Failure.as_dict` and `hypotheses.export` add it
  themselves, so nothing holding a candidate can be written without it.
- **What is sent.** The finding's group statistics and labels, and the bundle's public
  passages; for risks, the intervention's name and public passages naming it; for the second
  check, the hypothesis's own sentences (its claims, public quotes and titles, reasons, tests). Private papers' passages are left out of the prompt for now, so a finding whose
  only evidence is private fails without the LLM being asked. A finding with no passage
  at all fails the same way.

The run itself is `hypothesis_engine.pipeline.run_hypotheses`, which the command and the web app
share: it opens the index and library, checks every request with one `PrivateTextGuard`, reports
progress per finding, and returns each bundle's result with `sources`, the text and paper record
(DOI, PMCID, journal, year) of every passage cited, so a reader can be shown the quote in its
passage (`citations.quote_span` finds where) and a link to the paper.

## Co-mentions

The systems analysis (milestone 6) builds its knowledge graph from terms of a curated
vocabulary, `hypothesis_engine/vocabulary.toml`: some 350 HGNC gene symbols (steroid,
neurotransmitter, inflammation, neuropathy, epigenetic, retinoid and metabolic biology) with
their aliases, plus hormones, neurosteroids, receptors, symptoms and the three syndromes.
`library_co_mentions` finds the terms in every passage of the library's PubMed papers and
joins two terms found in the same passage. The researcher's private PDFs are not read: what
enters the graph must trace to a PMID.

```python
from hypothesis_engine.comentions import library_co_mentions
from hypothesis_engine.library import DEFAULT_LIBRARY, Library
from hypothesis_engine.vocabulary import load_vocabulary

with Library.open(DEFAULT_LIBRARY) as library:
    found = library_co_mentions(library, load_vocabulary())
found.edges[0]  # Edge(a, b, papers, passage_ids), label "co-mentioned in N papers"
found.nodes["BDNF"]  # Node(name, kind, papers, passage_ids)
```

- **Matching.** Symbols match case-sensitively and whole ("AR" is not found in "ARX" or
  "Ar"); lower-case words match with case aside, and a hyphen matches any hyphen. The
  abbreviations "PSSD" and "PFS" count only in a paper that also names the syndrome in full, as
  elsewhere they mean post-stroke sexual dysfunction, a nerve-testing device or progression-free
  survival; "PAS" (mostly periodic acid-Schiff staining) is not used. Ambiguous symbols ("AR", "CAT", "REST"...)
  count only with a gene word ("gene", "receptor", "expression", "polymorphism"...) nearby
  (`GENE_WORD_WINDOW` words), so "AR (augmented reality)" is no node and "AR gene expression" is.
- **Edges.** An edge's weight is the number of distinct papers, and it keeps the id of every
  passage it was counted from. It says the terms were mentioned together, never that one
  causes the other.
- **Seeds.** Every term names a library paper in which it is found (`pmid`). A term whose
  paper has left the library, or no longer names it, is left out (`left_out` says why), so
  nothing uncited enters the graph. The full library takes a few seconds.

## The knowledge graph

`library_graph` turns the co-mentions into a directed networkx graph and links the three
syndromes to the terms of their papers. The web app builds it as it opens; it is never stored.

```python
from hypothesis_engine.evidence import read_findings
from hypothesis_engine.graph import library_graph

with Library.open(DEFAULT_LIBRARY) as library:
    graph = library_graph(library, load_vocabulary(), read_findings("findings.csv"))
graph.edges["post-SSRI sexual dysfunction", "BDNF"]  # kind, papers, passage_ids, findings, label
```

- **Co-mentions** are arcs both ways between two terms, `kind` "co-mention".
- **Literature links** go from a syndrome to each term found in a paper its topic search found
  (`PSSD`, `PFS`, `PAS` in `topics.toml`) or in a passage that names the syndrome.
- **Finding links** come from a strong or suggestive finding of an analysis export, for a
  syndrome's group against a comparator that is not another syndrome. They are `kind`
  "finding", stronger than literature, and keep the literature behind them. The terms of a
  finding are those its marker names; an enzyme-activity ratio also names its enzyme's genes
  (`ENZYME_GENES`: 5-alpha reductase is SRD5A1 and SRD5A2).
- Every node and link has passages or findings behind it.

## Bottlenecks

`bottlenecks` ranks the graph's terms by how much they sit where the three syndromes converge.
The syndromes themselves are not ranked.

```python
from hypothesis_engine.bottlenecks import Weights, bottlenecks

ranked = bottlenecks(graph)  # highest score first
ranked[0]  # Bottleneck(name, kind, score, betweenness, syndromes, cascade); .overlap is len(syndromes)
bottlenecks(graph, Weights(overlap=0.45, cascade=0.35, betweenness=0.20))  # the defaults
```

- **Betweenness** is networkx's normalised betweenness centrality on the graph taken as
  undirected, syndromes included, counting hops.
- **Overlap** is which of PSSD, PFS and PAS link to the term, by literature or by a finding;
  a co-mention with a syndrome's name is not a link.
- **Cascade** is downstream reach: over every other term reached by co-mentions, the strength
  of the strongest path there, summed. A path's strength is the product of its edges'
  strengths, an edge's being its papers over the most papers of any co-mention. Paths do not
  pass through a syndrome.
- **Score.** Overlap is scaled over the three syndromes, the other two over the graph's
  highest, and the score (0 to 1) is their weighted sum. The weights are any non-negative
  numbers, used as shares of their sum; the defaults keep a generic hub from outranking a point
  where the syndromes converge. A score says where to look, not what causes what. The built
  library takes a few seconds.

## Drug data

The last step of `build`, and all of `drugs`, looks up the approved drugs acting on the 20 genes
with the highest bottleneck scores and writes them to `data/drugs.json`. The web app only reads
that file (`read_drug_cache`) and never fetches.

```python
from hypothesis_engine.drugs import read_drug_cache

cache = read_drug_cache()  # DrugCache(fetched_at, library_version, targets, drugs)
cache.ranked  # Drug(chembl_id, name, max_phase, properties, actions, known_cause, source, withdrawn, safety)
cache.excluded  # the drugs that cause the syndromes, never ranked
drug.safety  # Safety(label, risks, generic_applications, unchecked); .generic, .concerns, .penalty
```

- **ChEMBL** is asked, per gene symbol, for the human targets with that gene among their
  components (the protein, and the families and complexes it is in), the drug mechanisms on
  them, and those molecules. Only `max_phase` 4 is kept; a salt counts as its parent molecule.
- **MyChem.info** answers for a gene ChEMBL fails on (ChEMBL often answers HTTP 500 for a
  while): the molecules whose DrugCentral bioactivities name the gene as a mechanism of action.
- **openFDA** gives each drug's US label (the latest brand-name one, else the latest of any)
  and its Drugs@FDA applications. The label's boxed warning, warnings, precautions and adverse
  reactions are read for sexual dysfunction, libido, anhedonia and emotional blunting; each
  concern found keeps the label text it was found in, its section and the label's DailyMed set
  id. A concern its clause denies ("was not observed") or more desire ("increased libido") is
  not counted. `generic` is true when an ANDA of the drug has a product still marketed. A label
  or application counts when its ingredients are the drug's: a salt, hydrate or dosage form
  counts, another word ("ETHINYL ESTRADIOL" for estradiol) or a combination does not. Up to 20
  labels are looked through for one of the drug's own.
- **Unchecked**: a drug openFDA cannot answer for (an unreadable answer, an HTTP error other than
  "no match") is kept with `unchecked` saying why, and `generic` and `penalty` None; the rest
  are still asked. openFDA down stops the fetch and leaves the cache as it was.
- **Risks** listed for a drug in the `[risks]` table of `interventions.toml`, each with a PMID,
  DOI or https source, are merged in. `Safety.penalty` counts the label concerns and those risks:
  the safety criterion of the ranking (M6-06). It is None, unknown rather than safe, for a drug
  with no US label or unchecked.
- **Sent** is only the gene symbol, ChEMBL ids and, to openFDA, the drug's name, each request
  through the privacy guard first. No account, key or email. At most 3 requests a second to each service; HTTP 429, 5xx
  and dropped connections are sent again.
- **Properties** for the CNS MPO estimate: `alogp`, `mw_freebase`, `psa`, `hbd`, `hba`.
  ChEMBL 36 no longer gives the ChemAxon `cx_logp`, `cx_logd` and `cx_most_bpka`, kept as None.
- **Known causes** (SSRIs, SNRIs, finasteride, dutasteride, isotretinoin) are tagged with what
  they are, by any word of the name or ATC class N06AB, and listed apart. A drug withdrawn
  since its approval (alpidem, sibutramine) is kept and flagged `withdrawn`.
- **A failure** leaves the previous `drugs.json` as it was; `asclepius-library drugs` fetches
  again.

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

## Privacy test

Only group-level findings are sent to the LLM (spec H5). `tests/test_outgoing_requests.py`
runs the `hypotheses` command on the findings the analysis app exported from two synthetic
datasets, a lab-report one and a self-report one, with a private paper in the library and a
fake model that answers every kind of request usably, so each is sent. Through the request
hook it checks that no request holds a record id, case id, author key or individual value of
either dataset, nor any run of the private paper's words, and that a request carrying more of
the private paper than short quotes is refused before it reaches the model. A test adds one
record's id and value to the prompt and confirms the check finds them. It looks for ids and
lab values exactly as the datasets hold them: a value rounded before it is sent, or a field
too ordinary to search for (an age, a dose, a month), would not be caught.

To make a leak show, the datasets' record ids read `pt-...-x` and every lab value carries five
digits of its own after the decimal point (`1140.0000123`); the lab groups are of even size,
so no median is one record's value. They are in `tests/fixtures/privacy/` with their findings
exports; after a change to either engine's findings, write them again with the analysis
engine's environment: `uv run --project ../analysis python tests/privacy_fixture.py`. The
analysis engine's suite does the same into a temporary folder
(`analysis/tests/test_privacy_fixture.py`) and fails when the result differs from the committed
files (the run date aside), so a change to its export cannot leave them stale.

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
both recorded 2026-11-22. `tests/fixtures/drugs/` holds ChEMBL answers for SRD5A2, SLC6A4,
TSPO and AR and a MyChem.info answer for SRD5A2 (`tests/drugs_fixture.py`), recorded 2026-11-24. To
record them again (needs the network): `uv run python tests/record_responses.py build`,
`... fulltext` or `... drugs --fresh` (without `--fresh` it keeps what is recorded and asks only
for the rest), then update the values in `tests/test_build.py`, `tests/test_fulltext.py`,
`tests/fulltext_fixture.py` and `tests/test_drugs.py`.

An older library (schema 1 or 2) is upgraded in place by `build`, `fulltext` or `refresh`;
text downloaded before versions covered it belongs to the version it was downloaded under.
