"""The local search index: find the library passages relevant to a finding, on this machine.

`build_index` splits every paper of the library's current version into passages
(`passages.py`) and writes them to a second SQLite file with two indexes: a keyword index
(SQLite's FTS5, BM25 ranking, English stemming, so "persisting" finds "persisted") and a
meaning index (each passage's vector from a local embedding model, `embedding.py`).
`Index.search(query, k)` ranks the passages both ways and merges the two rankings by
reciprocal rank fusion: a passage's score is the sum of 1 / (60 + its rank) in each, so a
passage both find comes before one only one finds. Nothing here uses the network.

The index records the library version and the model it was built from, and refuses to open
for another (`asclepius-library index` rebuilds it). A rebuild embeds only passages it has
not embedded before with the same model, so after a refresh it is quick.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Protocol

import numpy as np

from hypothesis_engine.library import DEFAULT_LIBRARY, Library
from hypothesis_engine.passages import Passage, split

DEFAULT_INDEX = DEFAULT_LIBRARY.with_name("index.sqlite")

SCHEMA_VERSION = 1
FUSION_K = 60  # the usual constant for reciprocal rank fusion
EMBED_BATCH = 64

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE passages (
    n INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    paper_id TEXT NOT NULL,
    private INTEGER NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    sha256 TEXT NOT NULL,  -- of the model's name and the text it embedded
    vector BLOB NOT NULL  -- float32, unit length
);
CREATE VIRTUAL TABLE keywords USING fts5(
    title, text, content = 'passages', content_rowid = 'n', tokenize = 'porter unicode61 remove_diacritics 2'
);
"""
_WORDS = re.compile(r"\w+")


class Embedder(Protocol):
    """A local model that turns text into unit vectors; passages and queries may be embedded differently."""

    @property
    def name(self) -> str: ...

    def passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def query(self, text: str) -> np.ndarray: ...


class SearchIndexError(Exception):
    """No index at that path, or it does not match the library or the model."""


@dataclass(frozen=True)
class Hit:
    passage_id: str
    paper_id: str  # a PMID, or a private paper's id
    private: bool
    title: str
    text: str
    score: float  # reciprocal rank fusion of the two rankings
    keyword_rank: int | None  # 1 = best; None when the keyword search did not return it
    meaning_rank: int | None


@dataclass(frozen=True)
class IndexReport:
    version: int  # the library version indexed
    passages: int
    embedded: int  # passages newly embedded; the others were reused from the previous index


def _embedding_text(passage: Passage) -> str:
    return f"{passage.title}\n{passage.text}"


def _sha(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()


def _reusable(path: Path, model: str) -> dict[str, bytes]:
    """The vectors of the index at `path` made with `model`, by sha256; none if it cannot be read."""
    if not path.is_file():
        return {}
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = db.execute("SELECT value FROM meta WHERE key = 'model'").fetchone()
            if row is None or row[0] != model:
                return {}
            return dict(db.execute("SELECT sha256, vector FROM passages"))
        finally:
            db.close()
    except sqlite3.DatabaseError:
        return {}


def build_index(
    library: Path,
    index: Path,
    embedder: Embedder,
    progress: Callable[[str], None] = lambda _message: None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> IndexReport:
    """Index every passage of the library's current version at `index`, replacing what was there.

    The new index is written next to the old one and takes its place only when complete.
    """
    with Library.open(library) as opened:
        version = opened.version
        passages = [p for document in opened.documents() for p in split(document)]
    known = _reusable(index, embedder.name)
    shas = [_sha(embedder.name, _embedding_text(p)) for p in passages]
    vectors = {sha: known[sha] for sha in shas if sha in known}
    missing = [i for i, sha in enumerate(shas) if sha not in vectors]
    progress(f"{len(passages)} passages, {len(missing)} to embed")
    for start in range(0, len(missing), EMBED_BATCH):
        batch = missing[start : start + EMBED_BATCH]
        embedded = embedder.passages([_embedding_text(passages[i]) for i in batch])
        for i, vector in zip(batch, embedded, strict=True):
            vectors[shas[i]] = np.asarray(vector, dtype=np.float32).tobytes()
        if start and start % (EMBED_BATCH * 20) == 0:
            progress(f"embedded {start} of {len(missing)}")

    index.parent.mkdir(parents=True, exist_ok=True)
    partial = index.with_name(index.name + ".partial")
    partial.unlink(missing_ok=True)
    db = sqlite3.connect(partial)
    try:
        with db:
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    db.execute(statement)
            db.executemany(
                "INSERT INTO meta VALUES (?, ?)",
                [
                    ("library_version", str(version)),
                    ("model", embedder.name),
                    ("built_at", now().isoformat(timespec="seconds")),
                ],
            )
            db.executemany(
                "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (n, p.id, p.paper_id, int(p.private), p.title, p.text, sha, vectors[sha])
                    for n, (p, sha) in enumerate(zip(passages, shas, strict=True), start=1)
                ],
            )
            db.execute("INSERT INTO keywords (keywords) VALUES ('rebuild')")
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    finally:
        db.close()
    os.replace(partial, index)
    return IndexReport(version, len(passages), len(missing))


def _keyword_query(query: str) -> str:
    """Any of the query's words, each quoted so FTS5 reads none of them as an operator."""
    return " OR ".join(f'"{word}"' for word in _WORDS.findall(query.casefold()))


class Index:
    """A built index, opened for searching. Open with `Index.open(...)`; use as a context manager."""

    def __init__(self, db: sqlite3.Connection, embedder: Embedder) -> None:
        self._db = db
        self._embedder = embedder
        meta = dict(db.execute("SELECT key, value FROM meta"))
        self.library_version = int(meta["library_version"])
        self.model = meta["model"]
        self.built_at = meta["built_at"]
        rows = db.execute("SELECT n, vector FROM passages ORDER BY n").fetchall()
        self._numbers = np.array([n for n, _ in rows], dtype=np.int64)
        self._vectors = (
            np.stack([np.frombuffer(v, dtype=np.float32) for _, v in rows]) if rows else np.zeros((0, 0))
        )

    @classmethod
    def open(cls, path: Path, library: Path, embedder: Embedder) -> Index:
        """Open the index at `path`; refused unless it was built from the library's current
        version with this embedder's model."""
        rebuild = "run `asclepius-library index`"
        if not path.is_file():
            raise SearchIndexError(f"no search index at {path}; {rebuild}")
        with Library.open(library) as opened:
            current = opened.version
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            (schema,) = db.execute("PRAGMA user_version").fetchone()
            if schema != SCHEMA_VERSION:
                raise SearchIndexError(f"{path} is not a search index of this version; {rebuild}")
            index = cls(db, embedder)
        except sqlite3.DatabaseError as error:
            db.close()
            raise SearchIndexError(f"{path} is not a search index ({error}); {rebuild}") from error
        except SearchIndexError:
            db.close()
            raise
        if index.library_version != current:
            index.close()
            raise SearchIndexError(
                f"{path} indexes library version {index.library_version}, the library is at {current}; {rebuild}"
            )
        if index.model != embedder.name:
            index.close()
            raise SearchIndexError(f"{path} was built with {index.model}, not {embedder.name}; {rebuild}")
        return index

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The `k` passages most relevant to `query`, best first."""
        match = _keyword_query(query)
        if not match or k < 1 or not len(self._numbers):
            return []
        depth = max(50, k * 5)  # how far down each ranking to look
        by_keyword = [
            n
            for (n,) in self._db.execute(
                "SELECT rowid FROM keywords WHERE keywords MATCH ? ORDER BY bm25(keywords, 0.5, 1.0) LIMIT ?",
                (match, depth),
            )
        ]
        similarity = self._vectors @ np.asarray(self._embedder.query(query), dtype=np.float32)
        order = np.argsort(-similarity, kind="stable")[:depth]
        by_meaning = [int(n) for n in self._numbers[order]]

        keyword_rank = {n: rank for rank, n in enumerate(by_keyword, start=1)}
        meaning_rank = {n: rank for rank, n in enumerate(by_meaning, start=1)}
        score = {
            n: sum(1 / (FUSION_K + r[n]) for r in (keyword_rank, meaning_rank) if n in r)
            for n in keyword_rank.keys() | meaning_rank.keys()
        }
        best = sorted(score, key=lambda n: (-score[n], n))[:k]
        sql = f"SELECT n, id, paper_id, private, title, text FROM passages WHERE n IN ({','.join('?' * len(best))})"
        rows = {row[0]: row for row in self._db.execute(sql, best)}
        hits = []
        for n in best:
            _n, passage_id, paper_id, private, title, text = rows[n]
            hits.append(
                Hit(
                    passage_id,
                    paper_id,
                    bool(private),
                    title,
                    text,
                    score[n],
                    keyword_rank.get(n),
                    meaning_rank.get(n),
                )
            )
        return hits

    def passage(self, passage_id: str) -> Passage | None:
        """The passage with that id, if the index holds it."""
        row = self._db.execute(
            "SELECT id, paper_id, private, title, text FROM passages WHERE id = ?", (passage_id,)
        ).fetchone()
        return Passage(row[0], row[1], bool(row[2]), row[3], row[4]) if row else None


def search(
    query: str, k: int = 10, *, index: Path = DEFAULT_INDEX, library: Path = DEFAULT_LIBRARY
) -> list[Hit]:
    """The `k` library passages most relevant to `query`, best first, from the default index.

    Loads the model on every call; to search many times, open an `Index` once instead.
    """
    from hypothesis_engine.embedding import LocalEmbedder

    with Index.open(index, library, LocalEmbedder()) as opened:
        return opened.search(query, k)
