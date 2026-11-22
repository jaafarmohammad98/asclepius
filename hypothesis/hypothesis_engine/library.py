"""The paper library: one SQLite file on the researcher's machine.

Each build is written in one transaction and gets the next version number (1, 2, 3...).
Outputs record that number (spec H6). Papers are never deleted: `paper_topics` lists, per
version, the papers that build held and the topics that found them, so the papers an old
output was allowed to cite can still be read and checked. A paper's record is the latest one
fetched. Its `abstract` is NULL when PubMed has none, and `has_abstract` says so in a column
of its own.

`open_access` holds, per paper, whether Europe PMC has an open-access copy, its licence, and
its text once downloaded. Each lookup and each download is committed as it arrives, so an
interrupted download resumes where it stopped.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from hypothesis_engine.pubmed import Paper

SCHEMA_VERSION = 2

_OPEN_ACCESS = """
CREATE TABLE open_access (
    pmid TEXT PRIMARY KEY REFERENCES papers (pmid),
    checked_at TEXT NOT NULL,
    pmcid TEXT,  -- the open-access copy, NULL when there is none
    licence TEXT,
    fetched_at TEXT,  -- NULL until the copy is downloaded
    text TEXT CHECK (text IS NULL OR trim(text) <> ''),  -- NULL after a download: no readable text
    CHECK (pmcid IS NOT NULL OR fetched_at IS NULL),
    CHECK (fetched_at IS NOT NULL OR text IS NULL)
);
"""

_SCHEMA = (
    """
CREATE TABLE builds (
    version INTEGER PRIMARY KEY,
    built_at TEXT NOT NULL,
    topics_sha256 TEXT NOT NULL,
    papers INTEGER NOT NULL
);
CREATE TABLE papers (
    pmid TEXT PRIMARY KEY CHECK (pmid <> '' AND pmid NOT GLOB '*[^0-9]*'),
    title TEXT NOT NULL CHECK (title <> ''),
    abstract TEXT CHECK (abstract IS NULL OR trim(abstract) <> ''),
    has_abstract INTEGER GENERATED ALWAYS AS (abstract IS NOT NULL) VIRTUAL,
    doi TEXT,
    pmcid TEXT,
    journal TEXT,
    year INTEGER,
    first_version INTEGER NOT NULL REFERENCES builds (version)
);
CREATE TABLE paper_topics (
    version INTEGER NOT NULL REFERENCES builds (version),
    pmid TEXT NOT NULL REFERENCES papers (pmid),
    topic TEXT NOT NULL,
    rank INTEGER NOT NULL CHECK (rank >= 1),  -- 1 = the search's most relevant result
    PRIMARY KEY (version, pmid, topic)
);
"""
    + _OPEN_ACCESS
)

_COLUMNS = "pmid, title, abstract, doi, pmcid, journal, year"
_IN_VERSION = "pmid IN (SELECT pmid FROM paper_topics WHERE version = ?)"


class LibraryError(Exception):
    """No library at that path, or the file is not one."""


@dataclass(frozen=True)
class FullText:
    pmid: str
    pmcid: str
    licence: str | None
    text: str


@dataclass(frozen=True)
class Coverage:
    """How much of the current version has full text."""

    papers: int
    checked: int  # papers looked up in Europe PMC
    open_access: int  # papers with an open-access copy
    with_text: int  # papers whose copy has been downloaded and has readable text
    unreadable: int  # copies downloaded that held no readable text
    to_download: int  # copies not downloaded yet


def _no_library(path: Path) -> LibraryError:
    return LibraryError(f"no paper library at {path}; run `asclepius-library build` first")


def _check_schema(db: sqlite3.Connection, path: Path, *, allow_new: bool) -> bool:
    """Raise unless `path` holds a paper library; True when it is an empty file to create one in.

    With `allow_new` (a writer, inside a transaction) a schema-1 library is upgraded in place.
    """
    (schema,) = db.execute("PRAGMA user_version").fetchone()
    if schema == SCHEMA_VERSION:
        return False
    if allow_new and schema == 0 and not db.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone():
        return True
    if allow_new and schema == 1:
        db.execute(_OPEN_ACCESS)
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return False
    if schema == 1:
        raise LibraryError(
            f"{path} was made by an older version; run `asclepius-library fulltext` to upgrade it"
        )
    raise LibraryError(f"{path} is not a paper library (schema {schema}, expected {SCHEMA_VERSION})")


class Library:
    """Read access to a built library. Open with `Library.open(path)`; use as a context manager."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    @classmethod
    def open(cls, path: Path) -> Library:
        if not path.is_file():
            raise _no_library(path)
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            _check_schema(connection, path, allow_new=False)
            if connection.execute("SELECT 1 FROM builds LIMIT 1").fetchone() is None:
                raise LibraryError(f"{path} is not a paper library (no finished build)")
        except sqlite3.DatabaseError as error:
            connection.close()
            raise LibraryError(f"{path} is not a paper library ({error})") from error
        except LibraryError:
            connection.close()
            raise
        return cls(connection)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Library:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self.close()

    def _latest(self) -> tuple[int, str]:
        version, built_at = self._db.execute(
            "SELECT version, built_at FROM builds ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return int(version), str(built_at)

    @property
    def version(self) -> int:
        """The library version: the number of the build that wrote the current papers."""
        return self._latest()[0]

    @property
    def built_at(self) -> str:
        """When that build finished, ISO 8601 with time zone."""
        return self._latest()[1]

    def _version(self, version: int | None) -> int:
        return self.version if version is None else version

    def count(self) -> int:
        """Papers in the current version."""
        sql = f"SELECT count(*) FROM papers WHERE {_IN_VERSION}"
        return int(self._db.execute(sql, (self.version,)).fetchone()[0])

    def lacking_abstract(self) -> int:
        """Papers in the current version that PubMed has no abstract for."""
        sql = f"SELECT count(*) FROM papers WHERE NOT has_abstract AND {_IN_VERSION}"
        return int(self._db.execute(sql, (self.version,)).fetchone()[0])

    def paper(self, pmid: str, version: int | None = None) -> Paper | None:
        """The paper, if that version (the current one by default) holds it."""
        sql = f"SELECT {_COLUMNS} FROM papers WHERE pmid = ? AND {_IN_VERSION}"
        row = self._db.execute(sql, (pmid, self._version(version))).fetchone()
        return Paper(*row) if row else None

    def papers(self) -> Iterator[Paper]:
        """Every paper in the current version, by PMID."""
        sql = f"SELECT {_COLUMNS} FROM papers WHERE {_IN_VERSION} ORDER BY CAST(pmid AS INTEGER)"
        for row in self._db.execute(sql, (self.version,)):
            yield Paper(*row)

    def full_text(self, pmid: str) -> FullText | None:
        """The paper's open-access text and licence, if it has been downloaded and is readable."""
        row = self._db.execute(
            "SELECT pmid, pmcid, licence, text FROM open_access WHERE pmid = ? AND text IS NOT NULL", (pmid,)
        ).fetchone()
        return FullText(*row) if row else None

    def coverage(self) -> Coverage:
        """Full-text coverage of the current version."""
        row = self._db.execute(
            f"""SELECT count(*), count(o.pmid), count(o.pmcid), count(o.text),
                count(o.fetched_at) - count(o.text), count(o.pmcid) - count(o.fetched_at)
            FROM papers p LEFT JOIN open_access o USING (pmid) WHERE p.{_IN_VERSION}""",
            (self.version,),
        ).fetchone()
        return Coverage(*(int(n) for n in row))

    def topics(self, pmid: str, version: int | None = None) -> list[str]:
        """The topics whose search found this paper in that version, in topic-file order."""
        # rows go in per paper in topic-file order, so insertion order is that order
        sql = "SELECT topic FROM paper_topics WHERE pmid = ? AND version = ? ORDER BY rowid"
        return [topic for (topic,) in self._db.execute(sql, (pmid, self._version(version)))]


def write_build(
    path: Path,
    papers: Sequence[Paper],
    found_by: Mapping[str, Sequence[tuple[str, int]]],
    topics_sha256: str,
    built_at: str,
) -> int:
    """Store `papers` as the next version of the library at `path` (created if needed); return it.

    `found_by` maps each PMID to the (topic, rank) pairs of the searches that found it, in
    topic-file order. All or nothing: if anything fails, the file keeps its previous build.
    """
    write = lambda db: _write(db, path, papers, found_by, topics_sha256, built_at)  # noqa: E731
    return _transaction(path, write, create=True)


def _transaction[T](path: Path, work: Callable[[sqlite3.Connection], T], create: bool = False) -> T:
    """Run `work` in one write transaction on the library at `path`: all of it is stored, or none.

    Without `create`, a missing file is an error rather than a new, empty library.
    """
    if not create and not path.is_file():
        raise _no_library(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None: this code issues BEGIN and COMMIT itself, so nothing commits early
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        try:
            result = work(connection)
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        return result
    except sqlite3.DatabaseError as error:
        raise LibraryError(f"{path}: {error}") from error
    finally:
        connection.close()


def _existing(db: sqlite3.Connection, path: Path) -> int:
    """The library's current version; raises if `path` holds no built library."""
    if _check_schema(db, path, allow_new=True):
        raise _no_library(path)
    (version,) = db.execute("SELECT max(version) FROM builds").fetchone()
    if version is None:
        raise LibraryError(f"{path} is not a paper library (no finished build)")
    return int(version)


def unchecked(path: Path) -> list[str]:
    """PMIDs in the current version not yet looked up for an open-access copy, by PMID."""

    def read(db: sqlite3.Connection) -> list[str]:
        sql = f"""SELECT pmid FROM papers WHERE {_IN_VERSION}
            AND pmid NOT IN (SELECT pmid FROM open_access) ORDER BY CAST(pmid AS INTEGER)"""
        return [pmid for (pmid,) in db.execute(sql, (_existing(db, path),))]

    return _transaction(path, read)


def record_lookup(
    path: Path, pmids: Sequence[str], copies: Mapping[str, tuple[str, str | None]], at: str
) -> None:
    """Store one lookup: `copies` maps the PMIDs that have an open-access copy to (PMCID, licence)."""

    def write(db: sqlite3.Connection) -> None:
        _existing(db, path)
        db.executemany(
            "INSERT INTO open_access (pmid, checked_at, pmcid, licence) VALUES (?, ?, ?, ?)",
            [(pmid, at, *copies.get(pmid, (None, None))) for pmid in pmids],
        )

    _transaction(path, write)


def to_download(path: Path) -> list[tuple[str, str, str | None]]:
    """(PMID, PMCID, licence) of the current version's copies not downloaded yet, by PMID."""

    def read(db: sqlite3.Connection) -> list[tuple[str, str, str | None]]:
        sql = f"""SELECT pmid, pmcid, licence FROM open_access WHERE pmcid IS NOT NULL
            AND fetched_at IS NULL AND {_IN_VERSION} ORDER BY CAST(pmid AS INTEGER)"""
        return [(str(a), str(b), c) for a, b, c in db.execute(sql, (_existing(db, path),))]

    return _transaction(path, read)


def record_texts(path: Path, texts: Sequence[tuple[str, str | None, str | None]], at: str) -> None:
    """Store downloaded copies as (PMID, text, licence); text None: the copy held no readable text."""

    def write(db: sqlite3.Connection) -> None:
        _existing(db, path)
        db.executemany(
            "UPDATE open_access SET text = ?, licence = ?, fetched_at = ? WHERE pmid = ? AND pmcid IS NOT NULL",
            [(text, licence, at, pmid) for pmid, text, licence in texts],
        )

    _transaction(path, write)


def _write(
    db: sqlite3.Connection,
    path: Path,
    papers: Sequence[Paper],
    found_by: Mapping[str, Sequence[tuple[str, int]]],
    topics_sha256: str,
    built_at: str,
) -> int:
    unfound = [p.pmid for p in papers if not found_by.get(p.pmid)]
    if unfound:  # a paper no topic found would belong to no version
        raise ValueError(f"papers {unfound[:5]} have no topic")
    if _check_schema(db, path, allow_new=True):
        for statement in _SCHEMA.split(";"):
            if statement.strip():
                db.execute(statement)
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    (last,) = db.execute("SELECT coalesce(max(version), 0) FROM builds").fetchone()
    version = int(last) + 1
    db.execute("INSERT INTO builds VALUES (?, ?, ?, ?)", (version, built_at, topics_sha256, len(papers)))
    db.executemany(
        f"""INSERT INTO papers ({_COLUMNS}, first_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (pmid) DO UPDATE SET title = excluded.title, abstract = excluded.abstract,
            doi = excluded.doi, pmcid = excluded.pmcid, journal = excluded.journal, year = excluded.year""",
        [(p.pmid, p.title, p.abstract, p.doi, p.pmcid, p.journal, p.year, version) for p in papers],
    )
    db.executemany(
        "INSERT INTO paper_topics VALUES (?, ?, ?, ?)",
        [(version, p.pmid, topic, rank) for p in papers for topic, rank in found_by.get(p.pmid, ())],
    )
    return version
