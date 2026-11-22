"""The paper library: one SQLite file on the researcher's machine.

Each build is written in one transaction and gets the next version number (1, 2, 3...).
Outputs record that number (spec H6). Papers are never deleted: `paper_topics` lists, per
version, the papers that build held and the topics that found them, so the papers an old
output was allowed to cite can still be read and checked. A paper's record is the latest one
fetched. Its `abstract` is NULL when PubMed has none, and `has_abstract` says so in a column
of its own.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType

from hypothesis_engine.pubmed import Paper

SCHEMA_VERSION = 1

_SCHEMA = """
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

_COLUMNS = "pmid, title, abstract, doi, pmcid, journal, year"
_IN_VERSION = "pmid IN (SELECT pmid FROM paper_topics WHERE version = ?)"


class LibraryError(Exception):
    """No library at that path, or the file is not one."""


def _check_schema(db: sqlite3.Connection, path: Path, *, allow_new: bool) -> bool:
    """Raise unless `path` holds a paper library; True when it is an empty file to create one in."""
    (schema,) = db.execute("PRAGMA user_version").fetchone()
    if schema == SCHEMA_VERSION:
        return False
    if schema == 0 and allow_new and not db.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone():
        return True
    raise LibraryError(f"{path} is not a paper library (schema {schema}, expected {SCHEMA_VERSION})")


class Library:
    """Read access to a built library. Open with `Library.open(path)`; use as a context manager."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection

    @classmethod
    def open(cls, path: Path) -> Library:
        if not path.is_file():
            raise LibraryError(f"no paper library at {path}; run `asclepius-library build` first")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None: this code issues BEGIN and COMMIT itself, so nothing commits early
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        try:
            version = _write(connection, path, papers, found_by, topics_sha256, built_at)
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        return version
    except sqlite3.DatabaseError as error:
        raise LibraryError(f"{path}: {error}") from error
    finally:
        connection.close()


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
