"""The paper library: one SQLite file on the researcher's machine.

Each build is written in one transaction and gets the next version number (1, 2, 3...).
Outputs record that number (spec H6). Papers are never deleted: `paper_topics` lists, per
version, the papers that build held and the topics that found them, so the papers an old
output was allowed to cite can still be read and checked. A paper's record is the latest one
fetched. Its `abstract` is NULL when PubMed has none, and `has_abstract` says so in a column
of its own.

`open_access` holds, per paper, whether Europe PMC has an open-access copy, its licence, and
its text once downloaded. Each lookup and each download is committed as it arrives, so an
interrupted download resumes where it stopped. Downloaded text joins the library in a new
version (`seal_texts`), once per run, so a version always reads the same text.

`private_papers` holds the researcher's own PDFs (see `private.py`) and `private_versions`
which versions held each. The version goes up on every change: new papers, new full text, a
private paper added or removed. A refresh that finds nothing new adds no version.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from hypothesis_engine.private import PrivatePaper
from hypothesis_engine.pubmed import Paper

SCHEMA_VERSION = 3

_OPEN_ACCESS = """
CREATE TABLE open_access (
    pmid TEXT PRIMARY KEY REFERENCES papers (pmid),
    checked_at TEXT NOT NULL,
    pmcid TEXT,  -- the open-access copy, NULL when there is none
    licence TEXT,
    fetched_at TEXT,  -- NULL until the copy is downloaded
    text TEXT CHECK (text IS NULL OR trim(text) <> ''),  -- NULL after a download: no readable text
    version INTEGER REFERENCES builds (version),  -- the first version holding the text, NULL until one does
    CHECK (pmcid IS NOT NULL OR fetched_at IS NULL),
    CHECK (fetched_at IS NOT NULL OR text IS NULL),
    CHECK (version IS NULL OR text IS NOT NULL)
);
"""

_PRIVATE = """
CREATE TABLE private_papers (
    id TEXT PRIMARY KEY CHECK (id GLOB 'private-*'),
    file_name TEXT NOT NULL,  -- the latest name it was read under
    title TEXT NOT NULL CHECK (title <> ''),
    text TEXT NOT NULL CHECK (trim(text) <> ''),
    first_version INTEGER NOT NULL REFERENCES builds (version)
);
CREATE TABLE private_versions (
    version INTEGER NOT NULL REFERENCES builds (version),
    id TEXT NOT NULL REFERENCES private_papers (id),
    PRIMARY KEY (version, id)
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
    + _PRIVATE
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
class Document:
    """A paper as text to search and quote: a PubMed paper (id: its PMID) or a private one."""

    id: str
    title: str
    text: str  # the full text when the version has it, else the abstract ("" when neither)
    private: bool


@dataclass(frozen=True)
class Change:
    """What a refresh changed. `version` is None when nothing did and no version was added."""

    version: int | None
    new_papers: int
    private_added: int
    private_removed: int


@dataclass(frozen=True)
class Coverage:
    """How much of the current version has full text."""

    papers: int
    checked: int  # papers looked up in Europe PMC
    open_access: int  # papers with an open-access copy
    with_text: int  # papers whose copy has been downloaded and has readable text
    unreadable: int  # copies downloaded that held no readable text
    to_download: int  # copies not downloaded yet


def _run(db: sqlite3.Connection, script: str) -> None:
    for statement in script.split(";"):
        if statement.strip():
            db.execute(statement)


def _normal(text: str) -> str:
    """For word-for-word matching: case and runs of white space do not count."""
    return " ".join(text.casefold().split())


def _no_library(path: Path) -> LibraryError:
    return LibraryError(f"no paper library at {path}; run `asclepius-library build` first")


def _check_schema(db: sqlite3.Connection, path: Path, *, allow_new: bool) -> bool:
    """Raise unless `path` holds a paper library; True when it is an empty file to create one in.

    With `allow_new` (a writer, inside a transaction) an older library is upgraded in place:
    text downloaded before versions covered it belongs to the current version.
    """
    (schema,) = db.execute("PRAGMA user_version").fetchone()
    if schema == SCHEMA_VERSION:
        return False
    if allow_new and schema == 0 and not db.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone():
        return True
    if allow_new and schema in (1, 2):
        if schema == 1:
            _run(db, _OPEN_ACCESS)
        else:
            db.execute("ALTER TABLE open_access ADD COLUMN version INTEGER REFERENCES builds (version)")
            db.execute(
                "UPDATE open_access SET version = (SELECT max(version) FROM builds) WHERE text IS NOT NULL"
            )
        _run(db, _PRIVATE)
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return False
    if schema in (1, 2):
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

    def full_text(self, pmid: str, version: int | None = None) -> FullText | None:
        """The paper's open-access text and licence, if that version (the current one) holds it."""
        row = self._db.execute(
            "SELECT pmid, pmcid, licence, text FROM open_access WHERE pmid = ? AND version <= ?",
            (pmid, self._version(version)),
        ).fetchone()
        return FullText(*row) if row else None

    def private_paper(self, paper_id: str, version: int | None = None) -> PrivatePaper | None:
        """The researcher's private paper, if that version (the current one by default) holds it."""
        row = self._db.execute(
            """SELECT id, file_name, title, text FROM private_papers
            WHERE id = ? AND id IN (SELECT id FROM private_versions WHERE version = ?)""",
            (paper_id, self._version(version)),
        ).fetchone()
        return PrivatePaper(*row) if row else None

    def private_papers(self, version: int | None = None) -> list[PrivatePaper]:
        """The private papers in that version, by file name."""
        sql = """SELECT id, file_name, title, text FROM private_papers
            WHERE id IN (SELECT id FROM private_versions WHERE version = ?) ORDER BY file_name, id"""
        return [PrivatePaper(*row) for row in self._db.execute(sql, (self._version(version),))]

    def documents(self, version: int | None = None) -> Iterator[Document]:
        """Every paper of that version as text: PubMed papers by PMID, then the private ones."""
        version = self._version(version)
        sql = f"""SELECT p.pmid, p.title, coalesce(o.text, p.abstract, '') FROM papers p
            LEFT JOIN open_access o ON o.pmid = p.pmid AND o.version <= ?
            WHERE p.{_IN_VERSION} ORDER BY CAST(p.pmid AS INTEGER)"""
        for pmid, title, text in self._db.execute(sql, (version, version)):
            yield Document(pmid, title, text, private=False)
        for paper in self.private_papers(version):
            yield Document(paper.id, paper.title, paper.text, private=True)

    def find(self, passage: str, version: int | None = None) -> list[str]:
        """Ids of the papers whose title or text holds `passage` word for word (case and spacing aside)."""
        wanted = _normal(passage)
        if not wanted:
            return []
        return [d.id for d in self.documents(version) if wanted in _normal(f"{d.title}\n{d.text}")]

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
    private: Sequence[PrivatePaper] | None = None,
) -> int:
    """Store `papers` as the next version of the library at `path` (created if needed); return it.

    `found_by` maps each PMID to the (topic, rank) pairs of the searches that found it, in
    topic-file order. `private` is the private folder's papers; None keeps the previous
    version's. All or nothing: if anything fails, the file keeps its previous build.
    """
    write = lambda db: _write(db, path, papers, found_by, topics_sha256, built_at, private)  # noqa: E731
    return _transaction(path, write, create=True)


def write_refresh(
    path: Path,
    papers: Sequence[Paper],
    found_by: Mapping[str, Sequence[tuple[str, int]]],
    topics_sha256: str,
    built_at: str,
    private: Sequence[PrivatePaper],
) -> Change:
    """Add to the library at `path` what the searches and the private folder hold that its
    current version lacks, as one new version; add none when nothing is new.

    `papers` are the newly fetched records; `found_by` maps every PMID the searches found
    (stored or new) to its (topic, rank) pairs. A paper stays even when its search no longer
    finds it. The private folder replaces the previous version's private papers.
    """

    def write(db: sqlite3.Connection) -> Change:
        previous = _existing(db, path)
        pairs = _pairs(db, previous)
        stored = {pmid for (pmid,) in db.execute("SELECT pmid FROM papers")} | {p.pmid for p in papers}
        new_pairs = [
            (pmid, topic, rank)
            for pmid, found in found_by.items()
            if pmid in stored
            for topic, rank in found
            if (pmid, topic) not in pairs
        ]
        before = _private_ids(db, previous)
        after = {p.id for p in private}
        # a renamed file is not a change, but the stored name follows it
        db.executemany(
            "UPDATE private_papers SET file_name = ? WHERE id = ?", [(p.file_name, p.id) for p in private]
        )
        if not new_pairs and before == after:
            return Change(None, 0, 0, 0)
        in_previous = {pmid for pmid, _topic in pairs}
        new_pmids = {pmid for pmid, _topic, _rank in new_pairs} - in_previous
        version = _add_version(db, built_at, topics_sha256, len(in_previous | new_pmids))
        _store_papers(db, [p for p in papers if p.pmid in new_pmids], version)
        _carry_forward(db, previous, version, private=False)
        db.executemany(
            "INSERT INTO paper_topics VALUES (?, ?, ?, ?)", [(version, *pair) for pair in new_pairs]
        )
        _store_private(db, private, version)
        return Change(version, len(new_pmids), len(after - before), len(before - after))

    return _transaction(path, write)


def stored_pmids(path: Path) -> set[str]:
    """Every PMID the library at `path` has a record for, in any version."""

    def read(db: sqlite3.Connection) -> set[str]:
        _existing(db, path)
        return {pmid for (pmid,) in db.execute("SELECT pmid FROM papers")}

    return _transaction(path, read)


def seal_texts(path: Path, at: str) -> int | None:
    """Put the full text downloaded since the last version into a new one; return it (None: no new text)."""

    def write(db: sqlite3.Connection) -> int | None:
        previous = _existing(db, path)
        if (
            db.execute("SELECT 1 FROM open_access WHERE text IS NOT NULL AND version IS NULL").fetchone()
            is None
        ):
            return None
        sha, papers = db.execute(
            "SELECT topics_sha256, papers FROM builds WHERE version = ?", (previous,)
        ).fetchone()
        version = _add_version(db, at, sha, papers)
        _carry_forward(db, previous, version)
        db.execute(
            "UPDATE open_access SET version = ? WHERE text IS NOT NULL AND version IS NULL", (version,)
        )
        return version

    return _transaction(path, write)


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


def unchecked(path: Path, recheck_before: str = "") -> list[str]:
    """PMIDs in the current version to look up for an open-access copy, by PMID: those never
    looked up, and those found without a copy before `recheck_before` (ISO 8601, UTC)."""

    def read(db: sqlite3.Connection) -> list[str]:
        sql = f"""SELECT pmid FROM papers WHERE {_IN_VERSION}
            AND pmid NOT IN (SELECT pmid FROM open_access WHERE pmcid IS NOT NULL OR checked_at >= ?)
            ORDER BY CAST(pmid AS INTEGER)"""
        return [pmid for (pmid,) in db.execute(sql, (_existing(db, path), recheck_before))]

    return _transaction(path, read)


def record_lookup(
    path: Path, pmids: Sequence[str], copies: Mapping[str, tuple[str, str | None]], at: str
) -> None:
    """Store one lookup: `copies` maps the PMIDs that have an open-access copy to (PMCID, licence)."""

    def write(db: sqlite3.Connection) -> None:
        _existing(db, path)
        db.executemany(
            """INSERT INTO open_access (pmid, checked_at, pmcid, licence) VALUES (?, ?, ?, ?)
            ON CONFLICT (pmid) DO UPDATE SET checked_at = excluded.checked_at, pmcid = excluded.pmcid,
                licence = excluded.licence WHERE open_access.pmcid IS NULL""",
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
    private: Sequence[PrivatePaper] | None,
) -> int:
    unfound = [p.pmid for p in papers if not found_by.get(p.pmid)]
    if unfound:  # a paper no topic found would belong to no version
        raise ValueError(f"papers {unfound[:5]} have no topic")
    if _check_schema(db, path, allow_new=True):
        _run(db, _SCHEMA)
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    version = _add_version(db, built_at, topics_sha256, len(papers))
    _store_papers(db, papers, version)
    db.executemany(
        "INSERT INTO paper_topics VALUES (?, ?, ?, ?)",
        [(version, p.pmid, topic, rank) for p in papers for topic, rank in found_by.get(p.pmid, ())],
    )
    if private is not None:
        _store_private(db, private, version)
    else:
        _carry_forward(db, version - 1, version, topics=False)
    return version


def _carry_forward(
    db: sqlite3.Connection, previous: int, version: int, topics: bool = True, private: bool = True
) -> None:
    """Copy the previous version's papers (`topics`) and its private papers into `version`."""
    if topics:
        db.execute(
            "INSERT INTO paper_topics SELECT ?, pmid, topic, rank FROM paper_topics WHERE version = ?",
            (version, previous),
        )
    if private:
        db.execute(
            "INSERT INTO private_versions SELECT ?, id FROM private_versions WHERE version = ?",
            (version, previous),
        )


def _add_version(db: sqlite3.Connection, built_at: str, topics_sha256: str, papers: int) -> int:
    (last,) = db.execute("SELECT coalesce(max(version), 0) FROM builds").fetchone()
    version = int(last) + 1
    db.execute("INSERT INTO builds VALUES (?, ?, ?, ?)", (version, built_at, topics_sha256, papers))
    return version


def _store_papers(db: sqlite3.Connection, papers: Sequence[Paper], version: int) -> None:
    """Insert or update the records; a new one's first version is `version`."""
    db.executemany(
        f"""INSERT INTO papers ({_COLUMNS}, first_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (pmid) DO UPDATE SET title = excluded.title, abstract = excluded.abstract,
            doi = excluded.doi, pmcid = excluded.pmcid, journal = excluded.journal, year = excluded.year""",
        [(p.pmid, p.title, p.abstract, p.doi, p.pmcid, p.journal, p.year, version) for p in papers],
    )


def _store_private(db: sqlite3.Connection, papers: Sequence[PrivatePaper], version: int) -> None:
    """Store the private folder's papers as the private papers of `version`."""
    db.executemany(
        """INSERT INTO private_papers VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET file_name = excluded.file_name""",
        [(p.id, p.file_name, p.title, p.text, version) for p in papers],
    )
    db.executemany("INSERT INTO private_versions VALUES (?, ?)", [(version, p.id) for p in papers])


def _pairs(db: sqlite3.Connection, version: int) -> set[tuple[str, str]]:
    """The (PMID, topic) pairs of a version."""
    return {
        (p, t) for p, t in db.execute("SELECT pmid, topic FROM paper_topics WHERE version = ?", (version,))
    }


def _private_ids(db: sqlite3.Connection, version: int) -> set[str]:
    return {i for (i,) in db.execute("SELECT id FROM private_versions WHERE version = ?", (version,))}
