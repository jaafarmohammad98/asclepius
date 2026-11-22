"""The topic searches the library is built from: `topics.toml` beside this file by default."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TOPICS = Path(__file__).with_name("topics.toml")


class TopicsError(Exception):
    """The topics file is missing or malformed; the message names the file and the problem."""


@dataclass(frozen=True)
class Topic:
    name: str
    query: str  # PubMed search syntax
    max_papers: int  # keep this many of the most relevant results


def load_topics(path: Path = DEFAULT_TOPICS) -> list[Topic]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise TopicsError(f"{path}: {error}") from error
    entries = data.get("topic")
    if not isinstance(entries, list) or not entries:
        raise TopicsError(f"{path}: no [[topic]] entries")
    topics = []
    for number, entry in enumerate(entries, start=1):
        name, query, max_papers = entry.get("name"), entry.get("query"), entry.get("max_papers")
        if not isinstance(name, str) or not name.strip():
            raise TopicsError(f"{path}: topic {number} has no name")
        if not isinstance(query, str) or not query.strip():
            raise TopicsError(f"{path}: topic {name!r} has no query")
        # PubMed's esearch returns at most 10,000 ids per request
        if not isinstance(max_papers, int) or isinstance(max_papers, bool) or not 1 <= max_papers <= 10_000:
            raise TopicsError(f"{path}: topic {name!r} needs max_papers between 1 and 10000")
        topics.append(Topic(name.strip(), query.strip(), max_papers))
    names = [t.name for t in topics]
    if len(set(names)) != len(names):
        raise TopicsError(f"{path}: topic names must be different")
    return topics
