"""PubMed's public E-utilities interface: search, fetch and read paper records.

Only the standard library is used. Requests go through a `Transport` (a function from a URL
and its parameters to the response body), so tests replay recorded responses instead of
reaching the network.

NCBI's rules (https://www.ncbi.nlm.nih.gov/books/NBK25497/): at most 3 requests a second, or
10 with an API key, and every request names the tool. No email is sent: it would identify
the researcher, and NCBI does not require one.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

RATE_WITHOUT_KEY = 3
RATE_WITH_KEY = 10
# spacing requests 10% wider than the limit keeps a burst on a second's edge under it
_MARGIN = 1.1

_YEAR = re.compile(r"\b(1[89]\d\d|2\d\d\d)\b")


def rate_for(api_key: str | None) -> int:
    """NCBI's requests-a-second limit for a researcher with or without an API key."""
    return RATE_WITH_KEY if api_key else RATE_WITHOUT_KEY


class RateLimiter:
    """Spaces calls to `wait` so no second holds more than `per_second` of them."""

    def __init__(
        self,
        per_second: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = _MARGIN / per_second
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self._last + self._interval - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


@dataclass(frozen=True)
class Paper:
    """One paper record. `abstract` is None when PubMed has no abstract for it."""

    pmid: str
    title: str
    abstract: str | None
    doi: str | None
    pmcid: str | None
    journal: str | None
    year: int | None


def _text(element: ET.Element | None) -> str:
    """All text inside `element`, inline markup (<i>, <sup>) flattened, spaces tidied."""
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _abstract(parent: ET.Element) -> str | None:
    parts = []
    for section in parent.findall("Abstract/AbstractText"):
        text = _text(section)
        if not text:
            continue
        label = section.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts) or None


def _year(date: ET.Element | None) -> int | None:
    # <Year>, <Month>... or one <MedlineDate> like "1998 Dec-1999 Jan"; join parts so they stay apart
    match = _YEAR.search(" ".join(date.itertext()) if date is not None else "")
    return int(match.group(1)) if match else None


def _ids(id_list: ET.Element | None) -> dict[str, str]:
    if id_list is None:
        return {}
    return {i.get("IdType", ""): (i.text or "").strip() for i in id_list.findall("ArticleId") if i.text}


def _journal_article(record: ET.Element) -> Paper | None:
    citation = record.find("MedlineCitation")
    article = citation.find("Article") if citation is not None else None
    if citation is None or article is None:
        return None
    ids = _ids(record.find("PubmedData/ArticleIdList"))
    journal = _text(article.find("Journal/Title")) or None
    return Paper(
        pmid=_text(citation.find("PMID")),
        title=_text(article.find("ArticleTitle")),
        abstract=_abstract(article),
        doi=ids.get("doi"),
        pmcid=ids.get("pmc"),
        journal=journal,
        year=_year(article.find("Journal/JournalIssue/PubDate")),
    )


def _book_article(record: ET.Element) -> Paper | None:
    document = record.find("BookDocument")
    if document is None:
        return None
    ids = _ids(document.find("ArticleIdList"))
    return Paper(
        pmid=_text(document.find("PMID")),
        title=_text(document.find("ArticleTitle")) or _text(document.find("Book/BookTitle")),
        abstract=_abstract(document),
        doi=ids.get("doi"),
        pmcid=ids.get("pmc"),
        journal=_text(document.find("Book/BookTitle")) or None,
        year=_year(document.find("Book/PubDate")),
    )


def parse_pubmed_xml(body: bytes) -> list[Paper]:
    """The paper records in one efetch response (`retmode=xml`), in response order.

    Records without a PMID or a title are left out: a paper the engine cites must be
    identifiable and readable.
    """
    papers = []
    for record in ET.fromstring(body):
        if record.tag == "PubmedArticle":
            paper = _journal_article(record)
        elif record.tag == "PubmedBookArticle":
            paper = _book_article(record)
        else:
            continue
        if paper is not None and paper.pmid and paper.title:
            papers.append(paper)
    return papers


# --- talking to PubMed -------------------------------------------------------------------

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL = "asclepius"
FETCH_BATCH = 200  # NCBI's suggested ids per efetch request
RETRIES = 3
TIMEOUT = 120.0  # seconds; a 200-record efetch answer can be several megabytes

Transport = Callable[[str, dict[str, str]], bytes]
"""Sends one E-utilities request (`esearch`, `efetch`) with these parameters; returns the body."""


class TransientError(Exception):
    """A request that may work if sent again: a timeout, HTTP 429 or a 5xx answer."""


class PubMedError(Exception):
    """PubMed answered, but not with something usable."""


def http_transport(api_key: str | None = None) -> Transport:
    """The real network. Parameters go in a POST body, so long id lists fit (NCBI's advice)."""

    def send(endpoint: str, params: dict[str, str]) -> bytes:
        fields = {**params, "tool": TOOL}
        if api_key:
            fields["api_key"] = api_key
        data = urllib.parse.urlencode(fields).encode()
        request = urllib.request.Request(f"{BASE}{endpoint}.fcgi", data=data, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body: bytes = response.read()
                return body
        except urllib.error.HTTPError as error:
            if error.code == 429 or error.code >= 500:
                raise TransientError(f"{endpoint}: HTTP {error.code}") from error
            raise PubMedError(f"{endpoint}: HTTP {error.code} {error.reason}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise TransientError(f"{endpoint}: {error}") from error

    return send


class PubMed:
    """Searches and fetches through `transport`, waiting on `limiter` before every request."""

    def __init__(
        self,
        transport: Transport,
        limiter: RateLimiter,
        sleep: Callable[[float], None] = time.sleep,
        fetch_batch: int = FETCH_BATCH,
    ) -> None:
        self._transport = transport
        self._limiter = limiter
        self._sleep = sleep
        self._fetch_batch = fetch_batch

    def _request(self, endpoint: str, params: dict[str, str]) -> bytes:
        for attempt in range(RETRIES + 1):
            self._limiter.wait()
            try:
                return self._transport(endpoint, params)
            except TransientError:
                if attempt == RETRIES:
                    raise
                self._sleep(2.0 * 2**attempt)  # 2, 4, 8 seconds
        raise AssertionError("unreachable")

    def search(self, query: str, max_papers: int) -> list[str]:
        """PMIDs of the `max_papers` most relevant results for `query`."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_papers),
            "sort": "relevance",
            "retmode": "json",
        }
        body = self._request("esearch", params)
        try:
            result = json.loads(body)["esearchresult"]
        except (ValueError, KeyError) as error:
            raise PubMedError(f"esearch: unreadable answer for {query!r}") from error
        if "ERROR" in result:
            raise PubMedError(f"esearch: {result['ERROR']} (query {query!r})")
        return [str(pmid) for pmid in result.get("idlist", [])]

    def fetch(self, pmids: list[str]) -> list[Paper]:
        """Paper records for `pmids`, `fetch_batch` per request. Ids PubMed no longer has are skipped."""
        papers = []
        for start in range(0, len(pmids), self._fetch_batch):
            batch = pmids[start : start + self._fetch_batch]
            body = self._request("efetch", {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"})
            try:
                papers.extend(parse_pubmed_xml(body))
            except ET.ParseError as error:
                raise PubMedError(f"efetch: unreadable XML for ids {batch[0]}..{batch[-1]}") from error
        return papers
