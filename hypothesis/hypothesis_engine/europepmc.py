"""Europe PMC's public REST interface: which papers have an open-access copy, and its licence.

Europe PMC mirrors PubMed Central. Its open-access subset holds the papers whose licence
allows reuse; only those are downloaded (from NCBI, see `PubMed.fetch_pmc`). NCBI's own PMC
open-access service no longer answers (HTTP 404 on 2026-11-22), and Europe PMC's full-text
endpoint failed with HTTP 500 for most open-access copies tried that day, so Europe PMC is
asked only which papers are open access. No account or email is needed, and none is sent.

Europe PMC publishes no request limit; this client sends at most 3 requests a second and
backs off on HTTP 429 and 5xx answers. Requests go through a `Transport` (endpoint and form
fields to response body), so tests replay recorded answers instead of reaching the network.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hypothesis_engine.pubmed import RateLimiter, TransientError, Transport, send_politely

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
RATE = 3  # requests a second
LOOKUP_BATCH = 100  # PMIDs per search; a 100-id search takes Europe PMC about 10 seconds
TIMEOUT = 120.0
USER_AGENT = "asclepius (open-source research tool)"
SEARCH = "searchPOST"


class EuropePMCError(Exception):
    """Europe PMC answered, but not with something usable."""


@dataclass(frozen=True)
class OpenAccessCopy:
    pmid: str
    pmcid: str
    licence: str | None  # as Europe PMC gives it, e.g. "cc by"; None when it gives none


def http_transport() -> Transport:
    """The real network: form fields go in a POST body."""

    def send(path: str, fields: dict[str, str]) -> bytes:
        data = urllib.parse.urlencode(fields).encode()
        request = urllib.request.Request(
            f"{BASE}{path}", data=data, method="POST", headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body: bytes = response.read()
                return body
        except urllib.error.HTTPError as error:
            if error.code == 429 or error.code >= 500:
                raise TransientError(f"{path}: HTTP {error.code}") from error
            raise EuropePMCError(f"{path}: HTTP {error.code} {error.reason}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise TransientError(f"{path}: {error}") from error

    return send


class EuropePMC:
    """Looks up open-access copies through `transport`, waiting on `limiter` before every request."""

    def __init__(
        self,
        transport: Transport,
        limiter: RateLimiter,
        sleep: Callable[[float], None] = time.sleep,
        lookup_batch: int = LOOKUP_BATCH,
    ) -> None:
        self._transport = transport
        self._limiter = limiter
        self._sleep = sleep
        self.lookup_batch = lookup_batch

    def _request(self, path: str, fields: dict[str, str]) -> bytes:
        return send_politely(self._transport, path, fields, self._limiter, self._sleep)

    def open_access(self, pmids: Sequence[str]) -> list[OpenAccessCopy]:
        """The open-access copies of these papers (at most `lookup_batch`); papers without one are absent."""
        if not pmids:
            return []
        if len(pmids) > self.lookup_batch:
            raise ValueError(f"at most {self.lookup_batch} PMIDs per lookup")
        fields = {
            "query": f"EXT_ID:({' OR '.join(pmids)}) AND SRC:MED AND OPEN_ACCESS:y",
            "resultType": "core",  # the only result type that carries the licence
            "format": "json",
            "pageSize": "1000",
        }
        body = self._request(SEARCH, fields)
        try:
            results = json.loads(body)["resultList"]["result"]
        except (ValueError, KeyError, TypeError) as error:
            raise EuropePMCError(f"search: unreadable answer for PMIDs {pmids[0]}..{pmids[-1]}") from error
        asked = set(pmids)
        copies = {}
        for result in results:
            pmid, pmcid = str(result.get("pmid") or ""), result.get("pmcid")
            if pmid in asked and pmcid and result.get("isOpenAccess") == "Y":
                copies[pmid] = OpenAccessCopy(pmid, str(pmcid), result.get("license") or None)
        return [copies[pmid] for pmid in pmids if pmid in copies]
