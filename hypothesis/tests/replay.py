"""Recorded PubMed responses, so the suite never touches the network.

Each response is a file named after its request. `record_responses.py` makes them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hypothesis_engine.pubmed import Transport


def request_key(endpoint: str, params: dict[str, str]) -> str:
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    return f"{endpoint}-{digest}"


def suffix(endpoint: str) -> str:
    return ".json" if endpoint == "esearch" else ".xml"


class ReplayTransport:
    """Answers each request from `folder`; a request with no recording fails the test."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, endpoint: str, params: dict[str, str]) -> bytes:
        self.calls.append((endpoint, params))
        path = self.folder / f"{request_key(endpoint, params)}{suffix(endpoint)}"
        if not path.exists():
            raise AssertionError(f"no recorded response for {endpoint} {params}")
        return path.read_bytes()


def recording(transport: Transport, folder: Path) -> Transport:
    """Passes each request to `transport` and saves the answer in `folder`."""

    def send(endpoint: str, params: dict[str, str]) -> bytes:
        body = transport(endpoint, params)
        (folder / f"{request_key(endpoint, params)}{suffix(endpoint)}").write_bytes(body)
        return body

    return send
