"""Recorded PubMed, Europe PMC, ChEMBL and MyChem.info responses, so the suite never touches the network.

Each response is a file named after its request. `record_responses.py` makes them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hypothesis_engine.pubmed import Transport


def request_key(endpoint: str, params: dict[str, str]) -> str:
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    return f"{endpoint.replace('/', '_')}-{digest}"


def response_path(folder: Path, endpoint: str, params: dict[str, str]) -> Path:
    json_answer = endpoint in ("esearch", "searchPOST", "query") or endpoint.endswith(".json")
    suffix = ".json" if json_answer else ".xml"
    return folder / f"{request_key(endpoint, params)}{suffix}"


class ReplayTransport:
    """Answers each request from `folder`; a request with no recording fails the test."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, endpoint: str, params: dict[str, str]) -> bytes:
        self.calls.append((endpoint, params))
        path = response_path(self.folder, endpoint, params)
        if not path.exists():
            raise AssertionError(f"no recorded response for {endpoint} {params}")
        return path.read_bytes()


def recording(transport: Transport, folder: Path) -> Transport:
    """Passes each request to `transport` and saves the answer in `folder`."""

    def send(endpoint: str, params: dict[str, str]) -> bytes:
        body = transport(endpoint, params)
        response_path(folder, endpoint, params).write_bytes(body)
        return body

    return send
