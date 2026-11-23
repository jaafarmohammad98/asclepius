"""The engine's one way to an LLM: `complete(messages)`, paid for by the researcher.

Two back ends, chosen by settings alone:

- `anthropic`: Claude through the Anthropic API, with the researcher's own API key.
- `local`: a model server on the researcher's machine that speaks the OpenAI-style
  `/v1/chat/completions` protocol (Ollama, llama.cpp's `llama-server`, LM Studio, vLLM).

Settings come from environment variables, then `llm.toml` in the data folder
(`hypothesis_engine.home`; in a source checkout `hypothesis/data/`, which git ignores). The key is never written anywhere by this module: it is left out of every repr,
every recorded request, every completion record and every error message.
"""

from __future__ import annotations

import copy
import json
import os
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

import anthropic

from hypothesis_engine.library import DEFAULT_LIBRARY

DEFAULT_SETTINGS = DEFAULT_LIBRARY.parent / "llm.toml"

Backend = Literal["anthropic", "local"]
BACKENDS: tuple[Backend, ...] = ("anthropic", "local")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
ANTHROPIC_MODEL = "claude-opus-5-5"
LOCAL_URL = "http://localhost:11434/v1"  # Ollama's default
MAX_TOKENS = 16000
# Anthropic's server-side fallback: a request the model declines is run on another model in
# the same call (the record names the model that answered); models without it run alone
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# the settings file: top-level keys, and the keys of each back end's table
TOP_KEYS = {"backend", "max_tokens", *BACKENDS}
BACKEND_KEYS = {"anthropic": {"model", "effort", "api_key"}, "local": {"model", "url"}}
# environment variables, which override the file
ENV = ("ASCLEPIUS_LLM_BACKEND", "ASCLEPIUS_LLM_MODEL", "ASCLEPIUS_LLM_URL", "ANTHROPIC_API_KEY")

_HOW_TO = (
    "No LLM is configured. Either set ANTHROPIC_API_KEY to your own Anthropic API key, or run a "
    'local model server and set backend = "local" and model under [local] in {path} (or '
    "ASCLEPIUS_LLM_BACKEND=local and ASCLEPIUS_LLM_MODEL)."
)


class LLMNotConfigured(Exception):
    """Neither a key nor a local model is set up, or the settings are wrong."""


@dataclass(frozen=True)
class Settings:
    """Which LLM to reach and how. `api_key` is never shown: see `public()`."""

    backend: Backend
    model: str
    max_tokens: int = MAX_TOKENS
    effort: str | None = None  # anthropic only
    url: str = LOCAL_URL  # local only
    api_key: str | None = field(default=None, repr=False)

    def public(self) -> dict[str, object]:
        """The settings to record with an output: everything but the key."""
        shown: dict[str, object] = {
            "backend": self.backend,
            "model": self.model,
            "max_tokens": self.max_tokens,
        }
        if self.backend == "anthropic":
            shown["effort"] = self.effort
        else:
            shown["url"] = self.url
        return shown


def load_settings(env: Mapping[str, str], path: Path = DEFAULT_SETTINGS) -> Settings:
    """The settings from `env` (usually `os.environ`) over the file at `path`, if any.

    Each back end keeps its own table (`[anthropic]`, `[local]`), so switching between them
    is a change of `backend` alone.
    """
    file: dict[str, Any] = {}
    if path.exists():
        try:
            file = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as error:
            raise LLMNotConfigured(f"{path} is not valid TOML: {error}") from None
    unknown = sorted(set(file) - TOP_KEYS) + [
        f"{name}.{key}"
        for name in BACKENDS
        for key in sorted(set(_table(file, name, path)) - BACKEND_KEYS[name])
    ]
    if unknown:
        raise LLMNotConfigured(f"unknown settings in {path}: {', '.join(unknown)}")

    cloud = _table(file, "anthropic", path) | _set(api_key=env.get("ANTHROPIC_API_KEY"))
    local = _table(file, "local", path) | _set(url=env.get("ASCLEPIUS_LLM_URL"))
    backend = env.get("ASCLEPIUS_LLM_BACKEND") or file.get("backend")
    backend = backend or ("anthropic" if cloud.get("api_key") else None)
    if backend is None:
        raise LLMNotConfigured(_HOW_TO.format(path=path))
    if backend not in BACKENDS:
        raise LLMNotConfigured(f"backend must be one of {', '.join(BACKENDS)}, not {backend!r}")
    chosen = cloud if backend == "anthropic" else local
    chosen |= _set(model=env.get("ASCLEPIUS_LLM_MODEL"))

    if backend == "anthropic" and not cloud.get("api_key"):
        raise LLMNotConfigured(
            f"backend is anthropic but no key is set: set ANTHROPIC_API_KEY, or api_key under [anthropic] in {path}"
        )
    if backend == "local" and not local.get("model"):
        raise LLMNotConfigured(
            f"backend is local but no model is named: set model under [local] in {path}, or "
            "ASCLEPIUS_LLM_MODEL (the name your model server knows it by, e.g. llama3.1:8b)"
        )
    effort = cloud.get("effort") if backend == "anthropic" else None
    if effort is not None and effort not in EFFORTS:
        raise LLMNotConfigured(f"effort must be one of {', '.join(EFFORTS)}, not {effort!r}")
    max_tokens = file.get("max_tokens", MAX_TOKENS)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise LLMNotConfigured(f"max_tokens must be a whole number above 0, not {max_tokens!r}")
    return Settings(
        backend=backend,
        model=str(chosen.get("model") or ANTHROPIC_MODEL),
        max_tokens=max_tokens,
        effort=effort,
        url=str(local.get("url") or LOCAL_URL).rstrip("/"),
        api_key=str(cloud["api_key"]) if backend == "anthropic" else None,
    )


def _table(file: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    table = file.get(name, {})
    if not isinstance(table, dict):
        raise LLMNotConfigured(f"{name} in {path} must be a table: [{name}]")
    return dict(table)


def _set(**values: str | None) -> dict[str, str]:
    """The values that are set (an empty environment variable counts as unset)."""
    return {name: value for name, value in values.items() if value}


# --- calling it ----------------------------------------------------------------------------


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(Exception):
    """The LLM could not be reached or gave no usable answer. Never holds the key."""


@dataclass(frozen=True)
class OutgoingRequest:
    """One request as it is about to leave: where to, and the JSON body. No headers, so no key."""

    backend: str
    url: str
    body: dict[str, Any]

    def text(self) -> str:
        """The body as it is sent, for checks that look for text in it."""
        return json.dumps(self.body, ensure_ascii=False)


RequestHook = Callable[[OutgoingRequest], None]


@dataclass(frozen=True)
class Completion:
    """One answer, with what produced it (spec H6)."""

    text: str
    backend: str
    model: str  # the model that answered, as the server names it
    settings: dict[str, object]  # Settings.public(): never the key
    stop_reason: str | None
    usage: dict[str, int]

    def record(self) -> dict[str, object]:
        """What to store with an output that used this answer."""
        return {
            "backend": self.backend,
            "model": self.model,
            "settings": self.settings,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
        }


# a local model server: (url, JSON body) -> JSON response
LocalTransport = Callable[[str, dict[str, Any]], dict[str, Any]]
LOCAL_TIMEOUT = 600  # seconds; a local model on a laptop CPU is slow


def http_local_transport(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=LOCAL_TIMEOUT) as response:
            answer: dict[str, Any] = json.load(response)
            return answer
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise LLMError(f"the local model server at {url} answered HTTP {error.code}: {detail}") from None
    except ValueError:
        raise LLMError(f"the local model server at {url} did not answer in JSON") from None
    except (urllib.error.URLError, OSError) as error:
        raise LLMError(
            f"no local model server answered at {url} ({error}); is it running? Ollama: `ollama serve`"
        ) from None


class LLM:
    """`complete(messages)` on the configured back end. Make one with `connect`."""

    def __init__(
        self,
        settings: Settings,
        on_request: RequestHook | None,
        anthropic_client: anthropic.Anthropic | None,
        local_transport: LocalTransport,
    ) -> None:
        self.settings = settings
        self._on_request = on_request
        self._anthropic = anthropic_client
        self._local = local_transport

    def __repr__(self) -> str:
        return f"LLM({self.settings!r})"

    def complete(self, messages: Sequence[Message]) -> Completion:
        """The model's answer to `messages`; leading system messages are its instructions."""
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("messages must end with a user message")
        if self.settings.backend == "anthropic":
            return self._complete_anthropic(messages)
        return self._complete_local(messages)

    def _send(self, request: OutgoingRequest) -> None:
        if self._on_request is not None:
            self._on_request(OutgoingRequest(request.backend, request.url, copy.deepcopy(request.body)))

    def _complete_anthropic(self, messages: Sequence[Message]) -> Completion:
        settings = self.settings
        if self._anthropic is None:
            self._anthropic = anthropic.Anthropic(api_key=settings.api_key)
        client = self._anthropic
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body: dict[str, Any] = {
            "model": settings.model,
            "max_tokens": settings.max_tokens,
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
            ],
        }
        if system:
            body["system"] = system
        if settings.effort:
            body["output_config"] = {"effort": settings.effort}
        body["fallbacks"] = "default"
        self._send(OutgoingRequest("anthropic", f"{str(client.base_url).rstrip('/')}/v1/messages", body))

        # the SDK's exceptions hold the request, headers and key included: keep them out of ours
        failure = None
        try:
            response = client.beta.messages.create(**body, betas=[FALLBACK_BETA])
        except anthropic.APIStatusError as error:
            failure = f"the Anthropic API answered HTTP {error.status_code}: {_scrub(error.message, settings.api_key)}"
        except anthropic.APIConnectionError as error:
            failure = f"could not reach the Anthropic API: {_scrub(str(error), settings.api_key)}"
        if failure is not None:
            raise LLMError(failure)

        if response.stop_reason == "refusal":
            details = response.stop_details
            category = getattr(details, "category", None) if details else None
            raise LLMError(f"{response.model} declined the request (category: {category or 'none given'})")
        return Completion(
            text="".join(block.text for block in response.content if block.type == "text"),
            backend="anthropic",
            model=response.model,
            settings=settings.public(),
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def _complete_local(self, messages: Sequence[Message]) -> Completion:
        request = OutgoingRequest(
            "local",
            f"{self.settings.url}/chat/completions",
            {
                "model": self.settings.model,
                "max_tokens": self.settings.max_tokens,
                "messages": list(messages),
            },
        )
        self._send(request)
        answer = self._local(request.url, request.body)
        try:
            choice = answer["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"the local model server sent no answer: {str(answer)[:300]}") from None
        usage = answer.get("usage") or {}
        return Completion(
            text=text,
            backend="local",
            model=str(answer.get("model") or self.settings.model),
            settings=self.settings.public(),
            stop_reason=choice.get("finish_reason"),
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
            },
        )


def _scrub(text: str, key: str | None) -> str:
    return text.replace(key, "[key]") if key else text


def connect(
    settings: Settings | None = None,
    on_request: RequestHook | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
    local_transport: LocalTransport = http_local_transport,
) -> LLM:
    """The LLM the settings name (by default those from the environment and `DEFAULT_SETTINGS`).

    `on_request` is called with every request just before it is sent; raising stops it. Tests
    pass `anthropic_client` or `local_transport` to answer without the network.
    """
    return LLM(settings or load_settings(os.environ), on_request, anthropic_client, local_transport)
