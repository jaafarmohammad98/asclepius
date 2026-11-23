"""The settings screen: which LLM writes the hypotheses (spec H4), the researcher's own Anthropic API
key or a model server on this computer.

The settings live in the hypothesis engine's own file (`hypothesis/data/llm.toml`, which git
ignores), so the app and the `asclepius-library` command use the same ones; environment variables
still override it (`hypothesis_engine.llm`). The key goes from the page into the file and nowhere
else: `read` and `save` say only whether one is stored, and the file is readable by its owner alone.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hypothesis_engine.llm import (
    BACKENDS,
    ENV,
    LLM,
    LLMError,
    LLMNotConfigured,
    connect,
    load_settings,
)

CHECK_QUESTION = "Reply with the one word: ready"


class SettingsError(ValueError):
    """Settings that cannot be saved or used. The message says why, in plain words, never with the key."""


def read(path: Path, env: Mapping[str, str]) -> dict[str, object]:
    """What the screen shows: each back end's settings as stored (the key only as `key_set`), the
    settings in effect once `env` is applied, or the problem that stops them, and which
    environment variables override the file."""
    stored = _stored(path)
    cloud, local = stored.get("anthropic", {}), stored.get("local", {})
    active: dict[str, object] | None
    try:
        active, problem = load_settings(env, path).public(), None
    except LLMNotConfigured as error:
        active, problem = None, str(error)
    return {
        "path": str(path),
        "backend": stored.get("backend"),
        "anthropic": {
            "model": cloud.get("model"),
            "effort": cloud.get("effort"),
            "key_set": bool(cloud.get("api_key")),
        },
        "local": {"model": local.get("model"), "url": local.get("url")},
        "active": active,
        "problem": problem,
        "overridden_by": [name for name in ENV if env.get(name)],
    }


def save(path: Path, form: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, object]:
    """Store the screen's settings, then `read` them back. `form` is
    `{"backend": ..., "anthropic": {"api_key", "model", "effort"}, "local": {"model", "url"}}`; a blank
    key keeps the stored one and `forget_key` removes it. Raises `SettingsError`, changing nothing,
    for settings the engine would refuse."""
    stored = _stored(path, strict=True)
    backend = form.get("backend")
    if backend not in BACKENDS:
        raise SettingsError(f"Pick a back end: {' or '.join(BACKENDS)}.")
    cloud = dict(stored.get("anthropic", {}))
    local = dict(stored.get("local", {}))
    cloud_form, local_form = _table(form, "anthropic"), _table(form, "local")
    for key in ("model", "effort"):
        if key in cloud_form:
            _put(cloud, key, cloud_form[key])
    if cloud_form.get("api_key"):
        cloud["api_key"] = _text(cloud_form["api_key"])
    if form.get("forget_key"):
        cloud.pop("api_key", None)
    for key in ("model", "url"):
        if key in local_form:
            _put(local, key, local_form[key])
    url = str(local.get("url") or "")
    if url and not url.startswith(("http://", "https://")):
        raise SettingsError("The local model server's address must start with http:// or https://.")

    file: dict[str, Any] = {"backend": backend}
    if "max_tokens" in stored:
        file["max_tokens"] = stored["max_tokens"]
    file |= {name: table for name, table in (("anthropic", cloud), ("local", local)) if table}
    _write(path, file)
    return read(path, env)


def check(path: Path, env: Mapping[str, str], connect_llm: Callable[..., LLM] = connect) -> dict[str, object]:
    """Ask the configured LLM one short question (nothing from any dataset) and say which model
    answered. Raises `SettingsError` saying why when none does."""
    try:
        completion = connect_llm(load_settings(env, path)).complete(
            [{"role": "user", "content": CHECK_QUESTION}]
        )
    except (LLMNotConfigured, LLMError) as error:
        raise SettingsError(str(error)) from None
    return {"backend": completion.backend, "model": completion.model, "answer": completion.text.strip()[:200]}


def _stored(path: Path, strict: bool = False) -> dict[str, Any]:
    """The file's settings. An unreadable file reads as none (`read` then names its problem), or
    with `strict` is refused, so a save cannot drop what it holds."""
    try:
        stored = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        if strict:
            raise SettingsError(
                f"{path} is not valid TOML ({error}); fix it or delete it, then save again."
            ) from None
        return {}
    return {k: v for k, v in stored.items() if not (k in BACKENDS and not isinstance(v, dict))}


def _table(form: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    table = form.get(name) or {}
    if not isinstance(table, Mapping):
        raise SettingsError(f"The {name} settings must be a JSON object.")
    return table


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise SettingsError("Each setting must be text.")
    return value.strip()


def _put(table: dict[str, Any], key: str, value: object) -> None:
    """Set `key`, or remove it for a blank value (the engine's default then applies)."""
    text = "" if value is None else _text(value)
    if text:
        table[key] = text
    else:
        table.pop(key, None)


def _write(path: Path, file: dict[str, Any]) -> None:
    """Write the settings readable by their owner only, replacing the file only once the engine
    accepts them."""
    lines = []
    for key, value in file.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_toml(value)}")
    for key, value in file.items():
        if isinstance(value, dict):
            lines += ["", f"[{key}]", *(f"{k} = {_toml(v)}" for k, v in value.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    new = path.with_name(path.name + ".new")
    descriptor = os.open(new, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")
    os.chmod(new, 0o600)  # the umask may have taken bits off; never more than the owner's
    try:
        load_settings({}, new)
    except LLMNotConfigured as error:
        new.unlink()
        raise SettingsError(str(error).replace(str(new), str(path))) from None
    os.replace(new, path)


def _toml(value: object) -> str:
    """A TOML value: JSON's string escapes are TOML's, so a quote or new line stays inside the string."""
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise SettingsError("Each setting must be text or a whole number.")
    return str(value) if isinstance(value, int) else json.dumps(value, ensure_ascii=False)
