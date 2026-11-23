"""Where the engine keeps its files: the library, its search index, the embedding model, the
private folder and the LLM settings (`DATA`).

- `ASCLEPIUS_DATA`, when set, names the folder.
- Run from a source checkout, it is the checkout's `hypothesis/data/` (git ignores it).
- Installed (`uv tool install`), it is a folder of the researcher's own: on a Mac
  `~/Library/Application Support/Asclepius`, on Windows `%LOCALAPPDATA%\\Asclepius`, on Linux
  `$XDG_DATA_HOME/asclepius` (by default `~/.local/share/asclepius`). Never inside the
  installed package, which an upgrade replaces.
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path

PROJECT = "asclepius-hypothesis"  # the source checkout's project name


def data_folder(env: Mapping[str, str], package: Path, system: str, home: Path) -> Path:
    """The data folder for the package installed at `package`, on `system` (`sys.platform`)."""
    if chosen := env.get("ASCLEPIUS_DATA"):
        return home / chosen[2:] if chosen == "~" or chosen.startswith("~/") else Path(chosen)
    if _is_checkout(package.parent):
        return package.parent / "data"
    if system == "darwin":
        return home / "Library" / "Application Support" / "Asclepius"
    if system == "win32":
        return Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local") / "Asclepius"
    return Path(env.get("XDG_DATA_HOME") or home / ".local" / "share") / "asclepius"


def _is_checkout(folder: Path) -> bool:
    try:
        project = tomllib.loads((folder / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return project.get("project", {}).get("name") == PROJECT


DATA = data_folder(os.environ, Path(__file__).resolve().parent, sys.platform, Path.home())
