"""Where the engine keeps its files: the source checkout's `data/`, else a per-user folder."""

from __future__ import annotations

from pathlib import Path

from hypothesis_engine.home import data_folder

HOME = Path("/home/ana")


def checkout(tmp_path: Path) -> Path:
    """A package folder inside a source checkout (its project file beside it)."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "asclepius-hypothesis"\n')
    package = tmp_path / "hypothesis_engine"
    package.mkdir()
    return package


def installed(tmp_path: Path) -> Path:
    """A package folder installed into site-packages, beside other packages."""
    package = tmp_path / "site-packages" / "hypothesis_engine"
    package.mkdir(parents=True)
    return package


def test_a_source_checkout_keeps_its_data_folder(tmp_path):
    assert data_folder({}, checkout(tmp_path), "darwin", HOME) == tmp_path / "data"


def test_another_projects_file_beside_an_installed_package_is_not_a_checkout(tmp_path):
    package = installed(tmp_path)
    (package.parent / "pyproject.toml").write_text('[project]\nname = "something-else"\n')
    assert data_folder({}, package, "linux", HOME) == HOME / ".local/share/asclepius"


def test_installed_on_a_mac(tmp_path):
    assert (
        data_folder({}, installed(tmp_path), "darwin", HOME) == HOME / "Library/Application Support/Asclepius"
    )


def test_installed_on_linux_follows_xdg(tmp_path):
    package = installed(tmp_path)
    assert data_folder({}, package, "linux", HOME) == HOME / ".local/share/asclepius"
    assert data_folder({"XDG_DATA_HOME": "/data"}, package, "linux", HOME) == Path("/data/asclepius")


def test_installed_on_windows(tmp_path):
    package = installed(tmp_path)
    assert data_folder({"LOCALAPPDATA": r"C:\Users\ana\AppData\Local"}, package, "win32", HOME) == (
        Path(r"C:\Users\ana\AppData\Local") / "Asclepius"
    )
    assert data_folder({}, package, "win32", HOME) == HOME / "AppData/Local/Asclepius"


def test_asclepius_data_overrides_both(tmp_path):
    env = {"ASCLEPIUS_DATA": "~/papers"}
    assert data_folder(env, checkout(tmp_path), "darwin", HOME) == HOME / "papers"
    assert data_folder(env, installed(tmp_path), "linux", HOME) == HOME / "papers"
