"""Build the researcher's install: one wheel, `dist/asclepius-<version>-py3-none-any.whl`.

    python3 package/build.py            # from the repository, with git, npm and uv installed

It holds the analysis app (`analysis/asclepius`), the hypothesis engine
(`hypothesis/hypothesis_engine`) and the extractor's browser page, built with `npm run build`
and served by the app at /extractor/. Only files git tracks are copied, so nothing from a data
folder, the samples or the harvester can get in; the wheel is checked for that before it is kept.
Its dependencies are the exact versions in `analysis/uv.lock`, the ones the tests ran with.

A researcher then needs only uv: `uv tool install --python 3.14 asclepius-0.1.0-py3-none-any.whl`
(uv otherwise picks the newest Python, which onnxruntime may not support yet), then
`asclepius` starts the app and `asclepius-library` builds the paper library.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGES = {
    "asclepius": REPO / "analysis" / "asclepius",
    "hypothesis_engine": REPO / "hypothesis" / "hypothesis_engine",
}
EXTRACTOR = REPO / "extractor"
PAGE = "asclepius/web/extractor"  # where the built page goes in the wheel (`asclepius.web.server`)
FILL = "dependencies = []  # filled in by build.py"
REFUSED_PARTS = {"data", "tests", "__pycache__", "node_modules", "private", "samples"}
REFUSED_SUFFIXES = (".csv", ".sqlite", ".pdf", ".json", ".pyc")


def run(command: list[str], cwd: Path) -> str:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True).stdout


def build_page() -> Path:
    """The extractor's built page (`extractor/dist`), built afresh."""
    if not (EXTRACTOR / "node_modules").is_dir():
        run(["npm", "ci"], EXTRACTOR)
    run(["npm", "run", "build"], EXTRACTOR)
    return EXTRACTOR / "dist"


def tracked(folder: Path) -> list[Path]:
    """The files git tracks in `folder`, as they are on disk."""
    names = run(["git", "ls-files", "-z", "--", str(folder.relative_to(REPO))], REPO).split("\0")
    return [REPO / name for name in names if name and (REPO / name).is_file()]


def dependencies() -> list[str]:
    """The app's dependencies at the versions in analysis/uv.lock (the engine's are among them)."""
    exported = run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-hashes",
            "--no-header",
            "--no-annotate",
            "--no-emit-project",
            "--no-emit-package",
            "asclepius-hypothesis",
        ],
        REPO / "analysis",
    )
    lines = [line.strip() for line in exported.splitlines()]
    return [line for line in lines if line and not line.startswith(("#", "-"))]


def check(wheel: Path) -> None:
    """Refuse a wheel holding anything but the two packages' code and the built page."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    if f"{PAGE}/index.html" not in names:
        raise SystemExit(f"{wheel.name}: the extractor page is missing")
    for name in names:
        parts = name.split("/")
        if parts[0] not in PACKAGES and not parts[0].endswith(".dist-info"):
            raise SystemExit(f"{wheel.name}: {name} is not part of the app")
        if REFUSED_PARTS & set(parts) or name.endswith(REFUSED_SUFFIXES):
            raise SystemExit(f"{wheel.name}: {name} looks like data, a test or a build leftover")


def build_wheel(out: Path) -> Path:
    """Build and check the wheel into `out`; its path."""
    page = build_page()
    with tempfile.TemporaryDirectory(prefix="asclepius-package-") as folder:
        stage = Path(folder)
        for name, source in PACKAGES.items():
            for file in tracked(source):
                target = stage / name / file.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
        shutil.copytree(page, stage / PAGE)
        project = (Path(__file__).parent / "pyproject.toml").read_text()
        if FILL not in project:
            raise SystemExit(f"package/pyproject.toml has no line {FILL!r} to fill in")
        listed = "".join(f'    "{d}",\n' for d in dependencies())
        (stage / "pyproject.toml").write_text(project.replace(FILL, f"dependencies = [\n{listed}]"))
        run(["uv", "build", "--wheel", "--out-dir", str(out.resolve())], stage)
    version = tomllib.loads(project)["project"]["version"]
    wheel = out / f"asclepius-{version}-py3-none-any.whl"
    check(wheel)
    return wheel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="package/build.py", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=REPO / "dist", help="folder for the wheel (default dist/)"
    )
    args = parser.parse_args(argv)
    try:
        wheel = build_wheel(args.out)
    except subprocess.CalledProcessError as error:
        sys.exit(f"{' '.join(error.cmd)} failed:\n{error.stderr or error.stdout}")
    print(wheel)


if __name__ == "__main__":
    main()
