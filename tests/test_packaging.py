"""Release metadata stays consistent."""

import re
import tomllib
from pathlib import Path

import crisispaths as cp

ROOT = Path(__file__).resolve().parents[1]


def test_versions_agree_and_marker_exists():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    meta = tomllib.loads(text)
    proj = meta["project"]
    assert proj["version"] == cp.__version__ == "0.1.0"
    assert proj["requires-python"] == ">=3.12"
    assert (ROOT / "src" / "crisispaths" / "py.typed").exists()
    assert proj["urls"]["Homepage"] == "https://github.com/jcgtanaka/crisispaths"
    assert proj["scripts"]["crisispaths"] == "crisispaths.cli:main"
    assert "pdf" in proj["optional-dependencies"]
    assert meta["tool"]["ruff"]["target-version"] == "py312"
    assert not any("3.10" in c or "3.11" in c for c in proj["classifiers"])
    assert (ROOT / "src" / "crisispaths" / "schema" / "summary.schema.json").exists()


def test_pyproject_tables_are_well_formed():
    """Runtime dependencies must sit in [project], and every URL must be a string.

    A key added after [project.urls] silently becomes a URL entry; the editable install
    still works (dependencies are already present) but the built wheel declares none.
    """
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    deps = {re.split(r"[<>=!~ \[]", d, maxsplit=1)[0].lower() for d in meta["dependencies"]}
    assert {"numpy", "pandas", "scipy", "pyyaml", "platformdirs"} <= deps
    assert all(isinstance(v, str) for v in meta["urls"].values())
    assert set(meta["urls"]) == {"Homepage", "Repository", "Issues", "Changelog"}
    assert meta["scripts"]["crisispaths"] == "crisispaths.cli:main"
    assert {"data", "report", "dev"} <= set(meta["optional-dependencies"])


def test_docs_exist_and_changelog_has_release_sections():
    for name in ("docs/methodology.md", "docs/limitations.md", "RELEASING.md"):
        assert (ROOT / name).read_text(encoding="utf-8").strip()
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for head in ("## 0.1.0", "### Added", "### Changed", "### Fixed", "### Known limitations",
                 "### Planned"):
        assert head in log
    assert "GARCH" in log.split("### Planned")[1]
