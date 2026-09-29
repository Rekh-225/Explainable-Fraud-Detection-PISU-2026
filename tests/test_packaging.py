"""Package metadata, requirements files and Streamlit configuration stay consistent."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pins(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(("#", "-r"))
    }


def test_pyproject_dependencies_match_requirements_files():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert set(project["dependencies"]) == _pins(ROOT / "requirements.txt")
    assert set(project["optional-dependencies"]["ui"]) == _pins(ROOT / "requirements-ui.txt")
    assert set(project["optional-dependencies"]["dev"]) == _pins(ROOT / "requirements-dev.txt")
    assert all("==" in pin for pin in project["dependencies"])
    assert (ROOT / "requirements-ui.txt").read_text(encoding="utf-8").startswith("-r requirements.txt")
    assert (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").startswith("-r requirements-ui.txt")


def test_package_version_matches_pyproject():
    from fraud_pipeline import __version__

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == __version__


def test_lock_file_contains_every_pin():
    lock = _pins(ROOT / "requirements.lock.txt")
    for req in ("requirements.txt", "requirements-ui.txt", "requirements-dev.txt"):
        assert _pins(ROOT / req) <= lock, req


def test_streamlit_project_config_enforces_local_only_settings():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["address"] == "127.0.0.1"
    assert config["server"]["enableCORS"] is True
    assert config["server"]["enableXsrfProtection"] is True
    assert config["client"]["toolbarMode"] == "viewer"
