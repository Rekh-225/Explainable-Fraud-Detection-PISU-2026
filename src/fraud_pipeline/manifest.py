"""Run manifest: everything needed to audit or repeat a run.

The manifest is written last so that it can carry SHA-256 digests of every
other artifact in the run directory. ``verify_run`` re-hashes the artifacts
and reports mismatches.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__

MANIFEST_NAME = "run_manifest.json"
TRACKED_PACKAGES = ("pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "joblib")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def environment_info() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }


def git_info(repo_root: Path) -> dict[str, Any]:
    """Commit hash and dirty flag when the run happens inside a git checkout."""

    def _git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "branch": None}
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def artifact_checksums(run_dir: Path, exclude: tuple[str, ...] = (MANIFEST_NAME,)) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in exclude:
            checksums[path.relative_to(run_dir).as_posix()] = sha256_path(path)
    return checksums


def build_manifest(
    run_id: str,
    run_dir: Path,
    repo_root: Path,
    dataset: dict[str, Any],
    features: list[str],
    duplicate_policy: dict[str, Any],
    split: dict[str, Any],
    models: dict[str, Any],
    threshold_policy: dict[str, Any],
    selection: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "pipeline_version": __version__,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        "features": features,
        "duplicate_policy": duplicate_policy,
        "split": split,
        "models": models,
        "threshold_policy": threshold_policy,
        "selection": selection,
        "config": config,
        "environment": environment_info(),
        "code": git_info(repo_root),
        "artifacts": artifact_checksums(run_dir),
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = run_dir / MANIFEST_NAME
    save_json(path, manifest)
    return path


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def verify_run(run_dir: Path) -> dict[str, list[str]]:
    """Compare recorded artifact digests with the files on disk."""
    manifest = load_manifest(run_dir)
    recorded: dict[str, str] = manifest["artifacts"]
    actual = artifact_checksums(run_dir)
    return {
        "missing": sorted(set(recorded) - set(actual)),
        "unexpected": sorted(set(actual) - set(recorded)),
        "modified": sorted(k for k in recorded.keys() & actual.keys() if recorded[k] != actual[k]),
    }
