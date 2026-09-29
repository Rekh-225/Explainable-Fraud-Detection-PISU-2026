"""Run manifest: everything needed to audit or repeat a run.

The manifest is written last so that it can carry SHA-256 digests of every
other artifact in the run directory. ``verify_run`` re-hashes the artifacts
and reports mismatches.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
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


def sha256_path(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 so large artifacts are never read into memory at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_MANIFEST_KEYS = (
    "manifest_version", "pipeline_version", "run_id", "dataset", "features", "split",
    "models", "threshold_policy", "selection", "environment", "code", "artifacts",
)


class ManifestError(ValueError):
    """Raised when a manifest is structurally invalid."""


_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def is_safe_artifact_path(key: Any) -> bool:
    """Relative, forward-slash-only path with no traversal on either platform convention.

    Rejects backslashes (so ``..\\x`` cannot slip past a ``/``-only check),
    leading slashes, drive-qualified (``C:``) and UNC (``//server``) forms,
    empty / ``.`` / ``..`` components and control characters.
    """
    if not isinstance(key, str) or not key or len(key) > 512:
        return False
    if "\\" in key or "\x00" in key or any(ord(c) < 32 for c in key):
        return False
    if key.startswith("/") or _DRIVE_RE.match(key) or key.startswith("//"):
        return False
    parts = key.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def validate_manifest_structure(manifest: Any) -> dict[str, Any]:
    """Check top-level keys and the shape of ``artifacts`` before anything is consumed."""
    if not isinstance(manifest, dict):
        raise ManifestError("Manifest must be a JSON object")
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ManifestError(f"Manifest is missing keys: {missing}")
    if not isinstance(manifest["run_id"], str) or not manifest["run_id"]:
        raise ManifestError("Manifest run_id must be a non-empty string")
    if not isinstance(manifest["manifest_version"], int):
        raise ManifestError("Manifest manifest_version must be an integer")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ManifestError("Manifest 'artifacts' must be a non-empty object of path -> sha256")
    for key, value in artifacts.items():
        if not is_safe_artifact_path(key):
            raise ManifestError(f"Manifest artifact path {key!r} is not a safe relative path")
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            raise ManifestError(f"Manifest artifact {key!r} has a malformed sha256 digest")
    dataset = manifest["dataset"]
    if not isinstance(dataset, dict) or not _SHA256_RE.match(str(dataset.get("sha256", ""))):
        raise ManifestError("Manifest dataset.sha256 is missing or malformed")
    if not isinstance(manifest["features"], list) or not manifest["features"]:
        raise ManifestError("Manifest 'features' must be a non-empty list")
    return manifest


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
    """Code provenance for the run.

    The commit is recorded only when the *running package* lives inside the
    git checkout at ``repo_root``. An installed wheel executed from some
    unrelated repository's directory records the package version and its
    installed location instead of that repository's commit.
    """
    package_dir = Path(__file__).resolve().parent
    base = {
        "package_version": __version__,
        "package_location": str(package_dir),
        "commit": None,
        "dirty": None,
        "branch": None,
        "note": None,
    }

    def _git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    toplevel = _git("rev-parse", "--show-toplevel")
    if toplevel is None:
        base["note"] = "workspace is not a git checkout"
        return base
    try:
        inside = package_dir.is_relative_to(Path(toplevel).resolve())
    except OSError:
        inside = False
    tracked = bool(_git("ls-files", "--", str(package_dir / "__init__.py"))) if inside else False
    if not tracked:
        base["note"] = (
            f"package runs from {package_dir}, which is not tracked source of the git checkout at "
            f"{toplevel}; that repository's commit is not attributed to this run"
        )
        return base
    status = _git("status", "--porcelain", "--untracked-files=no")
    base.update(
        commit=_git("rev-parse", "HEAD"),
        dirty=bool(status) if status is not None else None,
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
    )
    return base


def artifact_checksums(run_dir: Path, exclude: tuple[str, ...] = (MANIFEST_NAME,)) -> dict[str, str]:
    """Digest every regular file under ``run_dir`` that resolves inside it (symlink escapes skipped)."""
    run_dir = Path(run_dir)
    root = run_dir.resolve()
    checksums: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in exclude:
            continue
        try:
            if not path.resolve().is_relative_to(root):
                continue
        except OSError:
            continue
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


MAX_MANIFEST_BYTES = 8 << 20


def load_manifest(run_dir: Path, validate: bool = False) -> dict[str, Any]:
    path = Path(run_dir) / MANIFEST_NAME
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ManifestError(f"{MANIFEST_NAME} exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{MANIFEST_NAME} is not valid JSON: {exc}") from exc
    return validate_manifest_structure(manifest) if validate else manifest


def verify_run(run_dir: Path, required: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """Compare recorded artifact digests with the files on disk.

    ``missing``     recorded in the manifest but absent on disk
    ``modified``    recorded and present but digest differs
    ``unexpected``  present on disk but not recorded (never verified; callers
                    must not consume them as trusted artifacts)
    ``unlisted_required``  files in ``required`` that the manifest does not
                    record at all - a required artifact without a checksum
                    entry escapes verification and must be rejected
    """
    manifest = load_manifest(run_dir, validate=True)
    recorded: dict[str, str] = manifest["artifacts"]
    actual = artifact_checksums(run_dir)
    return {
        "missing": sorted(set(recorded) - set(actual)),
        "unexpected": sorted(set(actual) - set(recorded)),
        "modified": sorted(k for k in recorded.keys() & actual.keys() if recorded[k] != actual[k]),
        "unlisted_required": sorted(name for name in required if name not in recorded),
    }
