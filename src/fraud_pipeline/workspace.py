"""Workspace resolution and safe run identifiers.

The package may be installed as a wheel far away from the project checkout,
so defaults must never be derived from the module's own location. The
workspace root is, in order of precedence:

1. an explicit ``root`` argument (CLI ``--workspace-root``),
2. the ``FRAUD_PIPELINE_ROOT`` environment variable,
3. the current working directory.

Datasets, ``output/runs``, ``output/analysis`` and the UI demo directory are
resolved beneath that root unless overridden individually.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT_ENV_VAR = "FRAUD_PIPELINE_ROOT"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def workspace_root(root: str | os.PathLike | None = None) -> Path:
    """Resolve the workspace root (explicit > environment > current directory)."""
    candidate = root or os.environ.get(ROOT_ENV_VAR) or Path.cwd()
    return Path(candidate).expanduser().resolve()


def default_output_root(root: str | os.PathLike | None = None) -> Path:
    return workspace_root(root) / "output" / "runs"


def historical_kaggle_csv(root: str | os.PathLike | None = None) -> Path:
    return workspace_root(root) / "data" / "raw" / "creditcardfraud" / "creditcard.csv"


def historical_results_dir(root: str | os.PathLike | None = None) -> Path:
    return workspace_root(root) / "output" / "analysis"


def ui_demo_root(root: str | os.PathLike | None = None) -> Path:
    return workspace_root(root) / "output" / "ui_demo"


def validate_run_id(run_id: str) -> str:
    """Accept only a single filename-safe path component.

    Allowed: 1-100 characters from ``A-Z a-z 0-9 . _ -`` starting with an
    alphanumeric. Rejected: separators (``/`` ``\\``), ``..`` anywhere,
    absolute, drive-qualified (``C:``) and UNC paths, names ending in a dot
    or space, and Windows reserved device names (``CON``, ``NUL``, ``COM1``...).
    """
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
        raise ValueError(
            f"Invalid run id {run_id!r}: use 1-100 characters from A-Z, a-z, 0-9, '.', '_', '-' "
            "starting with a letter or digit (no path separators, drive letters or '..')."
        )
    if ".." in run_id or run_id.endswith((".", " ")):
        raise ValueError(f"Invalid run id {run_id!r}: '..' and trailing dots are not allowed.")
    if run_id.split(".")[0].upper() in _WINDOWS_RESERVED:
        raise ValueError(f"Invalid run id {run_id!r}: reserved device name.")
    return run_id


def contained_run_dir(output_root: Path, run_id: str) -> Path:
    """Return ``output_root/run_id`` after validating the id and containment."""
    validate_run_id(run_id)
    root = Path(output_root).expanduser().resolve()
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root or not run_dir.is_relative_to(root):
        raise ValueError(f"Run directory {run_dir} escapes output root {root}")
    return run_dir
