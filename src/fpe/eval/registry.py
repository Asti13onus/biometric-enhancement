"""Append-only registry of every evaluation this project has ever run.

The rule this enforces (PROJECT_RULES.md non-negotiable 2): no number reaches the
thesis that a script did not produce. Every row carries enough provenance to
answer "which code, which config, which data produced this?" months later --
git SHA and dirty flag, a hash of the config, and a hash of the dataset
manifest.

Rows are appended, never edited or deleted. A wrong row is superseded by a
later one, not rewritten; the history is the point.

    from fpe.eval.registry import log_run

    log_run(
        experiment="baseline-nbis",
        dataset="fvc2004_db1_b",
        method="none",
        metrics={"eer": 0.0721, "eer_ci_low": 0.051, "eer_ci_high": 0.098},
        config={"matcher": "bozorth3", "extractor": "mindtct"},
    )
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = ["log_run", "hash_config", "hash_file", "git_state", "REGISTRY_PATH"]

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / "results" / "registry" / "runs.jsonl"


def _canonical(obj: Any) -> str:
    """Stable JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_config(config: Mapping[str, Any] | None) -> str | None:
    """Content hash of a config mapping, insensitive to key order."""
    if config is None:
        return None
    return hashlib.sha256(_canonical(config).encode("utf-8")).hexdigest()[:16]


def hash_file(path: str | os.PathLike[str] | None) -> str | None:
    """Content hash of a file -- used for dataset manifests and config YAMLs."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def git_state() -> dict[str, Any]:
    """Current commit and whether the working tree is dirty.

    A dirty tree means the row is not reproducible from the SHA alone, so it
    is recorded rather than silently ignored.
    """

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "git_sha": sha,
        "git_dirty": bool(status) if status is not None else None,
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def log_run(
    *,
    experiment: str,
    metrics: Mapping[str, Any],
    dataset: str | None = None,
    method: str | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    notes: str | None = None,
    extra: Mapping[str, Any] | None = None,
    registry_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Append one evaluation to the registry and return the row written.

    `experiment` and `metrics` are required; everything else is provenance
    that should be supplied whenever it exists.
    """
    if not experiment:
        raise ValueError("experiment must be a non-empty name")
    if not metrics:
        raise ValueError("refusing to log a run with no metrics")

    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": experiment,
        "dataset": dataset,
        "method": method,
        "metrics": dict(metrics),
        "config": dict(config) if config is not None else None,
        "config_hash": hash_config(config),
        "config_file_hash": hash_file(config_path),
        "manifest_hash": hash_file(manifest_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        **git_state(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "argv": " ".join(sys.argv),
    }
    if notes:
        row["notes"] = notes
    if extra:
        row["extra"] = dict(extra)

    path = Path(registry_path) if registry_path else REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(_canonical(row) + "\n")
    return row


def read_runs(registry_path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Every row in the registry, oldest first. Malformed lines are skipped loudly."""
    path = Path(registry_path) if registry_path else REGISTRY_PATH
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"{path}:{n}: skipping malformed registry line", file=sys.stderr)
    return rows
