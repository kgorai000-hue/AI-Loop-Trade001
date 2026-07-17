"""Resolve config-relative filesystem paths against a stable base directory.

Relative ``state_dir`` / ``log_dir`` must not depend on process cwd (Task Scheduler
often leaves "Start in" empty). Callers pass the config file's parent, or the
project root when no config path is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_under(path: str | Path, base: Path) -> Path:
    """Return an absolute path; relative values join ``base`` then resolve."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(base) / p
    return p.resolve()


def normalize_config_paths(cfg: dict[str, Any], base: str | Path) -> dict[str, Any]:
    """In-place: rewrite known relative path settings to absolute paths.

    Normalized keys:
    - ``paths.state_dir`` (default ``state``)
    - ``loop.log_dir`` (default ``logs``)
    - ``mt5.path`` when non-empty and relative
    """
    root = Path(base).resolve()

    paths = cfg.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("config paths: must be a mapping")
    state_dir = paths.get("state_dir", "state")
    paths["state_dir"] = str(resolve_under(state_dir, root))

    loop = cfg.setdefault("loop", {})
    if not isinstance(loop, dict):
        raise ValueError("config loop: must be a mapping")
    log_dir = loop.get("log_dir", "logs")
    loop["log_dir"] = str(resolve_under(log_dir, root))

    mt5 = cfg.setdefault("mt5", {})
    if not isinstance(mt5, dict):
        raise ValueError("config mt5: must be a mapping")
    mt5_path = mt5.get("path") or ""
    if str(mt5_path).strip():
        mt5["path"] = str(resolve_under(mt5_path, root))
    else:
        mt5["path"] = ""

    return cfg
