"""Load MT5 credentials from environment / gitignored files -- never from tracked config."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SECRETS_FILENAME = "secrets.yaml"


def load_dotenv(path: str | Path) -> bool:
    """Load ``KEY=VALUE`` lines into ``os.environ`` if the key is not already set.

    Returns True when the file existed. Does not override existing variables
    (Task Scheduler / shell exports win).
    """
    path = Path(path)
    if not path.is_file():
        return False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value
    return True


def _load_secrets_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a mapping")
    return data


def apply_mt5_credentials(
    cfg: dict[str, Any],
    *,
    config_dir: str | Path,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Overlay MT5 login/password (and optional server) from secrets / env.

    Precedence (highest last): ``secrets.yaml`` -> environment variables.
    Non-empty ``mt5.password`` / ``mt5.login`` in the tracked config are cleared
    with a warning so a mistaken commit cannot become the live credential source.
    """
    env = environ if environ is not None else os.environ
    mt5 = cfg.setdefault("mt5", {})
    if not isinstance(mt5, dict):
        raise ValueError("config mt5: must be a mapping")

    if str(mt5.get("password") or "").strip():
        logger.warning(
            "Ignoring mt5.password from config (do not store secrets in tracked "
            "files). Use MT5_PASSWORD or %s instead.",
            SECRETS_FILENAME,
        )
        mt5["password"] = ""

    # login: 0 means "attach to already-logged-in terminal"; treat any other
    # value in tracked config as a misplaced secret and clear it.
    raw_login = mt5.get("login", 0)
    try:
        tracked_login = int(raw_login or 0)
    except (TypeError, ValueError):
        tracked_login = 0
    if tracked_login:
        logger.warning(
            "Ignoring mt5.login=%s from config. Use MT5_LOGIN or %s instead.",
            tracked_login,
            SECRETS_FILENAME,
        )
        mt5["login"] = 0
    else:
        mt5["login"] = 0

    secrets_path = Path(config_dir) / SECRETS_FILENAME
    file_data = _load_secrets_file(secrets_path)
    file_mt5 = file_data.get("mt5") if isinstance(file_data.get("mt5"), dict) else {}
    if file_mt5.get("login") not in (None, "", 0, "0"):
        mt5["login"] = int(file_mt5["login"])
    if file_mt5.get("password") is not None and str(file_mt5.get("password")) != "":
        mt5["password"] = str(file_mt5["password"])
    if file_mt5.get("server"):
        mt5["server"] = str(file_mt5["server"])

    if env.get("MT5_LOGIN", "").strip():
        mt5["login"] = int(env["MT5_LOGIN"].strip())
    if "MT5_PASSWORD" in env:
        mt5["password"] = env.get("MT5_PASSWORD") or ""
    if env.get("MT5_SERVER", "").strip():
        mt5["server"] = env["MT5_SERVER"].strip()

    return cfg
