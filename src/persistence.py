"""STATE.md / SKILL.md persistence for per-symbol memory."""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .strategy import StrategyParams

logger = logging.getLogger(__name__)

_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
BACKUP_KEEP = 5


def _lock_for(path: Path) -> threading.RLock:
    """Process-wide RLock keyed by resolved path (shared across StateStore instances)."""
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _atomic_write_text(path: Path, content: str) -> None:
    """Write via temp file + fsync + os.replace so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        # Directory fsync is best-effort (unsupported on some Windows setups).
        pass


class StateStore:
    """Read/write YAML blocks embedded in Markdown STATE / SKILL files."""

    def __init__(self, state_dir: str | Path, state_key: str) -> None:
        self.root = Path(state_dir) / state_key
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "STATE.md"
        self.skill_path = self.root / "SKILL.md"
        self._lock = _lock_for(self.root)
        self._ensure_files()

    def _default_state(self) -> dict[str, Any]:
        return {
            "symbol": state_key_to_symbol(self.root.name),
            "updated_at": None,
            "cumulative_pnl": 0.0,
            "regime_probability": {"trend": 0.5, "mean_reversion": 0.5},
            "params": StrategyParams().as_dict(),
            "last_metrics": {
                "sharpe": None,
                "max_drawdown": None,
                "p_value": None,
                "ic": None,
                "oos_degradation": None,
            },
            "accepted": False,
            "position": {"side": "flat", "lots": 0.0},
            "equity": None,
            "margin": None,
            "equity_peak": None,
            "locked": False,
            "last_maker_run": None,
        }

    def _ensure_files(self) -> None:
        with self._lock:
            if not self.state_path.exists():
                self._write_state_unlocked(self._default_state(), backup=False)
            if not self.skill_path.exists():
                _atomic_write_text(
                    self.skill_path,
                    (
                        f"# SKILL — {self.root.name}\n\n"
                        "Lessons from Checker rejections, Validator failures, and kill-switch events.\n"
                        "Maker and the grid optimizer read this before the next search.\n\n"
                        "## Lessons\n\n- (none yet)\n"
                    ),
                )

    @staticmethod
    def _extract_yaml_block(text: str) -> tuple[dict[str, Any], bool]:
        """Parse STATE YAML. Returns ``(data, ok)``; ``ok=False`` on corruption."""
        match = re.search(r"```ya?ml\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        raw = match.group(1) if match else text
        try:
            data = yaml.safe_load(raw)
        except Exception as exc:
            logger.error("Failed to parse STATE YAML: %s", exc)
            return {}, False
        if data is None:
            return {}, False
        if not isinstance(data, dict):
            logger.error("STATE YAML root is not a mapping: %s", type(data).__name__)
            return {}, False
        return data, True

    def _rotate_backup_unlocked(self) -> None:
        if not self.state_path.exists():
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        backup = self.root / f"STATE.md.{ts}.bak"
        try:
            shutil.copy2(self.state_path, backup)
        except OSError:
            logger.exception("Failed to backup STATE %s", self.state_path)
            return
        backups = sorted(self.root.glob("STATE.md.*.bak"), reverse=True)
        for stale in backups[BACKUP_KEEP:]:
            try:
                stale.unlink()
            except OSError:
                logger.warning("Failed to prune STATE backup %s", stale)

    def _read_state_unlocked(self) -> dict[str, Any]:
        try:
            text = self.state_path.read_text(encoding="utf-8")
        except OSError:
            logger.exception("Failed to read STATE %s; forcing locked=true", self.state_path)
            state = self._default_state()
            state["locked"] = True
            self._write_state_unlocked(state, backup=False)
            return state

        state, ok = self._extract_yaml_block(text)
        if not ok:
            logger.error(
                "STATE corrupt or unreadable at %s; fail-closed locked=true",
                self.state_path,
            )
            state = self._default_state()
            state["locked"] = True
            state["state_corrupt"] = True
            self._write_state_unlocked(state, backup=True)
            return state

        defaults = self._default_state()
        repaired = False
        for key, value in defaults.items():
            if key not in state:
                # Missing lock key after a partial write is fail-closed.
                if key == "locked":
                    state[key] = True
                    repaired = True
                    logger.error(
                        "STATE missing 'locked' at %s; fail-closed locked=true",
                        self.state_path,
                    )
                else:
                    state[key] = value
        if repaired:
            self._write_state_unlocked(state, backup=True)
        return state

    def read_state(self) -> dict[str, Any]:
        with self._lock:
            return self._read_state_unlocked()

    def _write_state_unlocked(self, state: dict[str, Any], *, backup: bool = True) -> None:
        state = dict(state)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        body = yaml.safe_dump(state, sort_keys=False, allow_unicode=True)
        content = f"# STATE — {self.root.name}\n\n```yaml\n{body}```\n"
        if backup:
            self._rotate_backup_unlocked()
        _atomic_write_text(self.state_path, content)

    def write_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._write_state_unlocked(state, backup=True)

    def update_state(self, **kwargs: Any) -> dict[str, Any]:
        """Read-modify-write under the path lock so kill-switch cannot lose ``locked``."""
        with self._lock:
            state = self._read_state_unlocked()
            for k, v in kwargs.items():
                if isinstance(v, dict) and isinstance(state.get(k), dict):
                    merged = dict(state[k])
                    merged.update(v)
                    state[k] = merged
                else:
                    state[k] = v
            self._write_state_unlocked(state, backup=True)
            return state

    def is_locked(self) -> bool:
        # Fail closed: missing/corrupt → True via read_state.
        return bool(self.read_state().get("locked", True))

    def set_locked(self, locked: bool, reason: str = "") -> None:
        with self._lock:
            state = self._read_state_unlocked()
            state["locked"] = bool(locked)
            self._write_state_unlocked(state, backup=True)
        if locked and reason:
            self.append_lesson(f"Kill-switch locked: {reason}")

    def get_params(self) -> StrategyParams:
        state = self.read_state()
        p = state.get("params") or {}
        return StrategyParams(
            long_window=int(p.get("long_window", 240)),
            short_window=int(p.get("short_window", 48)),
            max_hold_bars=int(p.get("max_hold_bars", 16)),
        )

    def read_skills(self) -> list[str]:
        with self._lock:
            return self._read_skills_unlocked()

    def _read_skills_unlocked(self) -> list[str]:
        text = self.skill_path.read_text(encoding="utf-8")
        lessons: list[str] = []
        in_lessons = False
        for line in text.splitlines():
            if line.strip().lower().startswith("## lessons"):
                in_lessons = True
                continue
            if in_lessons:
                if line.startswith("## "):
                    break
                m = re.match(r"\s*-\s+(.*)", line)
                if m:
                    item = m.group(1).strip()
                    if item and item != "(none yet)":
                        lessons.append(item)
        return lessons

    def skills_text(self, max_lessons: int = 40) -> str:
        lessons = self.read_skills()[-max_lessons:]
        if not lessons:
            return "(none yet)"
        return "\n".join(f"- {x}" for x in lessons)

    def append_lesson(self, lesson: str) -> None:
        lesson = lesson.strip()
        if not lesson:
            return
        with self._lock:
            existing = self._read_skills_unlocked()
            if lesson in existing:
                return
            bare = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", lesson)
            for item in existing:
                if re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", item) == bare:
                    return
            existing.append(lesson)
            existing = existing[-200:]
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            lines = [
                f"# SKILL — {self.root.name}",
                "",
                "Lessons from Checker rejections, Validator failures, and kill-switch events.",
                "Maker and the grid optimizer read this before the next search.",
                "",
                "## Lessons",
                "",
            ]
            for item in existing:
                if item.startswith("["):
                    lines.append(f"- {item}")
                else:
                    lines.append(f"- [{ts}] {item}")
            lines.append("")
            _atomic_write_text(self.skill_path, "\n".join(lines))
            logger.info("Appended lesson to %s", self.skill_path)

    def lessons_as_constraints(self) -> dict[str, Any]:
        """
        Heuristically parse lessons into soft optimizer hints.
        Recognizes phrases like 'avoid long_window>260' or 'prefer short_window<=48'.
        """
        hints: dict[str, Any] = {
            "avoid_long_gt": None,
            "avoid_short_lt": None,
            "avoid_hold_gt": None,
            "notes": [],
        }
        for lesson in self.read_skills():
            hints["notes"].append(lesson)
            m = re.search(r"long_window\s*>\s*(\d+)", lesson, re.I)
            if m:
                hints["avoid_long_gt"] = int(m.group(1))
            m = re.search(r"short_window\s*<\s*(\d+)", lesson, re.I)
            if m:
                hints["avoid_short_lt"] = int(m.group(1))
            m = re.search(r"max_hold_bars\s*>\s*(\d+)", lesson, re.I)
            if m:
                hints["avoid_hold_gt"] = int(m.group(1))
        return hints


def state_key_to_symbol(key: str) -> str:
    if key.upper() == "US30":
        return "#US30"
    return key
