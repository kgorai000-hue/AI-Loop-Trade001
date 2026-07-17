"""STATE.md / SKILL.md persistence for per-symbol memory."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from .strategy import StrategyParams

logger = logging.getLogger(__name__)


class StateStore:
    """Read/write YAML blocks embedded in Markdown STATE / SKILL files."""

    def __init__(self, state_dir: str | Path, state_key: str) -> None:
        self.root = Path(state_dir) / state_key
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "STATE.md"
        self.skill_path = self.root / "SKILL.md"
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
        if not self.state_path.exists():
            self.write_state(self._default_state())
        if not self.skill_path.exists():
            self.skill_path.write_text(
                f"# SKILL — {self.root.name}\n\n"
                "Lessons from Checker rejections, Validator failures, and kill-switch events.\n"
                "Maker and the grid optimizer read this before the next search.\n\n"
                "## Lessons\n\n- (none yet)\n",
                encoding="utf-8",
            )

    @staticmethod
    def _extract_yaml_block(text: str) -> dict[str, Any]:
        match = re.search(r"```ya?ml\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            try:
                data = yaml.safe_load(text) or {}
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        try:
            data = yaml.safe_load(match.group(1)) or {}
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("Failed to parse STATE YAML: %s", exc)
            return {}

    def read_state(self) -> dict[str, Any]:
        text = self.state_path.read_text(encoding="utf-8")
        state = self._extract_yaml_block(text)
        # Backfill new keys for migrated STATE files
        defaults = self._default_state()
        for key, value in defaults.items():
            if key not in state:
                state[key] = value
        return state

    def write_state(self, state: dict[str, Any]) -> None:
        state = dict(state)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        body = yaml.safe_dump(state, sort_keys=False, allow_unicode=True)
        content = (
            f"# STATE — {self.root.name}\n\n"
            f"```yaml\n{body}```\n"
        )
        self.state_path.write_text(content, encoding="utf-8")

    def update_state(self, **kwargs: Any) -> dict[str, Any]:
        state = self.read_state()
        for k, v in kwargs.items():
            if isinstance(v, dict) and isinstance(state.get(k), dict):
                merged = dict(state[k])
                merged.update(v)
                state[k] = merged
            else:
                state[k] = v
        self.write_state(state)
        return state

    def is_locked(self) -> bool:
        return bool(self.read_state().get("locked", False))

    def set_locked(self, locked: bool, reason: str = "") -> None:
        self.update_state(locked=bool(locked))
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
        existing = self.read_skills()
        if lesson in existing:
            return
        # Deduplicate by suffix after timestamp
        bare = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", lesson)
        for item in existing:
            if re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", item) == bare:
                return
        existing.append(lesson)
        # Cap growth
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
        self.skill_path.write_text("\n".join(lines), encoding="utf-8")
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
