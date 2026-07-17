from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.loop_engine import LoopEngine
from src.persistence import StateStore


def _minimal_config(state_dir: str) -> dict:
    return {
        "EXECUTE": False,
        "account_type": "demo",
        "allow_live": False,
        "mt5": {"login": 0, "password": "", "server": "FxPro-Demo", "path": ""},
        "risk": {},
        "strategy": {"long_window": 20, "short_window": 5, "max_hold_bars": 8},
        "paths": {"state_dir": state_dir},
        "symbols": [
            {
                "name": "US30",
                "state_key": "US30",
                "enabled": True,
                "timeframe": "M30",
            }
        ],
        "loop": {"poll_seconds": 30, "review_weekday": 5, "review_hour_utc": 6},
        "kill_switch": {"max_drawdown": 0.5, "poll_seconds": 60},
        "optimizer": {},
        "validator": {"lookback_months": 6},
    }


def test_last_review_date_persisted_across_engine_restart(tmp_path: Path):
    cfg = _minimal_config(str(tmp_path))
    engine = LoopEngine(cfg)
    assert engine._last_review_date is None

    saturday = datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)  # weekday=5
    assert engine.should_review(saturday) is True

    engine._persist_last_review_date("2026-07-11")
    store = StateStore(tmp_path, "US30")
    assert store.read_state().get("last_review_date") == "2026-07-11"

    # Same-day review blocked via persisted STATE (simulates process restart).
    engine2 = LoopEngine(cfg)
    assert engine2._last_review_date == "2026-07-11"
    assert engine2.should_review(saturday) is False
