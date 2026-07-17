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


def test_loop_engine_rejects_relative_state_dir():
    import pytest

    cfg = _minimal_config("state")
    with pytest.raises(ValueError, match="state_dir must be absolute"):
        LoopEngine(cfg)


def test_each_symbol_gets_independent_risk_manager(tmp_path: Path):
    cfg = _minimal_config(str(tmp_path))
    cfg["symbols"] = [
        {"name": "#US30", "state_key": "US30", "enabled": True, "timeframe": "M30"},
        {"name": "#US100", "state_key": "US100", "enabled": True, "timeframe": "M30"},
    ]
    StateStore(tmp_path, "US30").update_state(recent_pnls=[1.0, 2.0])
    StateStore(tmp_path, "US100").update_state(recent_pnls=[10.0, 20.0, 30.0])

    engine = LoopEngine(cfg)
    assert len(engine.traders) == 2
    a, b = engine.traders
    assert a.risk is not b.risk
    assert a.risk.recent_pnls == [1.0, 2.0]
    assert b.risk.recent_pnls == [10.0, 20.0, 30.0]

    a.risk.record_trade(3.0)
    assert 3.0 in a.risk.recent_pnls
    assert 3.0 not in b.risk.recent_pnls
