from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import AccountConfig, Backtester, FillModel
from src.intelligence import IntelligenceLoop, IntelligenceOutcome
from src.risk import CostModel
from src.strategy import StrategyParams
from src.validator import StrategyValidator, ValidationResult, ValidatorConfig


class Store:
    def __init__(self):
        self.state = {"locked": False, "params": StrategyParams().as_dict()}
        self.lessons = []

    def read_state(self):
        return dict(self.state)

    def get_params(self):
        return StrategyParams(**{k: self.state["params"][k] for k in ("long_window", "short_window", "max_hold_bars")})

    def update_state(self, **kwargs):
        self.state.update(kwargs)

    def skills_text(self):
        return "(none)"

    def append_lesson(self, lesson):
        self.lessons.append(lesson)

    def lessons_as_constraints(self):
        return {}


def _frame(n=400):
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 0.2, size=n))
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC"),
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
        }
    )


def test_run_nested_holdout_not_used_for_fold_selection(monkeypatch):
    """Fold winners are chosen before holdout; holdout only gates once."""
    store = Store()
    app = {
        "risk": {"min_cost_bps": 0},
        "backtest": {
            "limit_fills": False,
            "account_sizing": False,
        },
        "validator": {
            "holdout_fraction": 0.2,
            "wf_folds": 2,
            "wf_min_train_fraction": 0.4,
            "is_fraction": 0.7,
            "max_drawdown": 1.0,
            "sharpe_min": -10.0,
            "sharpe_max": 100.0,
            "p_value_max": 1.0,
            "oos_degradation_max": 1.0,
        },
        "optimizer": {
            "nested_inner": "grid",
            "long_windows": [20],
            "short_windows": [5],
            "max_hold_bars": [8],
        },
        "anthropic": {},
    }
    intel = IntelligenceLoop(app, store, cost_model=CostModel(min_cost_bps=0.0))
    intel.backtester = Backtester(
        cost_model=CostModel(min_cost_bps=0.0),
        fill_model=FillModel(enabled=False),
        account=AccountConfig(enabled=False),
    )
    intel.validator = StrategyValidator(
        ValidatorConfig(
            max_drawdown=1.0,
            sharpe_min=-10.0,
            sharpe_max=100.0,
            p_value_max=1.0,
            oos_degradation_max=1.0,
            is_fraction=0.7,
            holdout_fraction=0.2,
            wf_folds=2,
            wf_min_train_fraction=0.4,
        )
    )

    seen_train_ends = []
    params = StrategyParams(long_window=20, short_window=5, max_hold_bars=8)

    def fake_inner(train):
        seen_train_ends.append(train["time"].iloc[-1])
        return IntelligenceOutcome(
            best_params=params,
            best_validation=ValidationResult(accepted=True, sharpe=2.0),
            tried=1,
            accepted_count=1,
            path="grid_fallback",
            rankings=[{"params": params.as_dict(), "accepted": True, "sharpe": 2.0}],
        )

    monkeypatch.setattr(intel, "_run_inner", fake_inner)

    df = _frame(400)
    holdout_start = df["time"].iloc[int(400 * 0.8)]
    outcome = intel.run_nested(df)

    assert outcome.path == "nested"
    assert outcome.fold_results
    assert all(t < holdout_start for t in seen_train_ends)
    assert outcome.holdout_validation is not None
    # Holdout gate ran; fold_results exist independently of holdout accept/reject.
    assert len(outcome.fold_results) >= 1


def test_validate_can_skip_oos_gate_for_holdout():
    v = StrategyValidator(
        ValidatorConfig(
            max_drawdown=1.0,
            sharpe_min=-100.0,
            sharpe_max=100.0,
            p_value_max=1.0,
            oos_degradation_max=0.0,  # would reject any positive deg if applied
            is_fraction=0.7,
        )
    )
    bt = Backtester(
        cost_model=CostModel(min_cost_bps=0.0),
        fill_model=FillModel(enabled=False),
        account=AccountConfig(enabled=False),
    )
    df = _frame(120)
    params = StrategyParams(long_window=10, short_window=3, max_hold_bars=20)
    val, _ = v.validate(df, params, backtester=bt, apply_oos_gate=False)
    assert val.oos_degradation == 0.0
    assert val.accepted is True
