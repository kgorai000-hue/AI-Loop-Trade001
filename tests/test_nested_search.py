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
        self.state = {
            "locked": False,
            "params": StrategyParams().as_dict(),
            "evaluated_rolling_oos": [],
            "last_rolling_oos": None,
            "last_review_date": None,
            "last_metrics": {},
        }
        self.lessons = []

    def read_state(self):
        return dict(self.state)

    def get_params(self):
        return StrategyParams(
            **{k: self.state["params"][k] for k in ("long_window", "short_window", "max_hold_bars")}
        )

    def update_state(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict) and isinstance(self.state.get(k), dict):
                merged = dict(self.state[k])
                merged.update(v)
                self.state[k] = merged
            else:
                self.state[k] = v

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


def _make_intel(store, rolling_oos_fraction=0.2):
    app = {
        "risk": {"min_cost_bps": 0},
        "backtest": {"limit_fills": False, "account_sizing": False},
        "validator": {
            "rolling_oos_fraction": rolling_oos_fraction,
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
    intel = IntelligenceLoop(app, store, cost_model=CostModel(round_trip_floor=0.0))
    intel.backtester = Backtester(
        cost_model=CostModel(round_trip_floor=0.0),
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
            rolling_oos_fraction=rolling_oos_fraction,
            wf_folds=2,
            wf_min_train_fraction=0.4,
            min_trades=1,
            min_oos_trades=1,
            min_regime_trades=0,
            block_bootstrap_reps=0,
            multiple_testing="none",
            min_dsr=0.0,
            pbo_enabled=False,
            pvalue_method="hac",
        )
    )
    return intel


def test_run_nested_rolling_oos_not_used_for_fold_selection(monkeypatch):
    """Fold winners are chosen before rolling OOS; gate only once on new window."""
    store = Store()
    intel = _make_intel(store, rolling_oos_fraction=0.2)

    seen_train_ends = []
    fold_params = StrategyParams(long_window=20, short_window=5, max_hold_bars=8)
    final_params = StrategyParams(long_window=24, short_window=6, max_hold_bars=8)
    calls = {"n": 0}

    def fake_inner(train):
        seen_train_ends.append(train["time"].iloc[-1])
        calls["n"] += 1
        params = final_params if calls["n"] > 2 else fold_params
        return IntelligenceOutcome(
            best_params=params,
            best_validation=ValidationResult(accepted=True, sharpe=2.0),
            tried=1,
            accepted_count=1,
            path="grid_fallback",
            rankings=[{"params": params.as_dict(), "accepted": True, "sharpe": 2.0}],
        )

    segment_calls = []

    def fake_segment(train, test, params):
        segment_calls.append(
            {
                "train_end": train["time"].iloc[-1],
                "test_start": test["time"].iloc[0],
                "params": params.as_dict(),
            }
        )
        return type(
            "BT",
            (),
            {
                "report": type(
                    "R",
                    (),
                    {
                        "sharpe": 1.5,
                        "max_drawdown": 0.01,
                        "n_trades": 10,
                        "p_value": 0.01,
                        "ic": 0.0,
                    },
                )(),
                "trades": [],
                "bar_returns": pd.Series([0.0] * len(test)),
                "signals": pd.Series([0] * len(test)),
                "params": params,
                "regimes": None,
            },
        )()

    def fake_validate_reports(*_args, **_kwargs):
        return ValidationResult(accepted=True, sharpe=1.2, reasons=[])

    monkeypatch.setattr(intel, "_run_inner", fake_inner)
    monkeypatch.setattr(intel, "_run_on_segment", fake_segment)
    monkeypatch.setattr(intel.validator, "validate_reports", fake_validate_reports)

    df = _frame(400)
    holdout_start = df["time"].iloc[int(400 * 0.8)]
    outcome = intel.run_nested(df)

    assert outcome.path == "nested"
    assert outcome.fold_results
    assert all(t < holdout_start for t in seen_train_ends)
    assert outcome.rolling_oos_validation is not None
    assert outcome.rolling_oos_validation.accepted
    assert outcome.rolling_oos_window is not None
    assert len(outcome.fold_results) >= 1
    assert outcome.best_params is not None
    assert outcome.best_params.as_dict() == final_params.as_dict()
    # Maker-facing validation is search-side, not rolling OOS sharpe.
    assert outcome.best_validation is not None
    assert outcome.best_validation.sharpe == 2.0
    assert outcome.outer_mean_sharpe == 1.5
    assert outcome.outer_median_sharpe == 1.5
    assert calls["n"] == 3  # 2 folds + 1 final pre-rolling-OOS re-search
    fold_param_sets = [c["params"] for c in segment_calls[:-1]]
    assert all(p == fold_params.as_dict() for p in fold_param_sets)
    assert segment_calls[-1]["params"] == final_params.as_dict()
    # Rolling OOS gate failures/success must not enter SKILL.
    assert store.lessons == []


def test_run_nested_empty_rolling_oos_is_fail_closed(monkeypatch):
    """Empty rolling OOS must reject optimization (no skip of final gate)."""
    store = Store()
    intel = _make_intel(store, rolling_oos_fraction=0.0)
    params = StrategyParams(long_window=20, short_window=5, max_hold_bars=8)

    def fake_inner(_train):
        return IntelligenceOutcome(
            best_params=params,
            best_validation=ValidationResult(accepted=True, sharpe=2.0),
            tried=1,
            accepted_count=1,
            path="grid_fallback",
            rankings=[{"params": params.as_dict(), "accepted": True, "sharpe": 2.0}],
        )

    def fake_segment(train, test, params):
        return type(
            "BT",
            (),
            {
                "report": type(
                    "R",
                    (),
                    {
                        "sharpe": 1.5,
                        "max_drawdown": 0.01,
                        "n_trades": 10,
                        "p_value": 0.01,
                        "ic": 0.0,
                    },
                )(),
                "trades": [],
                "bar_returns": pd.Series([0.0] * max(len(test), 1)),
                "signals": pd.Series([0] * max(len(test), 1)),
                "params": params,
                "regimes": None,
            },
        )()

    monkeypatch.setattr(intel, "_run_inner", fake_inner)
    monkeypatch.setattr(intel, "_run_on_segment", fake_segment)

    outcome = intel.run_nested(_frame(400))
    assert outcome.best_params is None
    assert outcome.rolling_oos_validation is not None
    assert outcome.rolling_oos_validation.accepted is False
    assert any("empty rolling OOS" in r for r in outcome.rolling_oos_validation.reasons)
    # Must not feed rolling OOS result into SKILL.
    assert not any("rolling OOS" in str(x).lower() or "holdout" in str(x).lower() for x in store.lessons)


def test_run_nested_rejects_reused_rolling_oos_window(monkeypatch):
    """Bars already used as rolling OOS cannot re-enter the gate."""
    store = Store()
    intel = _make_intel(store, rolling_oos_fraction=0.2)
    params = StrategyParams(long_window=20, short_window=5, max_hold_bars=8)

    def fake_inner(_train):
        return IntelligenceOutcome(
            best_params=params,
            best_validation=ValidationResult(accepted=True, sharpe=2.0),
            tried=1,
            accepted_count=1,
            path="grid_fallback",
            rankings=[{"params": params.as_dict(), "accepted": True, "sharpe": 2.0}],
        )

    def fake_segment(train, test, params):
        return type(
            "BT",
            (),
            {
                "report": type(
                    "R",
                    (),
                    {
                        "sharpe": 1.5,
                        "max_drawdown": 0.01,
                        "n_trades": 10,
                        "p_value": 0.01,
                        "ic": 0.0,
                    },
                )(),
                "trades": [],
                "bar_returns": pd.Series([0.0] * max(len(test), 1)),
                "signals": pd.Series([0] * max(len(test), 1)),
                "params": params,
                "regimes": None,
            },
        )()

    monkeypatch.setattr(intel, "_run_inner", fake_inner)
    monkeypatch.setattr(intel, "_run_on_segment", fake_segment)

    df = _frame(400)
    # Exclude through the end of the frame → no new OOS bars.
    outcome = intel.run_nested(df, exclude_oos_before=df["time"].iloc[-1])
    assert outcome.best_params is None
    assert outcome.rolling_oos_window is None
    assert outcome.rolling_oos_validation is not None
    assert outcome.rolling_oos_validation.accepted is False


def test_validate_can_skip_oos_gate_for_rolling_oos():
    v = StrategyValidator(
        ValidatorConfig(
            max_drawdown=1.0,
            sharpe_min=-100.0,
            sharpe_max=100.0,
            p_value_max=1.0,
            oos_degradation_max=0.0,  # would reject any positive deg if applied
            is_fraction=0.7,
            min_trades=40,
            min_oos_trades=1,
            min_regime_trades=0,
            block_bootstrap_reps=0,
            multiple_testing="none",
            min_dsr=0.0,
            pbo_enabled=False,
            pvalue_method="hac",
        )
    )
    bt = Backtester(
        cost_model=CostModel(round_trip_floor=0.0),
        fill_model=FillModel(enabled=False),
        account=AccountConfig(enabled=False),
    )
    df = _frame(120)
    params = StrategyParams(long_window=10, short_window=3, max_hold_bars=20)
    val, _ = v.validate(df, params, backtester=bt, apply_oos_gate=False)
    assert val.oos_degradation == 0.0
    assert val.accepted is True
