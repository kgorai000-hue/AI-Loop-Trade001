from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import Backtester, FillModel
from src.risk import CostModel
from src.strategy import StrategyParams


def _frame(closes, highs=None, lows=None):
    n = len(closes)
    data = {
        "time": pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC"),
        "close": closes,
        "high": highs if highs is not None else closes,
        "low": lows if lows is not None else closes,
    }
    return pd.DataFrame(data)


def test_round_trip_costs_with_immediate_fills(monkeypatch):
    """Legacy path: entry and exit each charge one-way; trade PnL uses RT cost."""
    n = 12
    closes = np.concatenate([[100.0] * 6, [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]])
    frame = _frame(closes)
    params = StrategyParams(long_window=3, short_window=2, max_hold_bars=100)
    cost = CostModel(min_cost_bps=10.0)
    cost_one = cost.one_way_fraction()
    cost_rt = cost.round_trip_fraction()

    signals = np.zeros(n, dtype=int)
    signals[5:8] = 1

    def fake_signal_series(self, df):
        out = df.copy()
        out["signal"] = signals
        return out

    monkeypatch.setattr(
        "src.strategy.RegressionStrategy.signal_series",
        fake_signal_series,
    )

    result = Backtester(
        cost_model=cost,
        fill_model=FillModel(enabled=False),
    ).run(frame, params=params)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade["entry_i"] == 5
    assert trade["exit_i"] == 8
    gross = (trade["exit"] / trade["entry"]) - 1.0
    assert trade["pnl"] == gross - cost_rt
    assert result.bar_returns.iloc[5] == -cost_one + (closes[6] / closes[5] - 1.0)
    assert result.bar_returns.iloc[8] == -cost_one


def test_limit_fill_requires_ohlc_touch(monkeypatch):
    n = 12
    closes = np.array([100.0] * n, dtype=float)
    # Bar 6 dips through the buy limit placed at close[5]=100
    lows = closes.copy()
    highs = closes.copy()
    lows[6] = 99.5
    frame = _frame(closes, highs=highs, lows=lows)
    params = StrategyParams(long_window=3, short_window=2, max_hold_bars=100)

    signals = np.zeros(n, dtype=int)
    signals[5:9] = 1

    def fake_signal_series(self, df):
        out = df.copy()
        out["signal"] = signals
        return out

    monkeypatch.setattr(
        "src.strategy.RegressionStrategy.signal_series",
        fake_signal_series,
    )

    result = Backtester(
        cost_model=CostModel(min_cost_bps=0.0),
        fill_model=FillModel(enabled=True, ttl_bars=2, slippage_bps=0.0),
    ).run(frame, params=params)

    assert len(result.trades) == 1
    assert result.trades[0]["entry_i"] == 6
    assert result.trades[0]["fill"] == "limit"
    assert result.unfilled_entries == 0


def test_limit_expires_unfilled(monkeypatch):
    n = 12
    # Signal close=100 → buy limit at 100. Later bars stay above the limit
    # with valid OHLC (low <= close <= high) so the order never touches.
    closes = np.full(n, 100.0)
    closes[6:] = 101.0
    lows = np.full(n, 100.0)
    lows[6:] = 100.5
    highs = np.full(n, 101.0)
    highs[6:] = 102.0
    frame = _frame(closes, highs=highs, lows=lows)
    params = StrategyParams(long_window=3, short_window=2, max_hold_bars=100)

    signals = np.zeros(n, dtype=int)
    signals[5] = 1

    def fake_signal_series(self, df):
        out = df.copy()
        out["signal"] = signals
        return out

    monkeypatch.setattr(
        "src.strategy.RegressionStrategy.signal_series",
        fake_signal_series,
    )

    result = Backtester(
        cost_model=CostModel(min_cost_bps=0.0),
        fill_model=FillModel(enabled=True, ttl_bars=1, slippage_bps=0.0),
    ).run(frame, params=params)

    assert result.trades == []
    assert result.unfilled_entries >= 1


def test_limit_fill_applies_adverse_slippage(monkeypatch):
    n = 12
    closes = np.full(n, 100.0)
    lows = closes.copy()
    highs = closes.copy()
    lows[6] = 99.0
    frame = _frame(closes, highs=highs, lows=lows)
    params = StrategyParams(long_window=3, short_window=2, max_hold_bars=100)
    signals = np.zeros(n, dtype=int)
    signals[5:8] = 1

    def fake_signal_series(self, df):
        out = df.copy()
        out["signal"] = signals
        return out

    monkeypatch.setattr(
        "src.strategy.RegressionStrategy.signal_series",
        fake_signal_series,
    )

    result = Backtester(
        cost_model=CostModel(min_cost_bps=0.0),
        fill_model=FillModel(enabled=True, ttl_bars=2, slippage_bps=10.0),
    ).run(frame, params=params)

    assert len(result.trades) == 1
    # Buy limit at 100, +10 bps adverse → 100.1
    assert abs(result.trades[0]["entry"] - 100.1) < 1e-9
