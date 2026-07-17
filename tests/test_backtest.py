from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import AccountConfig, Backtester, FillModel, SymbolSpec
from src.risk import CostModel, RiskManager
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
        account=AccountConfig(enabled=False),
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
        account=AccountConfig(enabled=False),
    ).run(frame, params=params)

    assert len(result.trades) == 1
    assert result.trades[0]["entry_i"] == 6
    assert result.trades[0]["fill"] == "limit"
    assert result.unfilled_entries == 0


def test_limit_expires_unfilled(monkeypatch):
    n = 12
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
        account=AccountConfig(enabled=False),
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
        account=AccountConfig(enabled=False),
    ).run(frame, params=params)

    assert len(result.trades) == 1
    assert abs(result.trades[0]["entry"] - 100.1) < 1e-9


def _force_long_signal(monkeypatch, n, start, end):
    signals = np.zeros(n, dtype=int)
    signals[start:end] = 1

    def fake_signal_series(self, df):
        out = df.copy()
        out["signal"] = signals
        return out

    monkeypatch.setattr(
        "src.strategy.RegressionStrategy.signal_series",
        fake_signal_series,
    )


def test_account_sizing_uses_kelly_lots_not_unit(monkeypatch):
    n = 16
    closes = np.full(n, 10_000.0)
    lows = closes.copy()
    highs = closes.copy()
    lows[6] = 9_990.0
    closes[7:] = 9_950.0
    lows[7:] = 9_940.0
    highs[7:] = 9_960.0
    closes[10:] = 10_000.0
    frame = _frame(closes, highs=highs, lows=lows)
    params = StrategyParams(long_window=3, short_window=2, max_hold_bars=100)
    _force_long_signal(monkeypatch, n, 5, 10)

    risk = RiskManager(
        half_kelly=False,
        default_win_rate=1.0,
        default_reward_risk=1.0,
        max_fraction=0.10,
        stop_pct=0.01,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=1.0,
        max_lots=5.0,
        min_lots=0.01,
    )
    spec = SymbolSpec(
        tick_size=1.0,
        tick_value=1.0,
        contract_size=1.0,
        point=1.0,
        digits=1,
        margin_per_lot=200.0,
        volume_step=0.01,
        volume_min=0.01,
        volume_max=5.0,
    )
    result = Backtester(
        cost_model=CostModel(min_cost_bps=0.0),
        fill_model=FillModel(enabled=True, ttl_bars=3, slippage_bps=0.0),
        account=AccountConfig(enabled=True, initial_equity=10_000.0, stop_out_level=0.01),
        symbol_spec=spec,
        risk=risk,
    ).run(frame, params=params)

    assert len(result.trades) >= 1
    assert result.trades[0]["lots"] == 5.0
    assert result.trades[0]["lots"] != 1.0
    assert result.final_equity is not None
    assert result.report.equity_curve.iloc[0] == 10_000.0


def test_account_dd_differs_from_unit_notional_dd(monkeypatch):
    n = 20
    closes = np.concatenate(
        [
            np.full(8, 100.0),
            np.array([100.0, 90.0, 80.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0]),
        ]
    )
    highs = closes.copy()
    lows = closes.copy()
    lows[9] = 89.0
    frame = _frame(closes, highs=highs, lows=lows)
    params = StrategyParams(long_window=3, short_window=2, max_hold_bars=100)
    _force_long_signal(monkeypatch, n, 8, 15)

    unit = Backtester(
        cost_model=CostModel(min_cost_bps=0.0),
        fill_model=FillModel(enabled=True, ttl_bars=5, slippage_bps=0.0),
        account=AccountConfig(enabled=False),
    ).run(frame, params=params)

    risk = RiskManager(
        half_kelly=False,
        default_win_rate=1.0,
        default_reward_risk=1.0,
        max_fraction=0.25,
        stop_pct=0.50,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=1.0,
        max_lots=1.0,
        min_lots=0.01,
    )
    acct = Backtester(
        cost_model=CostModel(min_cost_bps=0.0),
        fill_model=FillModel(enabled=True, ttl_bars=5, slippage_bps=0.0),
        account=AccountConfig(enabled=True, initial_equity=10_000.0, stop_out_level=0.01),
        symbol_spec=SymbolSpec(
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            point=1.0,
            margin_per_lot=50.0,
            volume_max=1.0,
        ),
        risk=risk,
    ).run(frame, params=params)

    assert unit.report.max_drawdown > 0
    assert acct.report.max_drawdown > 0
    assert abs(unit.report.max_drawdown - acct.report.max_drawdown) > 1e-6


def test_margin_stop_out_forces_liquidation(monkeypatch):
    n = 14
    closes = np.full(n, 100.0)
    highs = closes.copy()
    lows = closes.copy()
    lows[6] = 99.0
    closes[7:] = 10.0
    highs[7:] = 10.0
    lows[7:] = 10.0
    frame = _frame(closes, highs=highs, lows=lows)
    params = StrategyParams(long_window=3, short_window=2, max_hold_bars=100)
    _force_long_signal(monkeypatch, n, 5, 12)

    risk = RiskManager(
        half_kelly=False,
        default_win_rate=1.0,
        default_reward_risk=1.0,
        max_fraction=1.0,
        stop_pct=0.90,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=1.0,
        max_lots=50.0,
        max_margin_fraction=1.0,
    )
    result = Backtester(
        cost_model=CostModel(min_cost_bps=0.0),
        fill_model=FillModel(enabled=True, ttl_bars=3, slippage_bps=0.0),
        account=AccountConfig(enabled=True, initial_equity=1_000.0, stop_out_level=0.50),
        symbol_spec=SymbolSpec(
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            point=1.0,
            margin_per_lot=20.0,
            volume_max=50.0,
        ),
        risk=risk,
    ).run(frame, params=params)

    assert result.liquidations >= 1
    assert any(t.get("exit_fill") == "stop_out" or t.get("fill") == "stop_out" for t in result.trades)
