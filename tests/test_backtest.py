from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import Backtester
from src.risk import CostModel
from src.strategy import StrategyParams


def test_round_trip_costs_hit_equity_and_trade_pnl(monkeypatch):
    """Entry and exit each charge one-way; trade PnL uses full round-trip."""
    n = 12
    closes = np.concatenate([[100.0] * 6, [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]])
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC"),
            "close": closes,
        }
    )
    params = StrategyParams(long_window=3, short_window=2, max_hold_bars=100)
    cost = CostModel(min_cost_bps=10.0)  # 10 bps one-way floor → 0.001
    cost_one = cost.one_way_fraction()
    cost_rt = cost.round_trip_fraction()

    # Force a single long entry then flat so costs are deterministic.
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

    result = Backtester(cost_model=cost).run(frame, params=params)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade["entry_i"] == 5
    assert trade["exit_i"] == 8
    gross = (trade["exit"] / trade["entry"]) - 1.0
    assert trade["pnl"] == gross - cost_rt

    # Warmup zeroes bars 0..long_window-1; entry at 5 and exit at 8 both charge.
    assert result.bar_returns.iloc[5] == -cost_one + (closes[6] / closes[5] - 1.0)
    assert result.bar_returns.iloc[8] == -cost_one
