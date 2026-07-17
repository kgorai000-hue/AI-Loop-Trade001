"""Event-driven backtester with transaction costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import BARS_PER_YEAR_M30, PerformanceReport, build_report
from .risk import CostModel
from .strategy import RegressionStrategy, StrategyParams


@dataclass
class BacktestResult:
    report: PerformanceReport
    trades: list[dict]
    bar_returns: pd.Series
    signals: pd.Series
    params: StrategyParams


class Backtester:
    """
    Simple bar-close backtester:
    - Signal decided on bar i close, position for bar i+1 return.
    - Flat when signal is 0 or after max_hold_bars.
    - Round-trip cost applied when position changes.
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        initial_equity: float = 1.0,
        periods_per_year: float = BARS_PER_YEAR_M30,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.initial_equity = initial_equity
        self.periods_per_year = periods_per_year

    def run(
        self,
        df: pd.DataFrame,
        params: Optional[StrategyParams] = None,
        strategy: Optional[RegressionStrategy] = None,
    ) -> BacktestResult:
        params = params or StrategyParams()
        strategy = strategy or RegressionStrategy(params)
        strategy.update_params(params)

        annotated = strategy.signal_series(df)
        closes = annotated["close"].to_numpy(dtype=float)
        signals = annotated["signal"].to_numpy(dtype=int)
        n = len(annotated)

        # Forward returns
        fwd = np.zeros(n, dtype=float)
        fwd[:-1] = closes[1:] / closes[:-1] - 1.0

        position = 0
        hold = 0
        max_hold = params.max_hold_bars
        cost_rt = self.cost_model.round_trip_fraction()
        cost_one = self.cost_model.one_way_fraction()

        bar_rets = np.zeros(n, dtype=float)
        trades: list[dict] = []
        entry_price = 0.0
        entry_i = -1
        trade_pnls: list[float] = []

        for i in range(n - 1):
            desired = int(signals[i])
            # Max hold forces flat, then re-evaluate next bar
            if position != 0:
                hold += 1
                if hold >= max_hold:
                    desired = 0

            if desired != position:
                # Close existing
                if position != 0 and entry_i >= 0:
                    exit_price = closes[i]
                    pnl = position * (exit_price / entry_price - 1.0) - cost_one
                    # Approximate: entry already paid one-way; exit pays one-way
                    trade_pnls.append(pnl)
                    trades.append(
                        {
                            "entry_i": entry_i,
                            "exit_i": i,
                            "side": position,
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl": pnl,
                        }
                    )
                    entry_i = -1
                    hold = 0

                # Open new
                if desired != 0:
                    position = desired
                    entry_price = closes[i]
                    entry_i = i
                    hold = 0
                    # Entry cost taken against next bar return stream
                    bar_rets[i] -= cost_one
                else:
                    position = 0

            # Mark-to-market for holding bar i → i+1
            bar_rets[i] += position * fwd[i]

        # Force close at end
        if position != 0 and entry_i >= 0:
            exit_price = closes[-1]
            pnl = position * (exit_price / entry_price - 1.0) - cost_one
            trade_pnls.append(pnl)
            trades.append(
                {
                    "entry_i": entry_i,
                    "exit_i": n - 1,
                    "side": position,
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl": pnl,
                }
            )

        # Warmup mask: no returns before first valid signal bar
        warmup = params.long_window
        bar_series = pd.Series(bar_rets, index=annotated.index)
        bar_series.iloc[:warmup] = 0.0
        sig_series = pd.Series(signals, index=annotated.index)

        report = build_report(
            bar_series,
            signals=sig_series,
            trade_pnls=trade_pnls,
            periods_per_year=self.periods_per_year,
        )
        # Attach cost awareness note via total_return already cost-adjusted
        _ = cost_rt  # retained for clarity / future logging

        return BacktestResult(
            report=report,
            trades=trades,
            bar_returns=bar_series,
            signals=sig_series,
            params=params,
        )

    def run_is_oos(
        self,
        df: pd.DataFrame,
        params: StrategyParams,
        is_fraction: float = 0.70,
    ) -> tuple[BacktestResult, BacktestResult, float]:
        """Split chronologically; return IS result, OOS result, Sharpe degradation."""
        from .metrics import oos_degradation

        n = len(df)
        split = max(params.long_window + 10, int(n * is_fraction))
        split = min(split, n - max(20, params.short_window))
        is_df = df.iloc[:split].reset_index(drop=True)
        oos_df = df.iloc[split:].reset_index(drop=True)

        # OOS needs warmup history — prepend IS tail for signal computation only
        warmup = params.long_window
        if len(is_df) >= warmup and len(oos_df) > 0:
            oos_with_warm = pd.concat([is_df.iloc[-warmup:], oos_df], ignore_index=True)
            oos_full = self.run(oos_with_warm, params=params)
            # Strip warmup returns from report by zeroing first warmup bars already done;
            # rebuild report on OOS-only segment
            oos_rets = oos_full.bar_returns.iloc[warmup:].reset_index(drop=True)
            oos_sigs = oos_full.signals.iloc[warmup:].reset_index(drop=True)
            oos_trades = [t for t in oos_full.trades if t["entry_i"] >= warmup]
            # Re-index trade indices relative (optional); pnls intact
            from .metrics import build_report

            oos_report = build_report(
                oos_rets,
                signals=oos_sigs,
                trade_pnls=[t["pnl"] for t in oos_trades],
                periods_per_year=self.periods_per_year,
            )
            oos_result = BacktestResult(
                report=oos_report,
                trades=oos_trades,
                bar_returns=oos_rets,
                signals=oos_sigs,
                params=params,
            )
        else:
            oos_result = self.run(oos_df, params=params)

        is_result = self.run(is_df, params=params)
        deg = oos_degradation(is_result.report.sharpe, oos_result.report.sharpe)
        return is_result, oos_result, deg
