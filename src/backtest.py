"""Event-driven backtester with limit-fill execution assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import BARS_PER_YEAR_M30, PerformanceReport, build_report
from .risk import CostModel
from .strategy import RegressionStrategy, StrategyParams


@dataclass
class FillModel:
    """Live-aligned limit entry assumptions for backtests.

    Live places bid/ask limits and may wait across bars. Immediate bar-close
    fills inflate Sharpe/DD versus production; this model requires an OHLC
    touch, applies adverse slippage, and expires unfilled orders.
    """

    enabled: bool = True
    ttl_bars: int = 2
    slippage_bps: float = 1.0
    limit_offset_bps: float = 0.0

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "FillModel":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("limit_fills", True)),
            ttl_bars=max(1, int(cfg.get("limit_ttl_bars", 2))),
            slippage_bps=float(cfg.get("slippage_bps", 1.0)),
            limit_offset_bps=float(cfg.get("limit_offset_bps", 0.0)),
        )

    @property
    def slippage_fraction(self) -> float:
        return max(0.0, self.slippage_bps) / 10_000.0

    @property
    def limit_offset_fraction(self) -> float:
        return max(0.0, self.limit_offset_bps) / 10_000.0


@dataclass
class BacktestResult:
    report: PerformanceReport
    trades: list[dict]
    bar_returns: pd.Series
    signals: pd.Series
    params: StrategyParams
    unfilled_entries: int = 0
    fill_model: Optional[FillModel] = None


class Backtester:
    """
    Bar-close signal backtester with optional limit-entry simulation:

    - Signal on bar i close → place buy/sell limit (not a guaranteed fill).
    - Fill on a later bar only if OHLC trades through the limit (within TTL).
    - Fill price includes adverse slippage; expired limits are cancelled.
    - Exits (signal flip / max-hold / EOD) use market-at-close + slippage
      (matches live market flatten / reverse closes).
    - One-way costs hit equity on entry fill and on exit; trade PnL uses RT cost.
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        initial_equity: float = 1.0,
        periods_per_year: float = BARS_PER_YEAR_M30,
        fill_model: Optional[FillModel] = None,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.initial_equity = initial_equity
        self.periods_per_year = periods_per_year
        self.fill_model = fill_model or FillModel()

    def run(
        self,
        df: pd.DataFrame,
        params: Optional[StrategyParams] = None,
        strategy: Optional[RegressionStrategy] = None,
    ) -> BacktestResult:
        if not self.fill_model.enabled:
            return self._run_immediate(df, params=params, strategy=strategy)
        return self._run_limit(df, params=params, strategy=strategy)

    def _prepare(
        self,
        df: pd.DataFrame,
        params: Optional[StrategyParams],
        strategy: Optional[RegressionStrategy],
    ) -> tuple[StrategyParams, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = params or StrategyParams()
        strategy = strategy or RegressionStrategy(params)
        strategy.update_params(params)

        annotated = strategy.signal_series(df)
        closes = annotated["close"].to_numpy(dtype=float)
        highs, lows = self._high_low(annotated, closes)
        signals = annotated["signal"].to_numpy(dtype=int)
        n = len(annotated)
        fwd = np.zeros(n, dtype=float)
        fwd[:-1] = closes[1:] / closes[:-1] - 1.0
        return params, annotated, closes, highs, lows, signals, fwd

    @staticmethod
    def _high_low(annotated: pd.DataFrame, closes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if "high" in annotated.columns and "low" in annotated.columns:
            highs = annotated["high"].to_numpy(dtype=float)
            lows = annotated["low"].to_numpy(dtype=float)
            # Guard against bad feeds.
            highs = np.maximum(highs, closes)
            lows = np.minimum(lows, closes)
            return highs, lows
        return closes.copy(), closes.copy()

    def _limit_price(self, side: int, close: float) -> float:
        off = self.fill_model.limit_offset_fraction
        if side > 0:
            return float(close * (1.0 - off))
        return float(close * (1.0 + off))

    def _fill_price(self, side: int, limit: float) -> float:
        slip = self.fill_model.slippage_fraction
        if side > 0:
            return float(limit * (1.0 + slip))
        return float(limit * (1.0 - slip))

    @staticmethod
    def _limit_touched(side: int, limit: float, high: float, low: float) -> bool:
        if side > 0:
            return low <= limit
        return high >= limit

    def _market_exit_price(self, side: int, close: float) -> float:
        slip = self.fill_model.slippage_fraction
        if side > 0:
            return float(close * (1.0 - slip))
        return float(close * (1.0 + slip))

    def _run_limit(
        self,
        df: pd.DataFrame,
        params: Optional[StrategyParams] = None,
        strategy: Optional[RegressionStrategy] = None,
    ) -> BacktestResult:
        params, annotated, closes, highs, lows, signals, fwd = self._prepare(
            df, params, strategy
        )
        n = len(annotated)
        max_hold = params.max_hold_bars
        cost_rt = self.cost_model.round_trip_fraction()
        cost_one = self.cost_model.one_way_fraction()
        ttl = self.fill_model.ttl_bars

        bar_rets = np.zeros(n, dtype=float)
        trades: list[dict] = []
        trade_pnls: list[float] = []
        unfilled = 0

        position = 0
        hold = 0
        entry_price = 0.0
        entry_i = -1

        pending_side = 0
        pending_limit = 0.0
        pending_age = 0

        for i in range(n - 1):
            # --- Resolve working limit against this bar's range ---
            if pending_side != 0:
                pending_age += 1
                if self._limit_touched(pending_side, pending_limit, highs[i], lows[i]):
                    position = pending_side
                    entry_price = self._fill_price(pending_side, pending_limit)
                    entry_i = i
                    hold = 0
                    bar_rets[i] -= cost_one
                    trades.append(
                        {
                            "entry_i": entry_i,
                            "exit_i": None,
                            "side": position,
                            "entry": entry_price,
                            "exit": None,
                            "pnl": None,
                            "limit": pending_limit,
                            "fill": "limit",
                        }
                    )
                    pending_side = 0
                    pending_limit = 0.0
                    pending_age = 0
                elif pending_age >= ttl:
                    unfilled += 1
                    pending_side = 0
                    pending_limit = 0.0
                    pending_age = 0

            desired = int(signals[i])
            if position != 0:
                hold += 1
                if hold >= max_hold:
                    desired = 0

            # --- Exit open position (market at close + slippage) ---
            if position != 0 and desired != position:
                exit_price = self._market_exit_price(position, closes[i])
                bar_rets[i] -= cost_one
                pnl = position * (exit_price / entry_price - 1.0) - cost_rt
                trade_pnls.append(pnl)
                if trades and trades[-1].get("exit_i") is None and trades[-1].get("side") == position:
                    trades[-1].update({"exit_i": i, "exit": exit_price, "pnl": pnl})
                else:
                    trades.append(
                        {
                            "entry_i": entry_i,
                            "exit_i": i,
                            "side": position,
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl": pnl,
                            "fill": "market_exit",
                        }
                    )
                position = 0
                entry_i = -1
                hold = 0
                entry_price = 0.0
                # Signal reverse also cancels a stale opposite pending.
                if pending_side != 0 and pending_side != desired:
                    pending_side = 0
                    pending_limit = 0.0
                    pending_age = 0

            # Flat desired while pending same-side → keep waiting.
            # Desired flat → cancel pending.
            if desired == 0 and pending_side != 0:
                pending_side = 0
                pending_limit = 0.0
                pending_age = 0

            # --- Place new limit when flat and no working order ---
            if position == 0 and pending_side == 0 and desired != 0:
                pending_side = desired
                pending_limit = self._limit_price(desired, closes[i])
                pending_age = 0

            if position != 0:
                bar_rets[i] += position * fwd[i]

        # End-of-sample: cancel pending; force market flat.
        if pending_side != 0:
            unfilled += 1
        if position != 0 and entry_i >= 0:
            exit_price = self._market_exit_price(position, closes[-1])
            bar_rets[-1] -= cost_one
            pnl = position * (exit_price / entry_price - 1.0) - cost_rt
            trade_pnls.append(pnl)
            if trades and trades[-1].get("exit_i") is None:
                trades[-1].update({"exit_i": n - 1, "exit": exit_price, "pnl": pnl})
            else:
                trades.append(
                    {
                        "entry_i": entry_i,
                        "exit_i": n - 1,
                        "side": position,
                        "entry": entry_price,
                        "exit": exit_price,
                        "pnl": pnl,
                        "fill": "market_exit",
                    }
                )

        # Drop open placeholders without exits from trade list for metrics.
        closed_trades = [t for t in trades if t.get("exit_i") is not None and t.get("pnl") is not None]

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
        return BacktestResult(
            report=report,
            trades=closed_trades,
            bar_returns=bar_series,
            signals=sig_series,
            params=params,
            unfilled_entries=unfilled,
            fill_model=self.fill_model,
        )

    def _run_immediate(
        self,
        df: pd.DataFrame,
        params: Optional[StrategyParams] = None,
        strategy: Optional[RegressionStrategy] = None,
    ) -> BacktestResult:
        """Legacy guaranteed fill-at-signal-close path (optimistic vs live)."""
        params, annotated, closes, _highs, _lows, signals, fwd = self._prepare(
            df, params, strategy
        )
        n = len(annotated)
        max_hold = params.max_hold_bars
        cost_rt = self.cost_model.round_trip_fraction()
        cost_one = self.cost_model.one_way_fraction()

        bar_rets = np.zeros(n, dtype=float)
        trades: list[dict] = []
        trade_pnls: list[float] = []
        position = 0
        hold = 0
        entry_price = 0.0
        entry_i = -1

        for i in range(n - 1):
            desired = int(signals[i])
            if position != 0:
                hold += 1
                if hold >= max_hold:
                    desired = 0

            if desired != position:
                if position != 0 and entry_i >= 0:
                    exit_price = closes[i]
                    bar_rets[i] -= cost_one
                    pnl = position * (exit_price / entry_price - 1.0) - cost_rt
                    trade_pnls.append(pnl)
                    trades.append(
                        {
                            "entry_i": entry_i,
                            "exit_i": i,
                            "side": position,
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl": pnl,
                            "fill": "immediate",
                        }
                    )
                    entry_i = -1
                    hold = 0

                if desired != 0:
                    position = desired
                    entry_price = closes[i]
                    entry_i = i
                    hold = 0
                    bar_rets[i] -= cost_one
                else:
                    position = 0

            bar_rets[i] += position * fwd[i]

        if position != 0 and entry_i >= 0:
            exit_price = closes[-1]
            bar_rets[-1] -= cost_one
            pnl = position * (exit_price / entry_price - 1.0) - cost_rt
            trade_pnls.append(pnl)
            trades.append(
                {
                    "entry_i": entry_i,
                    "exit_i": n - 1,
                    "side": position,
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl": pnl,
                    "fill": "immediate",
                }
            )

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
        return BacktestResult(
            report=report,
            trades=trades,
            bar_returns=bar_series,
            signals=sig_series,
            params=params,
            unfilled_entries=0,
            fill_model=self.fill_model,
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

        warmup = params.long_window
        if len(is_df) >= warmup and len(oos_df) > 0:
            oos_with_warm = pd.concat([is_df.iloc[-warmup:], oos_df], ignore_index=True)
            oos_full = self.run(oos_with_warm, params=params)
            oos_rets = oos_full.bar_returns.iloc[warmup:].reset_index(drop=True)
            oos_sigs = oos_full.signals.iloc[warmup:].reset_index(drop=True)
            oos_trades = [t for t in oos_full.trades if t["entry_i"] >= warmup]
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
                unfilled_entries=oos_full.unfilled_entries,
                fill_model=self.fill_model,
            )
        else:
            oos_result = self.run(oos_df, params=params)

        is_result = self.run(is_df, params=params)
        deg = oos_degradation(is_result.report.sharpe, oos_result.report.sharpe)
        return is_result, oos_result, deg
