"""Dual-window OLS linear regression strategy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd


class Signal(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class Regime:
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    UNKNOWN = "unknown"


@dataclass
class StrategyParams:
    long_window: int = 240
    short_window: int = 48
    max_hold_bars: int = 16

    def as_dict(self) -> dict:
        return {
            "long_window": int(self.long_window),
            "short_window": int(self.short_window),
            "max_hold_bars": int(self.max_hold_bars),
        }


@dataclass
class StrategyDecision:
    signal: Signal
    regime: str
    b_long: float
    b_short: float
    bar_time: Optional[pd.Timestamp] = None


class RegressionStrategy:
    """
    Compute OLS slopes on close prices for long/short windows.

    - Same sign (b_long * b_short > 0): trend regime -> follow sign(b_long)
    - Opposite sign: mean-reversion -> fade sign(b_short)
    Price distance to the regression line is ignored.
    """

    def __init__(self, params: Optional[StrategyParams] = None) -> None:
        self.params = params or StrategyParams()

    def update_params(self, params: StrategyParams) -> None:
        self.params = params

    @staticmethod
    def ols_slope(closes: np.ndarray) -> float:
        n = len(closes)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.asarray(closes, dtype=float)
        # numpy.polyfit degree-1 -> [slope, intercept]
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    def compute_slopes(self, closes: np.ndarray) -> tuple[float, float]:
        lw = self.params.long_window
        sw = self.params.short_window
        if len(closes) < lw:
            raise ValueError(f"Need at least {lw} closes, got {len(closes)}")
        b_long = self.ols_slope(closes[-lw:])
        b_short = self.ols_slope(closes[-sw:])
        return b_long, b_short

    def decide_from_closes(
        self,
        closes: np.ndarray,
        bar_time: Optional[pd.Timestamp] = None,
    ) -> StrategyDecision:
        b_long, b_short = self.compute_slopes(closes)
        product = b_long * b_short

        if product > 0:
            regime = Regime.TREND
            direction = 1 if b_long > 0 else -1
            signal = Signal.LONG if direction > 0 else Signal.SHORT
        elif product < 0:
            regime = Regime.MEAN_REVERSION
            # Fade short-term move
            direction = -1 if b_short > 0 else 1
            signal = Signal.LONG if direction > 0 else Signal.SHORT
        else:
            regime = Regime.UNKNOWN
            signal = Signal.FLAT

        return StrategyDecision(
            signal=signal,
            regime=regime,
            b_long=b_long,
            b_short=b_short,
            bar_time=bar_time,
        )

    def decide(self, df: pd.DataFrame) -> StrategyDecision:
        closes = df["close"].to_numpy(dtype=float)
        bar_time = df["time"].iloc[-1] if "time" in df.columns else None
        return self.decide_from_closes(closes, bar_time=bar_time)

    def signal_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized walk-forward signals for backtesting (bar-close decision)."""
        closes = df["close"].to_numpy(dtype=float)
        n = len(closes)
        lw = self.params.long_window
        sw = self.params.short_window

        signals = np.zeros(n, dtype=int)
        regimes = np.array([Regime.UNKNOWN] * n, dtype=object)
        b_longs = np.full(n, np.nan)
        b_shorts = np.full(n, np.nan)

        for i in range(lw - 1, n):
            window_long = closes[i - lw + 1 : i + 1]
            window_short = closes[i - sw + 1 : i + 1]
            bl = self.ols_slope(window_long)
            bs = self.ols_slope(window_short)
            b_longs[i] = bl
            b_shorts[i] = bs
            product = bl * bs
            if product > 0:
                regimes[i] = Regime.TREND
                signals[i] = 1 if bl > 0 else -1
            elif product < 0:
                regimes[i] = Regime.MEAN_REVERSION
                signals[i] = -1 if bs > 0 else 1
            else:
                regimes[i] = Regime.UNKNOWN
                signals[i] = 0

        out = df.copy()
        out["signal"] = signals
        out["regime"] = regimes
        out["b_long"] = b_longs
        out["b_short"] = b_shorts
        return out
