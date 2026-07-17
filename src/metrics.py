"""Performance and statistical metrics for strategy validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


BARS_PER_YEAR_M30 = 48 * 252  # ~30m bars in a trading year


@dataclass
class PerformanceReport:
    total_return: float
    sharpe: float
    max_drawdown: float
    p_value: float
    ic: float
    n_trades: int
    win_rate: float
    reward_risk: float
    equity_curve: pd.Series

    def as_dict(self) -> dict:
        return {
            "total_return": float(self.total_return),
            "sharpe": float(self.sharpe),
            "max_drawdown": float(self.max_drawdown),
            "p_value": float(self.p_value),
            "ic": float(self.ic),
            "n_trades": int(self.n_trades),
            "win_rate": float(self.win_rate),
            "reward_risk": float(self.reward_risk),
        }


def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    r = returns.fillna(0.0)
    return initial * (1.0 + r).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(abs(dd.min())) if len(dd) else 0.0


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: float = BARS_PER_YEAR_M30,
    risk_free: float = 0.0,
) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def returns_pvalue(returns: pd.Series) -> float:
    """Two-sided t-test that mean return is zero. Lower p → more significant edge."""
    r = returns.dropna()
    if len(r) < 3:
        return 1.0
    # Skip all-zero series
    if np.allclose(r.to_numpy(), 0.0):
        return 1.0
    t_stat, p = stats.ttest_1samp(r.to_numpy(), popmean=0.0)
    if np.isnan(p):
        return 1.0
    return float(p)


def information_coefficient(signals: pd.Series, forward_returns: pd.Series) -> float:
    """Spearman rank IC between signal and next-bar (or forward) return."""
    aligned = pd.concat([signals, forward_returns], axis=1).dropna()
    if len(aligned) < 5:
        return 0.0
    a = aligned.iloc[:, 0].to_numpy()
    b = aligned.iloc[:, 1].to_numpy()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    corr, _ = stats.spearmanr(a, b)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def oos_degradation(
    is_metric: float,
    oos_metric: float,
    floor: float = 1e-9,
) -> float:
    """
    Relative degradation of OOS vs IS performance.
    Positive when OOS is worse. Uses absolute values for signed metrics like Sharpe.
    """
    base = abs(is_metric)
    if base < floor:
        # If IS is near zero, any positive OOS is fine; negative OOS is full degradation
        if oos_metric >= 0:
            return 0.0
        return 1.0
    deg = (is_metric - oos_metric) / base
    return float(max(0.0, deg))


def trade_stats(trade_pnls: list[float]) -> tuple[float, float, int]:
    if not trade_pnls:
        return 0.0, 1.0, 0
    arr = np.asarray(trade_pnls, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = float(len(wins) / len(arr)) if len(arr) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    rr = avg_win / avg_loss if avg_loss > 0 else (2.0 if avg_win > 0 else 1.0)
    return win_rate, rr, len(arr)


def build_report(
    bar_returns: pd.Series,
    signals: Optional[pd.Series] = None,
    trade_pnls: Optional[list[float]] = None,
    periods_per_year: float = BARS_PER_YEAR_M30,
) -> PerformanceReport:
    equity = equity_from_returns(bar_returns)
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    sharpe = sharpe_ratio(bar_returns, periods_per_year=periods_per_year)
    dd = max_drawdown(equity)
    p = returns_pvalue(bar_returns)

    ic = 0.0
    if signals is not None:
        fwd = bar_returns.shift(-1)
        ic = information_coefficient(signals, fwd)

    wr, rr, n = trade_stats(trade_pnls or [])
    return PerformanceReport(
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=dd,
        p_value=p,
        ic=ic,
        n_trades=n,
        win_rate=wr,
        reward_risk=rr,
        equity_curve=equity,
    )
