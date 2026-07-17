"""Statistical inference for strategy validation (dependence + selection bias)."""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def returns_pvalue(returns: pd.Series) -> float:
    """IID one-sample t-test H0: E[r]=0. Prefer ``hac_mean_pvalue`` for bar returns."""
    r = returns.dropna()
    if len(r) < 3:
        return 1.0
    if np.allclose(r.to_numpy(), 0.0):
        return 1.0
    _t_stat, p = stats.ttest_1samp(r.to_numpy(), popmean=0.0)
    if np.isnan(p):
        return 1.0
    return float(p)


def newey_west_lags(n: int, lags: int = 0) -> int:
    """Bartlett / Newey–West lag length. ``lags<=0`` → automatic rule."""
    n = max(1, int(n))
    if lags and lags > 0:
        return min(int(lags), n - 1)
    # Newey–West automatic bandwidth (approx.)
    return int(max(1, min(n - 1, np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))))


def hac_mean_pvalue(returns: pd.Series, *, lags: int = 0) -> float:
    """Two-sided p-value for H0: E[r]=0 using Newey–West (HAC) standard errors.

    Accounts for serial correlation in bar PnL; does not assume IID observations.
    """
    r = returns.fillna(0.0).to_numpy(dtype=float)
    n = len(r)
    if n < 5:
        return 1.0
    if np.allclose(r, 0.0):
        return 1.0
    mu = float(r.mean())
    L = newey_west_lags(n, lags)
    # γ0 + 2 Σ w_k γ_k  (Bartlett kernel)
    demean = r - mu
    gamma0 = float(np.dot(demean, demean) / n)
    nw_var = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        gamma_k = float(np.dot(demean[k:], demean[:-k]) / n)
        nw_var += 2.0 * w * gamma_k
    nw_var = max(nw_var, 0.0)
    se = np.sqrt(nw_var / n)
    if se < 1e-18:
        return 1.0 if abs(mu) < 1e-18 else 0.0
    t_stat = mu / se
    # Use Student-t with n-1 df as a conservative finite-sample approx.
    p = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
    if np.isnan(p):
        return 1.0
    return min(1.0, max(0.0, p))


def block_bootstrap_mean_pvalue(
    returns: pd.Series,
    *,
    block_size: int = 48,
    n_boot: int = 400,
    seed: int = 42,
) -> float:
    """Two-sided block-bootstrap p-value for H0: E[r] = 0.

    Returns are recentered under H0; contiguous blocks are resampled to preserve
    serial dependence. Falls back to HAC when the series is too short.
    """
    r = returns.fillna(0.0).to_numpy(dtype=float)
    n = len(r)
    block = max(1, int(block_size))
    boots = max(0, int(n_boot))
    if boots <= 0 or n < max(20, block * 2):
        return hac_mean_pvalue(returns)
    obs = float(r.mean())
    if abs(obs) < 1e-18 or np.allclose(r, 0.0):
        return 1.0
    centered = r - obs
    rng = np.random.default_rng(int(seed))
    n_blocks = int(np.ceil(n / block))
    extremes = 0
    for _ in range(boots):
        starts = rng.integers(0, n, size=n_blocks)
        pieces = [centered[s : s + block] for s in starts]
        sample = np.concatenate(pieces)[:n]
        if abs(float(sample.mean())) >= abs(obs):
            extremes += 1
    return float((extremes + 1) / (boots + 1))


def _nonannual_sharpe(r: np.ndarray) -> float:
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std < 1e-18 or np.isnan(std):
        return 0.0
    return float(r.mean() / std)


def sharpe_ratio_variance(sr: float, n: int, skew: float, kurt: float) -> float:
    """Estimated variance of the non-annualized Sharpe ratio (Lo / BLP)."""
    n = max(2, int(n))
    return (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2)) / (n - 1)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum Sharpe under multiple testing (Bailey & López de Prado)."""
    n = max(1, int(n_trials))
    sigma = max(0.0, float(sr_std))
    if n <= 1 or sigma <= 0.0:
        return 0.0
    emc = 0.5772156649015329  # Euler–Mascheroni
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    if np.isnan(z1) or np.isnan(z2):
        return 0.0
    return float(sigma * ((1.0 - emc) * z1 + emc * z2))


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    n_trials: int = 1,
) -> tuple[float, float, float]:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Returns ``(dsr, sr_star, nonannual_sr)`` where ``dsr = Φ[(SR−SR*)/√V]``
    is the probability that the true SR exceeds the multiple-testing threshold SR*.
    """
    r = returns.fillna(0.0).to_numpy(dtype=float)
    n = len(r)
    if n < 5 or np.allclose(r, 0.0):
        return 0.0, 0.0, 0.0
    sr = _nonannual_sharpe(r)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # Pearson kurtosis
    if np.isnan(skew):
        skew = 0.0
    if np.isnan(kurt) or kurt < 1.0:
        kurt = 3.0
    var_sr = sharpe_ratio_variance(sr, n, skew, kurt)
    if var_sr <= 0.0 or np.isnan(var_sr):
        return 0.0, 0.0, sr
    sr_std = float(np.sqrt(var_sr))
    sr_star = expected_max_sharpe(n_trials, sr_std)
    z = (sr - sr_star) / sr_std
    dsr = float(stats.norm.cdf(z))
    if np.isnan(dsr):
        return 0.0, sr_star, sr
    return dsr, float(sr_star), sr


def _column_sharpes(mat: np.ndarray) -> np.ndarray:
    """Non-annualized Sharpe per column of a (T, N) return matrix."""
    t, n = mat.shape
    out = np.zeros(n, dtype=float)
    if t < 2:
        return out
    mu = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1)
    good = std > 1e-18
    out[good] = mu[good] / std[good]
    return out


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    *,
    n_slices: int = 8,
) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al.).

    ``returns_matrix`` is shape ``(T, N)`` with aligned bar returns for N trials.
    Returns PBO in [0, 1] (higher = more overfit). Insufficient data → 1.0.
    """
    if returns_matrix.ndim != 2:
        return 1.0
    t, n_strats = returns_matrix.shape
    if n_strats < 2 or t < 20:
        return 1.0

    s = int(n_slices)
    if s % 2:
        s -= 1
    s = max(2, min(s, t // 2))
    t_use = (t // s) * s
    if t_use < s * 2 or n_strats < 2:
        return 1.0
    mat = np.asarray(returns_matrix[:t_use], dtype=float)

    part = t_use // s
    parts = [mat[i * part : (i + 1) * part] for i in range(s)]
    half = s // 2
    fails = 0
    total = 0
    for is_idx in combinations(range(s), half):
        oos_idx = [i for i in range(s) if i not in set(is_idx)]
        is_r = np.concatenate([parts[i] for i in is_idx], axis=0)
        oos_r = np.concatenate([parts[i] for i in oos_idx], axis=0)
        is_perf = _column_sharpes(is_r)
        oos_perf = _column_sharpes(oos_r)
        best = int(np.argmax(is_perf))
        # Overfit event: IS-best underperforms the OOS median.
        if oos_perf[best] < float(np.median(oos_perf)):
            fails += 1
        total += 1
    if total <= 0:
        return 1.0
    return float(fails / total)


def adjust_alpha(alpha: float, n_tests: int, method: str = "bonferroni") -> float:
    """Family-wise / FDR-style alpha for a single-test gate."""
    a = max(0.0, float(alpha))
    m = max(1, int(n_tests))
    method = (method or "none").lower()
    if method in ("none", "off", ""):
        return a
    if method in ("bonferroni", "holm"):
        # Per-comparison threshold under Bonferroni (conservative; Holm needs all p's).
        return a / m
    if method in ("fdr", "fdr_bh", "bh"):
        # Conservative single-test stand-in when only one p is available.
        return a * (1.0 / m)
    return a


def regime_trade_counts(
    trades: list[dict],
    regimes: Optional[pd.Series],
) -> dict[str, int]:
    """Count closed trades by regime at entry bar."""
    counts: dict[str, int] = {}
    if regimes is None or regimes.empty:
        return counts
    reg_vals = regimes.to_numpy()
    for trade in trades:
        idx = trade.get("entry_i")
        if idx is None:
            continue
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        if i < 0 or i >= len(reg_vals):
            continue
        key = str(reg_vals[i])
        counts[key] = counts.get(key, 0) + 1
    return counts
