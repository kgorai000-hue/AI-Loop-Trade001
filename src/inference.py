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
    """Bartlett / Newey-West lag length. ``lags<=0`` -> automatic rule."""
    n = max(1, int(n))
    if lags and lags > 0:
        return min(int(lags), n - 1)
    # Newey-West automatic bandwidth (approx.)
    return int(max(1, min(n - 1, np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))))


def hac_mean_pvalue(returns: pd.Series, *, lags: int = 0) -> float:
    """Two-sided p-value for H0: E[r]=0 using Newey-West (HAC) standard errors.

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
    # gamma0 + 2 Sum w_k gamma_k  (Bartlett kernel)
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


def circular_block_resample(
    data: np.ndarray,
    *,
    block: int,
    starts: np.ndarray,
) -> np.ndarray:
    """Concatenate fixed-length circular blocks and truncate to ``len(data)``.

    Each start index wraps with modulo ``n`` (Politis & Romano circular block
    bootstrap), so every block has exactly ``block`` observations -- unlike a
    naive end-truncated slice that shortens blocks near the series tail.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    block = max(1, int(block))
    if n == 0:
        return x.copy()
    starts = np.asarray(starts, dtype=int).reshape(-1)
    # shape (n_blocks, block) -> fixed-length indices with wrap-around
    idx = (starts[:, None] + np.arange(block, dtype=int)[None, :]) % n
    return x[idx.ravel()[:n]]


def moving_block_resample(
    data: np.ndarray,
    *,
    block: int,
    starts: np.ndarray,
) -> np.ndarray:
    """Concatenate fixed-length moving blocks (no wrap); truncate to ``len(data)``.

    ``starts`` must lie in ``[0, n - block]`` so each block is exactly ``block``
    long (Kunsch moving-block bootstrap).
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    block = max(1, int(block))
    if n == 0:
        return x.copy()
    if block > n:
        raise ValueError(f"block ({block}) exceeds series length ({n})")
    starts = np.asarray(starts, dtype=int).reshape(-1)
    if np.any(starts < 0) or np.any(starts > n - block):
        raise ValueError(
            f"moving-block starts must be in [0, {n - block}], got "
            f"[{int(starts.min())}, {int(starts.max())}]"
        )
    idx = starts[:, None] + np.arange(block, dtype=int)[None, :]
    return x[idx.ravel()[:n]]


def block_bootstrap_mean_pvalue(
    returns: pd.Series,
    *,
    block_size: int = 48,
    n_boot: int = 400,
    seed: int = 42,
    scheme: str = "circular",
) -> float:
    """Two-sided block-bootstrap p-value for H0: E[r] = 0.

    Returns are recentered under H0; contiguous **fixed-length** blocks are
    resampled to preserve serial dependence.

    ``scheme``:
    - ``circular`` (default): wrap-around blocks; starts uniform on ``{0..n-1}``
    - ``moving``: non-wrapping blocks; starts uniform on ``{0..n-block}``

    Falls back to HAC when the series is too short.
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
    scheme_l = (scheme or "circular").strip().lower()
    if scheme_l not in ("circular", "moving"):
        raise ValueError(f"unknown block bootstrap scheme: {scheme!r}")
    extremes = 0
    for _ in range(boots):
        if scheme_l == "moving":
            starts = rng.integers(0, n - block + 1, size=n_blocks)
            sample = moving_block_resample(centered, block=block, starts=starts)
        else:
            starts = rng.integers(0, n, size=n_blocks)
            sample = circular_block_resample(centered, block=block, starts=starts)
        if abs(float(sample.mean())) >= abs(obs):
            extremes += 1
    return float((extremes + 1) / (boots + 1))


def nonannual_sharpe(r: np.ndarray) -> float:
    """Non-annualized Sharpe (mean / sample std) for a return vector."""
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std < 1e-18 or np.isnan(std):
        return 0.0
    return float(r.mean() / std)


# Backward-compatible alias
_nonannual_sharpe = nonannual_sharpe


def sharpe_ratio_variance(sr: float, n: int, skew: float, kurt: float) -> float:
    """Estimated variance of the non-annualized Sharpe ratio (Lo / BLP)."""
    n = max(2, int(n))
    return (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr**2)) / (n - 1)


def trial_sharpe_dispersion(
    trial_sharpes: list[float] | np.ndarray,
    *,
    ddof: int = 1,
) -> tuple[float, float]:
    """Cross-sectional mean and std of non-annualized Sharpes across trials.

    This is the selection-bias scale ``sigma_{{SR_n}}`` in Bailey & Lopez de Prado --
    not the estimation SE of a single strategy's Sharpe.
    """
    arr = np.asarray(list(trial_sharpes), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, 0.0
    std = float(arr.std(ddof=max(0, int(ddof))))
    if np.isnan(std):
        std = 0.0
    return mean, max(0.0, std)


def expected_max_sharpe(
    n_trials: int,
    sr_trials_std: float,
    *,
    mean_sr: float = 0.0,
) -> float:
    """Expected maximum Sharpe under multiple testing (Bailey & Lopez de Prado).

    ``sr_trials_std`` is the **cross-trial** standard deviation of Sharpe ratios
    across the N candidates (selection-bias scale), not a single-strategy
    estimation standard error.
    """
    n = max(1, int(n_trials))
    sigma = max(0.0, float(sr_trials_std))
    mu = float(mean_sr)
    if n <= 1 or sigma <= 0.0:
        return mu
    emc = 0.5772156649015329  # Euler-Mascheroni
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    if np.isnan(z1) or np.isnan(z2):
        return mu
    return float(mu + sigma * ((1.0 - emc) * z1 + emc * z2))


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    n_trials: int = 1,
    trial_sharpes: list[float] | np.ndarray | None = None,
    sr_trials_std: float | None = None,
    sr_trials_mean: float | None = None,
) -> tuple[float, float, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Returns ``(dsr, sr_star, nonannual_sr)`` where
    ``dsr = Phi[(SR - SR*) / sqrt(V[SR])]``.

    - ``sqrt(V[SR])``: estimation SE of *this* strategy's Sharpe (skew/kurt).
    - ``SR* = E[max SR_n]``: uses the **cross-trial** Sharpe dispersion
      (``trial_sharpes`` / ``sr_trials_std``). Under a homogeneous null with
      no family provided, falls back to ``sqrt(V[SR])`` (same scale as independent
      zero-true-SR estimators) -- prefer passing the candidate family.
    """
    r = returns.fillna(0.0).to_numpy(dtype=float)
    n = len(r)
    if n < 5 or np.allclose(r, 0.0):
        return 0.0, 0.0, 0.0
    sr = nonannual_sharpe(r)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # Pearson kurtosis
    if np.isnan(skew):
        skew = 0.0
    if np.isnan(kurt) or kurt < 1.0:
        kurt = 3.0
    var_sr = sharpe_ratio_variance(sr, n, skew, kurt)
    if var_sr <= 0.0 or np.isnan(var_sr):
        return 0.0, 0.0, sr
    sr_se = float(np.sqrt(var_sr))  # estimation SE (DSR denominator)

    # Selection-bias scale: family dispersion of Sharpes, not sr_se.
    mean_for_star = 0.0
    if trial_sharpes is not None:
        emp_mean, emp_std = trial_sharpe_dispersion(trial_sharpes)
        trials_std = emp_std
        if sr_trials_mean is not None:
            mean_for_star = float(sr_trials_mean)
        else:
            # Null-centered threshold (avoid baking skill into SR*).
            mean_for_star = 0.0
        _ = emp_mean  # available for diagnostics; unused under null centering
    elif sr_trials_std is not None:
        trials_std = max(0.0, float(sr_trials_std))
        mean_for_star = float(sr_trials_mean or 0.0)
    else:
        # Homogeneous-null approximation: cross-section var ~= estimation var.
        trials_std = sr_se
        mean_for_star = float(sr_trials_mean or 0.0)

    sr_star = expected_max_sharpe(
        n_trials, trials_std, mean_sr=mean_for_star
    )
    z = (sr - sr_star) / sr_se
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
    Returns PBO in [0, 1] (higher = more overfit). Insufficient data -> 1.0.
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
    """Family-wise alpha for a *single-test* gate.

    - ``none``: unadjusted ``alpha``
    - ``bonferroni`` / ``holm``: ``alpha / m`` (Holm needs ordered p's; this is the
      first-step / conservative per-comparison stand-in)
    - ``fdr_bh``: returns unadjusted ``alpha``. True Benjamini-Hochberg needs the
      full p-value family -- use :func:`benjamini_hochberg_accept` after search.
      (Previously this incorrectly used ``alpha/m``, i.e. Bonferroni.)
    """
    a = max(0.0, float(alpha))
    m = max(1, int(n_tests))
    method = (method or "none").lower()
    if method in ("none", "off", ""):
        return a
    if method in ("bonferroni", "holm"):
        return a / m
    if method in ("fdr", "fdr_bh", "bh"):
        return a
    return a


def benjamini_hochberg_accept(
    p_values: list[float],
    *,
    alpha: float = 0.05,
) -> list[bool]:
    """Benjamini-Hochberg FDR control at level ``alpha``.

    Returns a boolean mask aligned with ``p_values`` (True = reject H0 / significant).
    """
    m = len(p_values)
    if m == 0:
        return []
    a = max(0.0, float(alpha))
    order = sorted(range(m), key=lambda i: float(p_values[i]))
    accepted = [False] * m
    max_k = -1
    for rank, idx in enumerate(order, start=1):
        p = float(p_values[idx])
        if p <= a * rank / m:
            max_k = rank
    if max_k < 0:
        return accepted
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            accepted[idx] = True
    return accepted


def min_bootstrap_pvalue(n_boot: int) -> float:
    """Smallest p-value ``(extremes+1)/(n_boot+1)`` can return (extremes=0)."""
    boots = max(0, int(n_boot))
    if boots <= 0:
        return 0.0
    return 1.0 / (boots + 1)


def bootstrap_gate_feasible(
    *,
    n_boot: int,
    alpha: float,
    n_tests: int,
    multiple_testing: str,
) -> bool:
    """False when Bonferroni (etc.) threshold is below the bootstrap resolution."""
    thr = adjust_alpha(alpha, n_tests, multiple_testing)
    return min_bootstrap_pvalue(n_boot) < thr


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
