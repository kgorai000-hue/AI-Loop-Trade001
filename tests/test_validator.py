"""Tests for sample-size gates, block bootstrap, and multiple-testing correction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import BacktestResult
from src.metrics import (
    PerformanceReport,
    adjust_alpha,
    block_bootstrap_mean_pvalue,
    regime_trade_counts,
)
from src.strategy import Regime, StrategyParams
from src.validator import StrategyValidator, ValidatorConfig


def _report(**kwargs) -> PerformanceReport:
    base = dict(
        total_return=0.1,
        sharpe=2.0,
        max_drawdown=0.05,
        p_value=0.01,
        ic=0.1,
        n_trades=50,
        win_rate=0.55,
        reward_risk=1.2,
        equity_curve=pd.Series([1.0, 1.1]),
    )
    base.update(kwargs)
    return PerformanceReport(**base)


def _result(
    *,
    n_trades: int = 50,
    sharpe: float = 2.0,
    p_value: float = 0.01,
    max_drawdown: float = 0.05,
    bar_returns: pd.Series | None = None,
    trades: list | None = None,
    regimes: pd.Series | None = None,
) -> BacktestResult:
    rng = np.random.default_rng(0)
    if bar_returns is None:
        bar_returns = pd.Series(rng.normal(0.0005, 0.01, size=200))
    return BacktestResult(
        report=_report(
            n_trades=n_trades,
            sharpe=sharpe,
            p_value=p_value,
            max_drawdown=max_drawdown,
        ),
        trades=trades or [],
        bar_returns=bar_returns,
        signals=pd.Series([0] * len(bar_returns)),
        params=StrategyParams(200, 48, 12),
        regimes=regimes,
    )


def test_adjust_alpha_bonferroni():
    assert abs(adjust_alpha(0.05, 100, "bonferroni") - 0.0005) < 1e-12
    assert adjust_alpha(0.05, 1, "none") == 0.05


def test_adjust_alpha_fdr_bh_is_not_bonferroni():
    # Single-test adjust_alpha must not divide by m (true BH is batch).
    assert adjust_alpha(0.05, 120, "fdr_bh") == 0.05


def test_benjamini_hochberg_accepts_strong_signals():
    from src.inference import benjamini_hochberg_accept

    # One very small p among noise should survive BH at alpha=0.05
    ps = [0.0001] + [0.5] * 19
    mask = benjamini_hochberg_accept(ps, alpha=0.05)
    assert mask[0] is True
    assert sum(mask) >= 1


def test_bootstrap_gate_infeasible_under_legacy_defaults():
    from src.inference import bootstrap_gate_feasible, min_bootstrap_pvalue

    assert abs(min_bootstrap_pvalue(400) - 1.0 / 401) < 1e-12
    assert not bootstrap_gate_feasible(
        n_boot=400,
        alpha=0.05,
        n_tests=120,
        multiple_testing="bonferroni",
    )
    assert bootstrap_gate_feasible(
        n_boot=400,
        alpha=0.05,
        n_tests=120,
        multiple_testing="none",
    )


def test_validator_falls_back_to_hac_when_bootstrap_gate_impossible():
    rng = np.random.default_rng(0)
    edged = pd.Series(rng.normal(0.002, 0.01, size=400))
    cfg = ValidatorConfig(
        min_trades=1,
        min_oos_trades=1,
        min_regime_trades=0,
        block_bootstrap_reps=400,
        multiple_testing="bonferroni",
        sharpe_min=0.0,
        sharpe_max=100.0,
        p_value_max=0.05,
        oos_degradation_max=1.0,
        max_drawdown=1.0,
        min_dsr=0.0,
        pbo_enabled=False,
        pvalue_method="max",
    )
    v = StrategyValidator(cfg)
    full = _result(n_trades=50, bar_returns=edged, sharpe=2.0)
    res = v.validate_reports(full, full, full, 0.0, n_tests=120)
    assert "bootstrap_gate_infeasible_fallback_hac" in res.flags
    # Gate p is HAC (can be below Bonferroni floor), not floored bootstrap.
    assert res.p_value == res.hac_p_value


def test_block_bootstrap_rejects_zero_mean():
    rng = np.random.default_rng(1)
    zeroish = pd.Series(rng.normal(0.0, 0.01, size=800))
    p = block_bootstrap_mean_pvalue(zeroish, block_size=40, n_boot=500, seed=1)
    # Under H0 the p-value should not be extremely small.
    assert p > 0.01


def test_block_bootstrap_detects_positive_drift():
    rng = np.random.default_rng(2)
    edged = pd.Series(rng.normal(0.002, 0.01, size=400))
    p = block_bootstrap_mean_pvalue(edged, block_size=20, n_boot=300, seed=2)
    assert p < 0.05


def test_circular_block_resample_fixed_length_at_tail():
    from src.inference import circular_block_resample

    data = np.arange(10, dtype=float)
    # Start at index 8 with block=4 → must wrap: 8,9,0,1 (not a short 8,9)
    out = circular_block_resample(data, block=4, starts=np.array([8, 2, 5]))
    assert len(out) == 10
    assert list(out[:4]) == [8.0, 9.0, 0.0, 1.0]


def test_moving_block_resample_rejects_short_tail_starts():
    from src.inference import moving_block_resample
    import pytest

    data = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="moving-block starts"):
        moving_block_resample(data, block=4, starts=np.array([7]))  # 7+4 > 10
    out = moving_block_resample(data, block=4, starts=np.array([6, 0, 3]))
    assert len(out) == 10
    assert list(out[:4]) == [6.0, 7.0, 8.0, 9.0]


def test_block_bootstrap_moving_scheme_runs():
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(0.001, 0.01, size=300))
    p = block_bootstrap_mean_pvalue(
        series, block_size=25, n_boot=80, seed=3, scheme="moving"
    )
    assert 0.0 < p <= 1.0


def test_regime_trade_counts():
    regimes = pd.Series(
        [Regime.TREND] * 5 + [Regime.MEAN_REVERSION] * 5
    )
    trades = [
        {"entry_i": 0},
        {"entry_i": 1},
        {"entry_i": 6},
        {"entry_i": 7},
        {"entry_i": 8},
    ]
    counts = regime_trade_counts(trades, regimes)
    assert counts[Regime.TREND] == 2
    assert counts[Regime.MEAN_REVERSION] == 3


def test_validator_rejects_low_trade_counts():
    cfg = ValidatorConfig(
        min_trades=40,
        min_oos_trades=15,
        min_regime_trades=0,
        block_bootstrap_reps=0,
        multiple_testing="none",
        sharpe_min=0.0,
        sharpe_max=10.0,
        p_value_max=1.0,
        oos_degradation_max=1.0,
        max_drawdown=1.0,
        min_dsr=0.0,
        pbo_enabled=False,
        pvalue_method="hac",
    )
    v = StrategyValidator(cfg)
    full = _result(n_trades=5)
    oos = _result(n_trades=2)
    res = v.validate_reports(full, full, oos, 0.0, n_tests=1)
    assert not res.accepted
    assert any("insufficient_trades" in r for r in res.reasons)
    assert any("insufficient_oos_trades" in r for r in res.reasons)


def test_validator_holdout_uses_oos_floor():
    cfg = ValidatorConfig(
        min_trades=40,
        min_oos_trades=15,
        min_regime_trades=0,
        block_bootstrap_reps=0,
        multiple_testing="none",
        sharpe_min=0.0,
        sharpe_max=10.0,
        p_value_max=1.0,
        oos_degradation_max=1.0,
        max_drawdown=1.0,
        min_dsr=0.0,
        pbo_enabled=False,
        pvalue_method="hac",
    )
    v = StrategyValidator(cfg)
    holdout = _result(n_trades=20)
    ok = v.validate_reports(
        holdout, holdout, holdout, 0.0, check_oos_trades=False, n_tests=1
    )
    assert ok.accepted

    thin = _result(n_trades=5)
    bad = v.validate_reports(
        thin, thin, thin, 0.0, check_oos_trades=False, n_tests=1
    )
    assert not bad.accepted
    assert any("insufficient_trades" in r for r in bad.reasons)


def test_validator_regime_and_bonferroni():
    regimes = pd.Series(
        [Regime.TREND] * 30 + [Regime.MEAN_REVERSION] * 30
    )
    trades = [{"entry_i": i, "pnl": 0.01} for i in range(8)]  # all trend
    # Near-zero returns → high HAC p; Bonferroni tightens further.
    flat = pd.Series(np.zeros(60))
    cfg = ValidatorConfig(
        min_trades=5,
        min_oos_trades=1,
        min_regime_trades=10,
        block_bootstrap_reps=0,
        multiple_testing="bonferroni",
        sharpe_min=0.0,
        sharpe_max=10.0,
        p_value_max=0.05,
        oos_degradation_max=1.0,
        max_drawdown=1.0,
        min_dsr=0.0,
        pbo_enabled=False,
        pvalue_method="hac",
    )
    v = StrategyValidator(cfg)
    full = _result(
        n_trades=8,
        p_value=0.01,
        trades=trades,
        regimes=regimes,
        bar_returns=flat,
    )
    res = v.validate_reports(full, full, full, 0.0, n_tests=10)
    assert not res.accepted
    assert abs(res.p_value_threshold - 0.005) < 1e-12
    assert any("insufficient_regime_trades" in r for r in res.reasons)
    assert any("p_value" in r for r in res.reasons)


def test_config_from_dict_loads_new_keys():
    cfg = StrategyValidator.config_from_dict(
        {
            "min_trades": 45,
            "min_oos_trades": 18,
            "min_regime_trades": 12,
            "block_bootstrap_reps": 100,
            "block_bootstrap_block_bars": 24,
            "multiple_testing": "fdr_bh",
            "pvalue_method": "hac",
            "min_dsr": 0.9,
            "pbo_max": 0.4,
            "pbo_slices": 10,
        }
    )
    assert cfg.min_trades == 45
    assert cfg.min_oos_trades == 18
    assert cfg.min_regime_trades == 12
    assert cfg.block_bootstrap_reps == 100
    assert cfg.multiple_testing == "fdr_bh"
    assert cfg.pvalue_method == "hac"
    assert cfg.min_dsr == 0.9
    assert cfg.pbo_max == 0.4
    assert cfg.pbo_slices == 10


def test_default_config_allows_hac_under_grid_n_tests():
    """Default hac + none must be able to pass where bootstrap+Bonferroni cannot."""
    rng = np.random.default_rng(2)
    edged = pd.Series(rng.normal(0.003, 0.01, size=500))
    cfg = ValidatorConfig(
        min_trades=1,
        min_oos_trades=1,
        min_regime_trades=0,
        block_bootstrap_reps=400,
        multiple_testing="none",
        sharpe_min=0.0,
        sharpe_max=100.0,
        p_value_max=0.05,
        oos_degradation_max=1.0,
        max_drawdown=1.0,
        min_dsr=0.0,
        pbo_enabled=False,
        pvalue_method="hac",
    )
    v = StrategyValidator(cfg)
    full = _result(n_trades=50, bar_returns=edged, sharpe=2.0)
    res = v.validate_reports(full, full, full, 0.0, n_tests=120)
    assert res.p_value_threshold == 0.05
    assert res.p_value < 0.05
    assert res.accepted


def test_hac_inflates_p_under_autocorrelation():
    from src.metrics import hac_mean_pvalue, returns_pvalue

    rng = np.random.default_rng(7)
    e = rng.normal(0.0, 0.01, size=500)
    # AR(1) with mild positive drift — IID t-test overstates significance.
    r = np.zeros(500)
    for i in range(1, 500):
        r[i] = 0.6 * r[i - 1] + e[i] + 0.0003
    s = pd.Series(r)
    p_iid = returns_pvalue(s)
    p_hac = hac_mean_pvalue(s, lags=20)
    assert p_hac >= p_iid * 0.9  # HAC typically larger (more conservative)


def test_deflated_sharpe_penalizes_many_trials():
    from src.metrics import deflated_sharpe_ratio

    rng = np.random.default_rng(3)
    edged = pd.Series(rng.normal(0.0015, 0.01, size=600))
    dsr1, _, _ = deflated_sharpe_ratio(edged, n_trials=1)
    dsr_many, sr_star, _ = deflated_sharpe_ratio(edged, n_trials=120)
    assert dsr1 > dsr_many
    assert sr_star > 0.0


def test_dsr_sr_star_uses_trial_dispersion_not_estimation_se():
    from src.inference import deflated_sharpe_ratio, expected_max_sharpe, sharpe_ratio_variance
    from scipy import stats as sp_stats

    rng = np.random.default_rng(7)
    edged = pd.Series(rng.normal(0.002, 0.01, size=500))
    # Wide family of candidate Sharpes → larger SR* than homogeneous SE fallback.
    family = list(rng.normal(0.0, 0.15, size=40))
    _dsr_f, sr_star_f, sr = deflated_sharpe_ratio(
        edged, n_trials=40, trial_sharpes=family
    )
    r = edged.to_numpy(dtype=float)
    se = float(
        np.sqrt(
            sharpe_ratio_variance(
                sr,
                len(r),
                float(sp_stats.skew(r, bias=False)),
                float(sp_stats.kurtosis(r, fisher=False, bias=False)),
            )
        )
    )
    sr_star_se = expected_max_sharpe(40, se)
    assert sr_star_f > sr_star_se


def test_validator_defers_dsr_without_family():
    cfg = ValidatorConfig(
        min_trades=1,
        min_oos_trades=1,
        min_regime_trades=0,
        block_bootstrap_reps=0,
        multiple_testing="none",
        sharpe_min=0.0,
        sharpe_max=100.0,
        p_value_max=1.0,
        oos_degradation_max=1.0,
        max_drawdown=1.0,
        min_dsr=0.95,
        pbo_enabled=False,
        pvalue_method="hac",
    )
    v = StrategyValidator(cfg)
    rng = np.random.default_rng(4)
    full = _result(n_trades=50, bar_returns=pd.Series(rng.normal(0.001, 0.01, 200)))
    res = v.validate_reports(full, full, full, 0.0, n_tests=50)
    assert "dsr_gate_deferred_family" in res.flags
    assert not any("deflated_sharpe" in r for r in res.reasons)


def test_pbo_high_when_noise_strategies():
    from src.metrics import probability_of_backtest_overfitting

    rng = np.random.default_rng(11)
    # Unrelated noise strategies → high PBO
    mat = rng.normal(0.0, 0.01, size=(400, 20))
    pbo = probability_of_backtest_overfitting(mat, n_slices=8)
    assert pbo > 0.3


def test_validator_dsr_gate():
    cfg = ValidatorConfig(
        min_trades=1,
        min_oos_trades=1,
        min_regime_trades=0,
        block_bootstrap_reps=0,
        multiple_testing="none",
        sharpe_min=0.0,
        sharpe_max=100.0,
        p_value_max=1.0,
        oos_degradation_max=1.0,
        max_drawdown=1.0,
        min_dsr=0.99,
        pbo_enabled=False,
        pvalue_method="hac",
    )
    v = StrategyValidator(cfg)
    weak = _result(n_trades=50, bar_returns=pd.Series(np.zeros(100)))
    # n_tests=1 applies DSR immediately (no family deferral).
    res = v.validate_reports(weak, weak, weak, 0.0, n_tests=1)
    assert not res.accepted
    assert any("deflated_sharpe" in r for r in res.reasons)
