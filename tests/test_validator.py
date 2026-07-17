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
    )
    v = StrategyValidator(cfg)
    full = _result(n_trades=8, p_value=0.01, trades=trades, regimes=regimes)
    # p=0.01 fails Bonferroni vs n_tests=10 → 0.005
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
        }
    )
    assert cfg.min_trades == 45
    assert cfg.min_oos_trades == 18
    assert cfg.min_regime_trades == 12
    assert cfg.block_bootstrap_reps == 100
    assert cfg.multiple_testing == "fdr_bh"
