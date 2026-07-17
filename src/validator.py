"""Strategy validation gates (reject overfitting / weak variants)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .backtest import BacktestResult, Backtester
from .metrics import (
    adjust_alpha,
    block_bootstrap_mean_pvalue,
    regime_trade_counts,
)
from .strategy import Regime, StrategyParams


@dataclass
class ValidatorConfig:
    max_drawdown: float = 0.10
    sharpe_min: float = 1.5
    sharpe_max: float = 3.0
    p_value_max: float = 0.05
    oos_degradation_max: float = 0.30
    is_fraction: float = 0.70
    # Nested search: outer walk-forward + final holdout (used once)
    holdout_fraction: float = 0.15
    wf_folds: int = 3
    wf_min_train_fraction: float = 0.40
    # Sample-size / statistical rigor
    min_trades: int = 40
    min_oos_trades: int = 15
    min_regime_trades: int = 10
    block_bootstrap_reps: int = 400
    block_bootstrap_block_bars: int = 48
    multiple_testing: str = "bonferroni"  # none | bonferroni | fdr_bh
    n_tests: int = 0  # 0 → use per-call n_tests or 1


@dataclass
class ValidationResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    p_value: float = 1.0
    ic: float = 0.0
    oos_degradation: float = 0.0
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    overfitting: bool = False
    n_trades: int = 0
    oos_n_trades: int = 0
    regime_trades: dict = field(default_factory=dict)
    p_value_threshold: float = 0.05

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "flags": list(self.flags),
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "p_value": self.p_value,
            "ic": self.ic,
            "oos_degradation": self.oos_degradation,
            "is_sharpe": self.is_sharpe,
            "oos_sharpe": self.oos_sharpe,
            "overfitting": self.overfitting,
            "n_trades": self.n_trades,
            "oos_n_trades": self.oos_n_trades,
            "regime_trades": dict(self.regime_trades),
            "p_value_threshold": self.p_value_threshold,
        }


class StrategyValidator:
    """
    Reject variants that fail any gate:
    - max DD / Sharpe band / OOS degradation
    - block-bootstrap p-value (multiple-testing adjusted)
    - full-sample and OOS trade counts
    - per-regime trade counts (trend / mean_reversion)
    """

    def __init__(self, config: Optional[ValidatorConfig] = None) -> None:
        self.config = config or ValidatorConfig()

    def validate_reports(
        self,
        full: BacktestResult,
        is_result: BacktestResult,
        oos_result: BacktestResult,
        oos_deg: float,
        *,
        check_oos_trades: bool = True,
        n_tests: Optional[int] = None,
    ) -> ValidationResult:
        cfg = self.config
        reasons: list[str] = []
        flags: list[str] = []
        overfitting = False

        sharpe = full.report.sharpe
        dd = full.report.max_drawdown
        ic = full.report.ic
        n_trades = int(full.report.n_trades)
        oos_n_trades = int(oos_result.report.n_trades)

        if cfg.block_bootstrap_reps > 0:
            p = block_bootstrap_mean_pvalue(
                full.bar_returns,
                block_size=cfg.block_bootstrap_block_bars,
                n_boot=cfg.block_bootstrap_reps,
            )
            flags.append("p_value_block_bootstrap")
        else:
            p = full.report.p_value

        tests = int(n_tests) if n_tests is not None and n_tests > 0 else int(cfg.n_tests or 1)
        p_threshold = adjust_alpha(cfg.p_value_max, tests, cfg.multiple_testing)

        if dd >= cfg.max_drawdown:
            reasons.append(f"max_drawdown {dd:.4f} >= {cfg.max_drawdown}")

        if sharpe > cfg.sharpe_max:
            overfitting = True
            flags.append("overfitting_suspected_sharpe_gt_max")
            reasons.append(f"sharpe {sharpe:.4f} > {cfg.sharpe_max} (data-snooping risk)")
        elif sharpe < cfg.sharpe_min:
            reasons.append(f"sharpe {sharpe:.4f} < {cfg.sharpe_min}")

        if p >= p_threshold:
            reasons.append(
                f"p_value {p:.4f} >= {p_threshold:.6f} "
                f"(alpha={cfg.p_value_max}, n_tests={tests}, method={cfg.multiple_testing})"
            )

        if oos_deg > cfg.oos_degradation_max:
            reasons.append(
                f"oos_degradation {oos_deg:.4f} > {cfg.oos_degradation_max}"
            )

        if check_oos_trades:
            if n_trades < cfg.min_trades:
                reasons.append(f"insufficient_trades {n_trades} < {cfg.min_trades}")
            if oos_n_trades < cfg.min_oos_trades:
                reasons.append(
                    f"insufficient_oos_trades {oos_n_trades} < {cfg.min_oos_trades}"
                )
        elif n_trades < cfg.min_oos_trades:
            # Holdout / single-window gate: OOS-level sample size only.
            reasons.append(
                f"insufficient_trades {n_trades} < {cfg.min_oos_trades}"
            )

        regime_counts = regime_trade_counts(full.trades, full.regimes)
        if check_oos_trades and cfg.min_regime_trades > 0:
            active_regimes = {
                str(r)
                for r in (
                    full.regimes.dropna().unique()
                    if full.regimes is not None
                    else []
                )
                if str(r) in (Regime.TREND, Regime.MEAN_REVERSION)
            }
            for reg in (Regime.TREND, Regime.MEAN_REVERSION):
                if reg not in active_regimes and regime_counts.get(reg, 0) == 0:
                    continue
                count = int(regime_counts.get(reg, 0))
                if count < cfg.min_regime_trades:
                    reasons.append(
                        f"insufficient_regime_trades {reg}={count} "
                        f"< {cfg.min_regime_trades}"
                    )

        accepted = len(reasons) == 0
        return ValidationResult(
            accepted=accepted,
            reasons=reasons,
            flags=flags,
            sharpe=sharpe,
            max_drawdown=dd,
            p_value=p,
            ic=ic,
            oos_degradation=oos_deg,
            is_sharpe=is_result.report.sharpe,
            oos_sharpe=oos_result.report.sharpe,
            overfitting=overfitting,
            n_trades=n_trades,
            oos_n_trades=oos_n_trades,
            regime_trades=regime_counts,
            p_value_threshold=p_threshold,
        )

    def validate(
        self,
        df: pd.DataFrame,
        params: StrategyParams,
        backtester: Optional[Backtester] = None,
        *,
        apply_oos_gate: bool = True,
        n_tests: Optional[int] = None,
    ) -> tuple[ValidationResult, BacktestResult]:
        """Validate params on ``df``.

        When ``apply_oos_gate`` is True (inner search), also compute an IS/OOS
        split *within* ``df`` for degradation. When False (final holdout), gate
        only on the window metrics so the holdout is not re-split for selection.
        """
        bt = backtester or Backtester()
        full = bt.run(df, params=params)
        if apply_oos_gate:
            is_r, oos_r, deg = bt.run_is_oos(
                df, params=params, is_fraction=self.config.is_fraction
            )
        else:
            is_r, oos_r, deg = full, full, 0.0
        result = self.validate_reports(
            full,
            is_r,
            oos_r,
            deg,
            check_oos_trades=apply_oos_gate,
            n_tests=n_tests,
        )
        return result, full

    @classmethod
    def config_from_dict(cls, vcfg: Optional[dict] = None) -> ValidatorConfig:
        vcfg = vcfg or {}
        return ValidatorConfig(
            max_drawdown=float(vcfg.get("max_drawdown", 0.10)),
            sharpe_min=float(vcfg.get("sharpe_min", 1.5)),
            sharpe_max=float(vcfg.get("sharpe_max", 3.0)),
            p_value_max=float(vcfg.get("p_value_max", 0.05)),
            oos_degradation_max=float(vcfg.get("oos_degradation_max", 0.30)),
            is_fraction=float(vcfg.get("is_fraction", 0.70)),
            holdout_fraction=float(vcfg.get("holdout_fraction", 0.15)),
            wf_folds=int(vcfg.get("wf_folds", 3)),
            wf_min_train_fraction=float(vcfg.get("wf_min_train_fraction", 0.40)),
            min_trades=int(vcfg.get("min_trades", 40)),
            min_oos_trades=int(vcfg.get("min_oos_trades", 15)),
            min_regime_trades=int(vcfg.get("min_regime_trades", 10)),
            block_bootstrap_reps=int(vcfg.get("block_bootstrap_reps", 400)),
            block_bootstrap_block_bars=int(vcfg.get("block_bootstrap_block_bars", 48)),
            multiple_testing=str(vcfg.get("multiple_testing", "bonferroni")),
            n_tests=int(vcfg.get("n_tests", 0)),
        )
