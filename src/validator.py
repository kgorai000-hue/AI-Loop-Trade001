"""Strategy validation gates (reject overfitting / weak variants)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .backtest import BacktestResult, Backtester
from .strategy import StrategyParams
import pandas as pd


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
        }


class StrategyValidator:
    """
    Reject variants that fail any gate:
    - max DD < 10%
    - 1.5 <= Sharpe <= 3.0 (Sharpe > 3.0 → overfitting flag + reject)
    - p-value < 0.05
    - OOS Sharpe degradation <= 30%
    """

    def __init__(self, config: Optional[ValidatorConfig] = None) -> None:
        self.config = config or ValidatorConfig()

    def validate_reports(
        self,
        full: BacktestResult,
        is_result: BacktestResult,
        oos_result: BacktestResult,
        oos_deg: float,
    ) -> ValidationResult:
        cfg = self.config
        reasons: list[str] = []
        flags: list[str] = []
        overfitting = False

        sharpe = full.report.sharpe
        dd = full.report.max_drawdown
        p = full.report.p_value
        ic = full.report.ic

        if dd >= cfg.max_drawdown:
            reasons.append(f"max_drawdown {dd:.4f} >= {cfg.max_drawdown}")

        if sharpe > cfg.sharpe_max:
            overfitting = True
            flags.append("overfitting_suspected_sharpe_gt_max")
            reasons.append(f"sharpe {sharpe:.4f} > {cfg.sharpe_max} (data-snooping risk)")
        elif sharpe < cfg.sharpe_min:
            reasons.append(f"sharpe {sharpe:.4f} < {cfg.sharpe_min}")

        if p >= cfg.p_value_max:
            reasons.append(f"p_value {p:.4f} >= {cfg.p_value_max}")

        if oos_deg > cfg.oos_degradation_max:
            reasons.append(
                f"oos_degradation {oos_deg:.4f} > {cfg.oos_degradation_max}"
            )

        if full.report.n_trades < 5:
            reasons.append(f"insufficient_trades {full.report.n_trades}")

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
        )

    def validate(
        self,
        df: pd.DataFrame,
        params: StrategyParams,
        backtester: Optional[Backtester] = None,
        *,
        apply_oos_gate: bool = True,
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
        result = self.validate_reports(full, is_r, oos_r, deg)
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
        )