"""Strategy validation gates (reject overfitting / weak variants)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .backtest import BacktestResult, Backtester
from .inference import (
    adjust_alpha,
    block_bootstrap_mean_pvalue,
    bootstrap_gate_feasible,
    deflated_sharpe_ratio,
    hac_mean_pvalue,
    min_bootstrap_pvalue,
    regime_trade_counts,
    returns_pvalue,
)
from .strategy import Regime, StrategyParams

logger = logging.getLogger(__name__)


@dataclass
class ValidatorConfig:
    max_drawdown: float = 0.10
    sharpe_min: float = 1.5
    sharpe_max: float = 3.0
    p_value_max: float = 0.05
    oos_degradation_max: float = 0.30
    is_fraction: float = 0.70
    # Nested search: outer walk-forward + rolling OOS gate (periodic ops).
    # This is not a permanent sealed holdout — evaluated windows are persisted
    # and must not be reused; scores must not feed Maker/Checker/SKILL.
    rolling_oos_fraction: float = 0.15
    # Deprecated alias kept for ValidatorConfig(holdout_fraction=...) call sites.
    holdout_fraction: Optional[float] = None
    wf_folds: int = 3
    wf_min_train_fraction: float = 0.40
    # Sample-size / statistical rigor
    min_trades: int = 40
    min_oos_trades: int = 15
    min_regime_trades: int = 10
    block_bootstrap_reps: int = 400
    block_bootstrap_block_bars: int = 48
    # Selection bias: DSR/PBO are the primary family-wise gates. Per-test
    # Bonferroni on bootstrap p-values is often *mathematically impossible*
    # (min p = 1/(n_boot+1) > alpha/m). Default: unadjusted HAC p-gate.
    multiple_testing: str = "none"  # none | bonferroni | fdr_bh
    n_tests: int = 0  # 0 → use per-call n_tests or 1
    # pvalue_method: iid | hac | block_bootstrap | max
    # Gate uses this method; with ``hac`` (default) bootstrap is optional diagnostics.
    pvalue_method: str = "hac"
    hac_lags: int = 0  # 0 → Newey–West automatic bandwidth
    min_dsr: float = 0.95  # Deflated Sharpe Ratio floor (0 disables)
    pbo_max: float = 0.50  # Probability of Backtest Overfitting ceiling
    pbo_slices: int = 8  # CSCV partitions (even)
    pbo_enabled: bool = True
    # When True, still compute bootstrap p for reporting under hac gate.
    block_bootstrap_report: bool = False

    def __post_init__(self) -> None:
        if self.holdout_fraction is not None:
            self.rolling_oos_fraction = float(self.holdout_fraction)
        self.holdout_fraction = float(self.rolling_oos_fraction)


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
    dsr: float = 0.0
    sr_star: float = 0.0
    hac_p_value: float = 1.0
    bootstrap_p_value: float = 1.0
    pbo: Optional[float] = None

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
            "dsr": self.dsr,
            "sr_star": self.sr_star,
            "hac_p_value": self.hac_p_value,
            "bootstrap_p_value": self.bootstrap_p_value,
            "pbo": self.pbo,
        }


class StrategyValidator:
    """
    Reject variants that fail any gate:
    - max DD / Sharpe band / OOS degradation
    - HAC p-value by default (bootstrap optional; infeasible bootstrap+Bonferroni
      combinations fall back to HAC)
    - Deflated Sharpe Ratio + search PBO for selection bias
    - full-sample and OOS trade counts
    - per-regime trade counts (trend / mean_reversion)
    """

    def __init__(self, config: Optional[ValidatorConfig] = None) -> None:
        self.config = config or ValidatorConfig()

    def _infer_p_values(
        self, bar_returns: pd.Series, *, n_tests: int = 1
    ) -> tuple[float, float, float, list[str]]:
        """Return (gate_p, hac_p, boot_p, flags).

        Bootstrap resolution is ``1/(n_boot+1)``. If the configured multiple-testing
        threshold sits *below* that floor, a bootstrap (or max) gate can never pass —
        fall back to HAC for the gate and keep bootstrap as a diagnostic.
        """
        cfg = self.config
        flags: list[str] = []
        method = (cfg.pvalue_method or "hac").lower()
        hac_p = hac_mean_pvalue(bar_returns, lags=cfg.hac_lags)
        boot_p = 1.0
        wants_boot = cfg.block_bootstrap_reps > 0 and (
            method in ("block_bootstrap", "bootstrap", "max", "both")
            or bool(cfg.block_bootstrap_report)
        )
        if wants_boot:
            boot_p = block_bootstrap_mean_pvalue(
                bar_returns,
                block_size=cfg.block_bootstrap_block_bars,
                n_boot=cfg.block_bootstrap_reps,
            )
            flags.append("p_value_block_bootstrap")

        if method in ("iid", "t", "ttest"):
            p = returns_pvalue(bar_returns)
            flags.append("p_value_iid_ttest")
        elif method in ("hac", "newey_west", "nw"):
            p = hac_p
            flags.append("p_value_hac")
        elif method in ("block_bootstrap", "bootstrap"):
            p = boot_p if cfg.block_bootstrap_reps > 0 else hac_p
            if cfg.block_bootstrap_reps <= 0:
                flags.append("p_value_hac")
        else:
            # max / both: require the more conservative (larger) p
            p = max(hac_p, boot_p) if cfg.block_bootstrap_reps > 0 else hac_p
            flags.append("p_value_hac")
            if cfg.block_bootstrap_reps > 0:
                flags.append("p_value_max_hac_bootstrap")

        uses_boot_in_gate = method in ("block_bootstrap", "bootstrap", "max", "both")
        if (
            uses_boot_in_gate
            and cfg.block_bootstrap_reps > 0
            and not bootstrap_gate_feasible(
                n_boot=cfg.block_bootstrap_reps,
                alpha=cfg.p_value_max,
                n_tests=n_tests,
                multiple_testing=cfg.multiple_testing,
            )
        ):
            min_p = min_bootstrap_pvalue(cfg.block_bootstrap_reps)
            thr = adjust_alpha(cfg.p_value_max, n_tests, cfg.multiple_testing)
            logger.error(
                "Bootstrap gate infeasible: min_p=1/(%d+1)=%.6f >= threshold=%.6f "
                "(alpha=%s, n_tests=%s, multiple_testing=%s, pvalue_method=%s). "
                "Falling back to HAC for the p-value gate; raise block_bootstrap_reps "
                "or use multiple_testing=none with DSR/PBO.",
                cfg.block_bootstrap_reps,
                min_p,
                thr,
                cfg.p_value_max,
                n_tests,
                cfg.multiple_testing,
                method,
            )
            p = hac_p
            flags.append("bootstrap_gate_infeasible_fallback_hac")
            flags.append("p_value_hac")

        return float(p), float(hac_p), float(boot_p), flags

    def validate_reports(
        self,
        full: BacktestResult,
        is_result: BacktestResult,
        oos_result: BacktestResult,
        oos_deg: float,
        *,
        check_oos_trades: bool = True,
        n_tests: Optional[int] = None,
        pbo: Optional[float] = None,
        defer_p_value_gate: bool = False,
        trial_sharpes: Optional[list[float]] = None,
        sr_trials_std: Optional[float] = None,
        defer_dsr_gate: Optional[bool] = None,
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

        tests = int(n_tests) if n_tests is not None and n_tests > 0 else int(cfg.n_tests or 1)
        p, hac_p, boot_p, p_flags = self._infer_p_values(full.bar_returns, n_tests=tests)
        flags.extend(p_flags)
        p_threshold = adjust_alpha(cfg.p_value_max, tests, cfg.multiple_testing)
        mt = (cfg.multiple_testing or "none").lower()
        # True FDR-BH needs the full p-value family; defer until search post-process.
        defer_p = defer_p_value_gate or mt in ("fdr", "fdr_bh", "bh")

        has_family = trial_sharpes is not None or sr_trials_std is not None
        # Multi-trial DSR needs cross-sectional Sharpe dispersion; defer until the
        # search family is known (see apply_dsr_family_gate).
        if defer_dsr_gate is None:
            defer_dsr = bool(cfg.min_dsr > 0 and tests > 1 and not has_family)
        else:
            defer_dsr = bool(defer_dsr_gate)

        dsr, sr_star, _sr = deflated_sharpe_ratio(
            full.bar_returns,
            n_trials=tests,
            trial_sharpes=trial_sharpes,
            sr_trials_std=sr_trials_std,
        )
        if defer_dsr:
            flags.append("dsr_gate_deferred_family")
        elif not has_family and tests > 1:
            flags.append("dsr_trials_std_from_estimation_se")

        if dd >= cfg.max_drawdown:
            reasons.append(f"max_drawdown {dd:.4f} >= {cfg.max_drawdown}")

        if sharpe > cfg.sharpe_max:
            overfitting = True
            flags.append("overfitting_suspected_sharpe_gt_max")
            reasons.append(f"sharpe {sharpe:.4f} > {cfg.sharpe_max} (data-snooping risk)")
        elif sharpe < cfg.sharpe_min:
            reasons.append(f"sharpe {sharpe:.4f} < {cfg.sharpe_min}")

        if not defer_p and p >= p_threshold:
            reasons.append(
                f"p_value {p:.4f} >= {p_threshold:.6f} "
                f"(alpha={cfg.p_value_max}, n_tests={tests}, method={cfg.multiple_testing}, "
                f"pvalue={cfg.pvalue_method})"
            )
        elif defer_p:
            flags.append("p_value_gate_deferred_fdr_bh")

        if cfg.min_dsr > 0 and not defer_dsr and dsr < cfg.min_dsr:
            reasons.append(
                f"deflated_sharpe {dsr:.4f} < {cfg.min_dsr} "
                f"(sr_star={sr_star:.6f}, n_trials={tests})"
            )
            flags.append("dsr_gate")

        if pbo is not None and cfg.pbo_enabled and cfg.pbo_max < 1.0:
            if pbo > cfg.pbo_max:
                reasons.append(f"pbo {pbo:.4f} > {cfg.pbo_max}")
                flags.append("pbo_gate")
                overfitting = True

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
            dsr=dsr,
            sr_star=sr_star,
            hac_p_value=hac_p,
            bootstrap_p_value=boot_p,
            pbo=pbo,
        )

    def validate(
        self,
        df: pd.DataFrame,
        params: StrategyParams,
        backtester: Optional[Backtester] = None,
        *,
        apply_oos_gate: bool = True,
        n_tests: Optional[int] = None,
        pbo: Optional[float] = None,
    ) -> tuple[ValidationResult, BacktestResult]:
        """Validate params on ``df``.

        When ``apply_oos_gate`` is True (inner search), also compute an IS/OOS
        split *within* ``df`` for degradation. When False (rolling OOS gate),
        score only the window metrics so the gate is not re-split for selection.
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
            pbo=pbo,
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
            rolling_oos_fraction=float(
                vcfg.get(
                    "rolling_oos_fraction",
                    vcfg.get("holdout_fraction", 0.15),
                )
            ),
            wf_folds=int(vcfg.get("wf_folds", 3)),
            wf_min_train_fraction=float(vcfg.get("wf_min_train_fraction", 0.40)),
            min_trades=int(vcfg.get("min_trades", 40)),
            min_oos_trades=int(vcfg.get("min_oos_trades", 15)),
            min_regime_trades=int(vcfg.get("min_regime_trades", 10)),
            block_bootstrap_reps=int(vcfg.get("block_bootstrap_reps", 400)),
            block_bootstrap_block_bars=int(vcfg.get("block_bootstrap_block_bars", 48)),
            multiple_testing=str(vcfg.get("multiple_testing", "none")),
            n_tests=int(vcfg.get("n_tests", 0)),
            pvalue_method=str(vcfg.get("pvalue_method", "hac")),
            hac_lags=int(vcfg.get("hac_lags", 0)),
            min_dsr=float(vcfg.get("min_dsr", 0.95)),
            pbo_max=float(vcfg.get("pbo_max", 0.50)),
            pbo_slices=int(vcfg.get("pbo_slices", 8)),
            pbo_enabled=bool(vcfg.get("pbo_enabled", True)),
            block_bootstrap_report=bool(vcfg.get("block_bootstrap_report", False)),
        )
