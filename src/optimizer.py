"""Parameter grid search constrained by Validator and SKILL lessons."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .backtest import Backtester
from .persistence import StateStore
from .risk import CostModel
from .strategy import StrategyParams
from .validator import StrategyValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    long_windows: list[int] = field(default_factory=lambda: [180, 200, 220, 240, 260, 280])
    short_windows: list[int] = field(default_factory=lambda: [24, 36, 48, 60, 72])
    max_hold_bars: list[int] = field(default_factory=lambda: [8, 12, 16, 24])


@dataclass
class OptimizeOutcome:
    best_params: Optional[StrategyParams]
    best_validation: Optional[ValidationResult]
    tried: int
    accepted_count: int
    rankings: list[dict[str, Any]] = field(default_factory=list)


class ParameterOptimizer:
    def __init__(
        self,
        validator: Optional[StrategyValidator] = None,
        backtester: Optional[Backtester] = None,
        config: Optional[OptimizerConfig] = None,
        state_store: Optional[StateStore] = None,
    ) -> None:
        self.validator = validator or StrategyValidator()
        self.backtester = backtester or Backtester()
        self.config = config or OptimizerConfig()
        self.state_store = state_store

    def _candidate_params(self) -> list[StrategyParams]:
        hints = self.state_store.lessons_as_constraints() if self.state_store else {}
        avoid_long_gt = hints.get("avoid_long_gt")
        avoid_short_lt = hints.get("avoid_short_lt")
        avoid_hold_gt = hints.get("avoid_hold_gt")

        out: list[StrategyParams] = []
        for lw, sw, mh in itertools.product(
            self.config.long_windows,
            self.config.short_windows,
            self.config.max_hold_bars,
        ):
            if sw >= lw:
                continue
            if avoid_long_gt is not None and lw > avoid_long_gt:
                continue
            if avoid_short_lt is not None and sw < avoid_short_lt:
                continue
            if avoid_hold_gt is not None and mh > avoid_hold_gt:
                continue
            out.append(StrategyParams(long_window=lw, short_window=sw, max_hold_bars=mh))
        return out

    def optimize(self, df: pd.DataFrame) -> OptimizeOutcome:
        rankings: list[dict[str, Any]] = []
        best_params: Optional[StrategyParams] = None
        best_val: Optional[ValidationResult] = None
        accepted_count = 0

        candidates = self._candidate_params()
        logger.info("Optimizer evaluating %d candidates", len(candidates))

        for params in candidates:
            try:
                val, full = self.validator.validate(df, params, backtester=self.backtester)
            except Exception as exc:
                logger.warning("Candidate %s failed: %s", params.as_dict(), exc)
                continue

            row = {
                "params": params.as_dict(),
                "accepted": val.accepted,
                "sharpe": val.sharpe,
                "max_drawdown": val.max_drawdown,
                "p_value": val.p_value,
                "ic": val.ic,
                "oos_degradation": val.oos_degradation,
                "overfitting": val.overfitting,
                "reasons": val.reasons,
                "n_trades": full.report.n_trades,
            }
            rankings.append(row)

            if val.accepted:
                accepted_count += 1
                if best_val is None or val.sharpe > best_val.sharpe:
                    # Prefer higher Sharpe within [1.5, 3.0]
                    best_val = val
                    best_params = params
            elif val.overfitting and self.state_store:
                self.state_store.append_lesson(
                    f"Rejected overfitting sharpe={val.sharpe:.2f} params={params.as_dict()}"
                )

        rankings.sort(key=lambda r: (r["accepted"], r["sharpe"]), reverse=True)

        if best_params is None and rankings:
            # Record top failure reason into SKILL for next cycle
            top = rankings[0]
            if self.state_store and top.get("reasons"):
                self.state_store.append_lesson(
                    f"No accepted variant; best rejected due to: {', '.join(top['reasons'][:3])}"
                )

        return OptimizeOutcome(
            best_params=best_params,
            best_validation=best_val,
            tried=len(candidates),
            accepted_count=accepted_count,
            rankings=rankings,
        )

    @classmethod
    def from_config_dict(
        cls,
        cfg: dict[str, Any],
        cost_model: Optional[CostModel] = None,
        state_store: Optional[StateStore] = None,
    ) -> "ParameterOptimizer":
        vcfg = cfg.get("validator", {})
        ocfg = cfg.get("optimizer", {})
        validator = StrategyValidator(StrategyValidator.config_from_dict(vcfg))
        bt = Backtester.from_app_config(cfg, cost_model=cost_model)
        opt_cfg = OptimizerConfig(
            long_windows=list(ocfg.get("long_windows", OptimizerConfig().long_windows)),
            short_windows=list(ocfg.get("short_windows", OptimizerConfig().short_windows)),
            max_hold_bars=list(ocfg.get("max_hold_bars", OptimizerConfig().max_hold_bars)),
        )
        return cls(validator=validator, backtester=bt, config=opt_cfg, state_store=state_store)
