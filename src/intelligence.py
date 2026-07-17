"""Self-improvement orchestrator: Maker → Checker → math Validator → grid fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from .anthropic_client import AnthropicClient
from .backtest import Backtester
from .checker import StrategyChecker
from .maker import StrategyMaker
from .optimizer import OptimizeOutcome, ParameterOptimizer
from .persistence import StateStore
from .risk import CostModel
from .strategy import StrategyParams
from .validator import StrategyValidator, ValidationResult, ValidatorConfig

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceOutcome:
    best_params: Optional[StrategyParams]
    best_validation: Optional[ValidationResult]
    tried: int
    accepted_count: int
    path: str  # llm | grid_fallback | unavailable
    rankings: list[dict[str, Any]] = field(default_factory=list)
    checker_rejected: int = 0
    maker_proposed: int = 0


class IntelligenceLoop:
    """
    LLM-first parameter search with mathematical Validator as final gate.
    Falls back to grid ParameterOptimizer when API unavailable or no LLM accept.
    """

    def __init__(
        self,
        app_config: dict[str, Any],
        state_store: StateStore,
        cost_model: Optional[CostModel] = None,
    ) -> None:
        self.app_config = app_config
        self.state_store = state_store
        self.cost_model = cost_model or CostModel(
            min_cost_bps=float(app_config.get("risk", {}).get("min_cost_bps", 10))
        )
        acfg = app_config.get("anthropic", {})
        self.client = AnthropicClient(
            max_retries=int(acfg.get("max_retries", 5)),
            enable_prompt_cache=bool(acfg.get("enable_prompt_cache", True)),
        )
        ocfg = app_config.get("optimizer", {})
        longs = list(ocfg.get("long_windows", [180, 200, 220, 240, 260, 280]))
        shorts = list(ocfg.get("short_windows", [24, 36, 48, 60, 72]))
        holds = list(ocfg.get("max_hold_bars", [8, 12, 16, 24]))
        self.maker = StrategyMaker(
            client=self.client,
            model=str(acfg.get("maker_model", "claude-sonnet-4-5")),
            n_candidates=int(acfg.get("maker_candidates", 8)),
            long_range=(min(longs), max(longs)),
            short_range=(min(shorts), max(shorts)),
            hold_range=(min(holds), max(holds)),
        )
        self.checker = StrategyChecker(
            client=self.client,
            model=str(acfg.get("checker_model", "claude-opus-4")),
        )
        vcfg = app_config.get("validator", {})
        self.validator = StrategyValidator(
            ValidatorConfig(
                max_drawdown=float(vcfg.get("max_drawdown", 0.10)),
                sharpe_min=float(vcfg.get("sharpe_min", 1.5)),
                sharpe_max=float(vcfg.get("sharpe_max", 3.0)),
                p_value_max=float(vcfg.get("p_value_max", 0.05)),
                oos_degradation_max=float(vcfg.get("oos_degradation_max", 0.30)),
                is_fraction=float(vcfg.get("is_fraction", 0.70)),
            )
        )
        self.backtester = Backtester(cost_model=self.cost_model)

    def run(self, df: pd.DataFrame) -> IntelligenceOutcome:
        if self.client.available():
            llm_out = self._run_llm(df)
            if llm_out.best_params is not None:
                return llm_out
            logger.warning("LLM path produced no accepted params → grid fallback")
            grid = self._run_grid(df)
            grid.path = "grid_fallback"
            grid.maker_proposed = llm_out.maker_proposed
            grid.checker_rejected = llm_out.checker_rejected
            grid.rankings = llm_out.rankings + grid.rankings
            return grid

        logger.warning("ANTHROPIC_API_KEY missing → grid optimizer only")
        grid = self._run_grid(df)
        grid.path = "unavailable"
        return grid

    def _run_llm(self, df: pd.DataFrame) -> IntelligenceOutcome:
        state = self.state_store.read_state()
        current = self.state_store.get_params()
        skills = self.state_store.skills_text()
        last_metrics = state.get("last_metrics") or {}

        candidates = self.maker.propose(
            current_params=current,
            last_metrics=last_metrics,
            skills_text=skills,
        )
        maker_n = len(candidates)
        logger.info("Maker proposed %d candidates", maker_n)

        self.state_store.update_state(
            last_maker_run=datetime.now(timezone.utc).isoformat()
        )

        if not candidates:
            self.state_store.append_lesson("Maker returned zero valid candidates")
            return IntelligenceOutcome(
                best_params=None,
                best_validation=None,
                tried=0,
                accepted_count=0,
                path="llm",
                maker_proposed=0,
            )

        reviews = self.checker.review(candidates, skills_text=skills)
        approved = [r for r in reviews if r.approved]
        rejected = [r for r in reviews if not r.approved]
        for r in rejected:
            self.state_store.append_lesson(
                f"Checker reject params={r.params.as_dict()}: {r.reason}"
            )

        rankings: list[dict[str, Any]] = []
        best_params: Optional[StrategyParams] = None
        best_val: Optional[ValidationResult] = None
        accepted_count = 0

        for rev in approved:
            try:
                val, full = self.validator.validate(
                    df, rev.params, backtester=self.backtester
                )
            except Exception as exc:
                logger.warning("Validator failed for %s: %s", rev.params.as_dict(), exc)
                self.state_store.append_lesson(
                    f"Validator exception params={rev.params.as_dict()}: {exc}"
                )
                continue

            row = {
                "params": rev.params.as_dict(),
                "accepted": val.accepted,
                "sharpe": val.sharpe,
                "max_drawdown": val.max_drawdown,
                "p_value": val.p_value,
                "ic": val.ic,
                "oos_degradation": val.oos_degradation,
                "overfitting": val.overfitting,
                "reasons": val.reasons,
                "n_trades": full.report.n_trades,
                "source": "llm",
                "checker_reason": rev.reason,
            }
            rankings.append(row)

            if val.accepted:
                accepted_count += 1
                if best_val is None or val.sharpe > best_val.sharpe:
                    best_val = val
                    best_params = rev.params
            else:
                for reason in val.reasons[:2]:
                    self.state_store.append_lesson(
                        f"Validator reject params={rev.params.as_dict()}: {reason}"
                    )
                if val.overfitting:
                    self.state_store.append_lesson(
                        f"Rejected overfitting sharpe={val.sharpe:.2f} "
                        f"params={rev.params.as_dict()}"
                    )

        rankings.sort(key=lambda r: (r["accepted"], r["sharpe"]), reverse=True)
        return IntelligenceOutcome(
            best_params=best_params,
            best_validation=best_val,
            tried=len(approved),
            accepted_count=accepted_count,
            path="llm",
            rankings=rankings,
            checker_rejected=len(rejected),
            maker_proposed=maker_n,
        )

    def _run_grid(self, df: pd.DataFrame) -> IntelligenceOutcome:
        opt = ParameterOptimizer.from_config_dict(
            self.app_config,
            cost_model=self.cost_model,
            state_store=self.state_store,
        )
        outcome: OptimizeOutcome = opt.optimize(df)
        for row in outcome.rankings:
            row.setdefault("source", "grid")
        return IntelligenceOutcome(
            best_params=outcome.best_params,
            best_validation=outcome.best_validation,
            tried=outcome.tried,
            accepted_count=outcome.accepted_count,
            path="grid_fallback",
            rankings=outcome.rankings,
        )
