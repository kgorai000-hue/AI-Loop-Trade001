"""Self-improvement orchestrator: Maker → Checker → math Validator → grid fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from .anthropic_client import AnthropicClient
from .backtest import Backtester, BacktestResult
from .checker import StrategyChecker
from .maker import StrategyMaker
from .metrics import build_report
from .optimizer import OptimizeOutcome, ParameterOptimizer
from .persistence import StateStore
from .risk import CostModel
from .splits import chronological_holdout, walk_forward_folds, with_warmup
from .strategy import StrategyParams
from .validator import StrategyValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceOutcome:
    best_params: Optional[StrategyParams]
    best_validation: Optional[ValidationResult]
    tried: int
    accepted_count: int
    path: str  # llm | grid_fallback | unavailable | nested
    rankings: list[dict[str, Any]] = field(default_factory=list)
    checker_rejected: int = 0
    maker_proposed: int = 0
    fold_results: list[dict[str, Any]] = field(default_factory=list)
    holdout_validation: Optional[ValidationResult] = None
    outer_mean_sharpe: Optional[float] = None


class IntelligenceLoop:
    """
    LLM-first parameter search with mathematical Validator as final gate.
    Falls back to grid ParameterOptimizer when API unavailable or no LLM accept.

    Nested mode (``run_nested``):
      1) Peel final holdout (never used for selection)
      2) Outer walk-forward: inner search on train only, score on fold OOS
      3) Holdout gate once on the selected params
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
        self.validator = StrategyValidator(
            StrategyValidator.config_from_dict(app_config.get("validator"))
        )
        self.backtester = Backtester.from_app_config(
            app_config,
            cost_model=self.cost_model,
        )
        self.nested_inner = str(ocfg.get("nested_inner", "grid")).lower()

    def run(self, df: pd.DataFrame) -> IntelligenceOutcome:
        """Inner search on a single window (no outer holdout). Prefer ``run_nested``."""
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

    def run_nested(self, df: pd.DataFrame) -> IntelligenceOutcome:
        """Nested selection: inner search → outer walk-forward → holdout once."""
        vcfg = self.validator.config
        search_df, holdout = chronological_holdout(
            df,
            vcfg.holdout_fraction,
        )
        if search_df.empty:
            return IntelligenceOutcome(
                best_params=None,
                best_validation=None,
                tried=0,
                accepted_count=0,
                path="nested",
            )

        folds = walk_forward_folds(
            search_df,
            n_folds=vcfg.wf_folds,
            min_train_fraction=vcfg.wf_min_train_fraction,
        )
        if not folds:
            logger.warning("Nested search: insufficient data for WF folds; abort")
            return IntelligenceOutcome(
                best_params=None,
                best_validation=None,
                tried=0,
                accepted_count=0,
                path="nested",
            )

        fold_results: list[dict[str, Any]] = []
        tried = 0
        accepted_count = 0
        rankings: list[dict[str, Any]] = []

        for fold_i, (train, outer_oos) in enumerate(folds):
            logger.info(
                "Nested fold %d/%d train=%d outer_oos=%d",
                fold_i + 1,
                len(folds),
                len(train),
                len(outer_oos),
            )
            inner = self._run_inner(train)
            tried += inner.tried
            accepted_count += inner.accepted_count
            for row in inner.rankings:
                row = dict(row)
                row["fold"] = fold_i
                rankings.append(row)

            if inner.best_params is None:
                fold_results.append(
                    {
                        "fold": fold_i,
                        "params": None,
                        "outer_sharpe": None,
                        "accepted_inner": False,
                    }
                )
                continue

            outer_bt = self._run_on_segment(train, outer_oos, inner.best_params)
            outer_sharpe = float(outer_bt.report.sharpe)
            fold_results.append(
                {
                    "fold": fold_i,
                    "params": inner.best_params.as_dict(),
                    "outer_sharpe": outer_sharpe,
                    "outer_max_drawdown": float(outer_bt.report.max_drawdown),
                    "outer_n_trades": int(outer_bt.report.n_trades),
                    "accepted_inner": True,
                    "inner_sharpe": (
                        inner.best_validation.sharpe if inner.best_validation else None
                    ),
                    "inner_path": inner.path,
                }
            )

        candidates = [f for f in fold_results if f.get("params") and f.get("outer_sharpe") is not None]
        if not candidates:
            self.state_store.append_lesson(
                "Nested search: no fold produced an inner-accepted params set"
            )
            return IntelligenceOutcome(
                best_params=None,
                best_validation=None,
                tried=tried,
                accepted_count=accepted_count,
                path="nested",
                rankings=rankings,
                fold_results=fold_results,
            )

        # Prefer params with the best outer-fold Sharpe (selection uses outer OOS only,
        # never the final holdout).
        best_fold = max(candidates, key=lambda f: float(f["outer_sharpe"]))
        best_params = StrategyParams(
            long_window=int(best_fold["params"]["long_window"]),
            short_window=int(best_fold["params"]["short_window"]),
            max_hold_bars=int(best_fold["params"]["max_hold_bars"]),
        )

        # Score the chosen params on every outer fold for a stable mean.
        all_outer: list[float] = []
        for train, outer_oos in folds:
            scored = self._run_on_segment(train, outer_oos, best_params)
            all_outer.append(float(scored.report.sharpe))
        outer_mean = float(sum(all_outer) / len(all_outer)) if all_outer else None

        holdout_val: Optional[ValidationResult] = None
        if holdout.empty:
            logger.warning("Nested search: empty holdout; skipping final gate")
            accepted = True
        else:
            holdout_bt = self._run_on_segment(search_df, holdout, best_params)
            holdout_val = self.validator.validate_reports(
                holdout_bt, holdout_bt, holdout_bt, 0.0
            )
            accepted = bool(holdout_val.accepted)

        if not accepted:
            reasons = holdout_val.reasons if holdout_val else ["holdout failed"]
            self.state_store.append_lesson(
                f"Nested holdout rejected params={best_params.as_dict()}: "
                f"{', '.join(reasons[:3])}"
            )
            return IntelligenceOutcome(
                best_params=None,
                best_validation=holdout_val,
                tried=tried,
                accepted_count=accepted_count,
                path="nested",
                rankings=rankings,
                fold_results=fold_results,
                holdout_validation=holdout_val,
                outer_mean_sharpe=outer_mean,
            )

        logger.info(
            "Nested search accepted params=%s outer_mean_sharpe=%s holdout_sharpe=%s",
            best_params.as_dict(),
            outer_mean,
            holdout_val.sharpe if holdout_val else None,
        )
        return IntelligenceOutcome(
            best_params=best_params,
            best_validation=holdout_val,
            tried=tried,
            accepted_count=accepted_count,
            path="nested",
            rankings=rankings,
            fold_results=fold_results,
            holdout_validation=holdout_val,
            outer_mean_sharpe=outer_mean,
        )

    def _run_inner(self, train: pd.DataFrame) -> IntelligenceOutcome:
        if self.nested_inner == "auto":
            return self.run(train)
        return self._run_grid(train)

    def _run_on_segment(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        params: StrategyParams,
    ) -> BacktestResult:
        combined, warm = with_warmup(train, test, params.long_window)
        full = self.backtester.run(combined, params=params)
        if warm <= 0 or len(full.bar_returns) <= warm:
            return full
        rets = full.bar_returns.iloc[warm:].reset_index(drop=True)
        sigs = full.signals.iloc[warm:].reset_index(drop=True)
        trades = [t for t in full.trades if t["entry_i"] >= warm]
        report = build_report(
            rets,
            signals=sigs,
            trade_pnls=[t["pnl"] for t in trades],
            periods_per_year=self.backtester.periods_per_year,
            initial_equity=(
                float(self.backtester.account.initial_equity)
                if self.backtester.account.enabled
                else 1.0
            ),
        )
        return BacktestResult(
            report=report,
            trades=trades,
            bar_returns=rets,
            signals=sigs,
            params=params,
            unfilled_entries=full.unfilled_entries,
            liquidations=full.liquidations,
            skipped_entries=full.skipped_entries,
            fill_model=full.fill_model,
            account_config=full.account_config,
            final_equity=full.final_equity,
        )

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
