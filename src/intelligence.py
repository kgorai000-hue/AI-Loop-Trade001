"""Self-improvement orchestrator: Maker -> Checker -> math Validator -> grid fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from .anthropic_client import AnthropicClient
from .backtest import Backtester, BacktestResult, SymbolSpec
from .checker import StrategyChecker
from .maker import StrategyMaker
from .optimizer import OptimizeOutcome, OptimizerConfig, ParameterOptimizer
from .persistence import StateStore
from .risk import CostModel, RiskManager
from .search import apply_search_family_gates, validation_ranking_row
from .splits import chronological_rolling_oos, walk_forward_folds, with_warmup
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
    # Rolling OOS gate only -- never feed these metrics/reasons into Maker,
    # Checker, SKILL, or last_metrics (that would adapt to the gate).
    rolling_oos_validation: Optional[ValidationResult] = None
    rolling_oos_window: Optional[dict[str, str]] = None
    # Mean/median of *per-fold* OOS Sharpes (each fold's own selected params on
    # that fold's OOS only). Never re-apply one fold's winner across earlier OOS.
    outer_mean_sharpe: Optional[float] = None
    outer_median_sharpe: Optional[float] = None
    pbo: Optional[float] = None


class IntelligenceLoop:
    """
    LLM-first parameter search with mathematical Validator as final gate.
    Falls back to grid ParameterOptimizer when API unavailable or no LLM accept.

    Nested mode (``run_nested``) for periodic ops:
      1) Peel *rolling OOS* (not a permanent sealed holdout)
      2) Outer walk-forward: each fold selects on train, scores on *that* fold OOS
      3) Aggregate fold OOS Sharpes (mean/median) -- no cross-fold re-application
      4) Re-search final params on all pre-rolling-OOS data; gate once on rolling OOS
         (empty / reused window -> fail-closed: best_params=None)
      5) Rolling OOS results are not written to SKILL / Maker / Checker feedback
    """

    def __init__(
        self,
        app_config: dict[str, Any],
        state_store: StateStore,
        cost_model: Optional[CostModel] = None,
        *,
        risk: Optional[RiskManager] = None,
        symbol_spec: Optional[SymbolSpec] = None,
        initial_equity: Optional[float] = None,
        backtester: Optional[Backtester] = None,
    ) -> None:
        self.app_config = app_config
        self.state_store = state_store
        self.cost_model = cost_model or CostModel.from_risk_config(
            app_config.get("risk")
        )
        self.risk = risk
        self.symbol_spec = symbol_spec
        self.initial_equity = initial_equity
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
            model=str(acfg.get("checker_model", "claude-opus-4-8")),
        )
        self.validator = StrategyValidator(
            StrategyValidator.config_from_dict(app_config.get("validator"))
        )
        # Prefer the same live-aligned Backtester as backtest_and_validate().
        self.backtester = backtester or Backtester.from_app_config(
            app_config,
            cost_model=self.cost_model,
            risk=risk,
            symbol_spec=symbol_spec,
            initial_equity=initial_equity,
        )
        self.nested_inner = str(ocfg.get("nested_inner", "grid")).lower()

    def run(self, df: pd.DataFrame) -> IntelligenceOutcome:
        """Inner search on a single window (no outer holdout). Prefer ``run_nested``."""
        if self.client.available():
            llm_out = self._run_llm(df)
            if llm_out.best_params is not None:
                return llm_out
            logger.warning("LLM path produced no accepted params -> grid fallback")
            grid = self._run_grid(df)
            grid.path = "grid_fallback"
            grid.maker_proposed = llm_out.maker_proposed
            grid.checker_rejected = llm_out.checker_rejected
            grid.rankings = llm_out.rankings + grid.rankings
            return grid

        logger.warning("ANTHROPIC_API_KEY missing -> grid optimizer only")
        grid = self._run_grid(df)
        grid.path = "unavailable"
        return grid

    @staticmethod
    def _nested_outcome(**kwargs: Any) -> IntelligenceOutcome:
        kwargs.setdefault("path", "nested")
        kwargs.setdefault("best_params", None)
        kwargs.setdefault("best_validation", None)
        kwargs.setdefault("tried", 0)
        kwargs.setdefault("accepted_count", 0)
        return IntelligenceOutcome(**kwargs)

    def _score_walk_forward_folds(
        self,
        folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float], int, int]:
        fold_results: list[dict[str, Any]] = []
        rankings: list[dict[str, Any]] = []
        fold_oos_sharpes: list[float] = []
        tried = 0
        accepted_count = 0

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
            fold_oos_sharpes.append(outer_sharpe)
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
        return fold_results, rankings, fold_oos_sharpes, tried, accepted_count

    def _gate_rolling_oos(
        self,
        *,
        search_df: pd.DataFrame,
        rolling_oos: pd.DataFrame,
        oos_window: Optional[dict[str, str]],
        best_params: StrategyParams,
        exclude_oos_before: Optional[pd.Timestamp],
    ) -> tuple[Optional[ValidationResult], Optional[dict[str, str]], bool]:
        """Return ``(validation, window, accepted)``. Never writes SKILL."""
        if rolling_oos.empty or oos_window is None:
            logger.warning(
                "Nested search: empty rolling OOS (exclude_before=%s); "
                "rejecting optimization (fail-closed, no final gate)",
                exclude_oos_before,
            )
            return (
                ValidationResult(
                    accepted=False,
                    reasons=["empty rolling OOS; final gate unavailable"],
                ),
                None,
                False,
            )

        rolling_bt = self._run_on_segment(search_df, rolling_oos, best_params)
        rolling_val = self.validator.validate_reports(
            rolling_bt,
            rolling_bt,
            rolling_bt,
            0.0,
            check_oos_trades=False,
            n_tests=1,
        )
        if not rolling_val.accepted:
            logger.info(
                "Nested rolling OOS rejected params=%s reasons=%s "
                "(not written to SKILL)",
                best_params.as_dict(),
                (rolling_val.reasons or ["rolling OOS failed"])[:3],
            )
            return rolling_val, oos_window, False
        return rolling_val, oos_window, True

    def run_nested(
        self,
        df: pd.DataFrame,
        *,
        exclude_oos_before: Optional[pd.Timestamp] = None,
    ) -> IntelligenceOutcome:
        """Nested selection with rolling OOS gate (no cross-fold look-ahead).

        - Each WF fold picks params on its train window and is scored only on
          that fold's outer OOS (pure fold OOS).
        - ``outer_mean_sharpe`` / ``outer_median_sharpe`` aggregate those
          fold-specific OOS scores -- never re-apply one fold's winner to earlier
          OOS windows.
        - Final params are re-searched on all pre-rolling-OOS data, then gated
          once on a *new* rolling OOS window (bars after ``exclude_oos_before``).
        - Rolling OOS metrics/reasons are never appended to SKILL and must not be
          written into Maker/Checker-facing ``last_metrics``.
        """
        vcfg = self.validator.config
        search_df, rolling_oos, oos_window = chronological_rolling_oos(
            df,
            vcfg.rolling_oos_fraction,
            exclude_oos_before=exclude_oos_before,
        )
        if search_df.empty:
            return self._nested_outcome()

        folds = walk_forward_folds(
            search_df,
            n_folds=vcfg.wf_folds,
            min_train_fraction=vcfg.wf_min_train_fraction,
        )
        if not folds:
            logger.warning("Nested search: insufficient data for WF folds; abort")
            return self._nested_outcome()

        fold_results, rankings, fold_oos_sharpes, tried, accepted_count = (
            self._score_walk_forward_folds(folds)
        )

        if not fold_oos_sharpes:
            self.state_store.append_lesson(
                "Nested search: no fold produced an inner-accepted params set"
            )
            return self._nested_outcome(
                tried=tried,
                accepted_count=accepted_count,
                rankings=rankings,
                fold_results=fold_results,
            )

        outer_mean = float(sum(fold_oos_sharpes) / len(fold_oos_sharpes))
        outer_median = float(pd.Series(fold_oos_sharpes).median())

        logger.info(
            "Nested final re-search on pre-rolling-OOS bars=%d "
            "(fold_oos mean=%.4f median=%.4f n=%d)",
            len(search_df),
            outer_mean,
            outer_median,
            len(fold_oos_sharpes),
        )
        final_inner = self._run_inner(search_df)
        tried += final_inner.tried
        accepted_count += final_inner.accepted_count
        for row in final_inner.rankings:
            row = dict(row)
            row["fold"] = "final_pre_rolling_oos"
            rankings.append(row)

        if final_inner.best_params is None:
            self.state_store.append_lesson(
                "Nested search: final pre-rolling-OOS re-search found no accepted params"
            )
            return self._nested_outcome(
                tried=tried,
                accepted_count=accepted_count,
                rankings=rankings,
                fold_results=fold_results,
                outer_mean_sharpe=outer_mean,
                outer_median_sharpe=outer_median,
            )

        rolling_val, oos_window, accepted = self._gate_rolling_oos(
            search_df=search_df,
            rolling_oos=rolling_oos,
            oos_window=oos_window,
            best_params=final_inner.best_params,
            exclude_oos_before=exclude_oos_before,
        )
        if not accepted:
            return self._nested_outcome(
                tried=tried,
                accepted_count=accepted_count,
                rankings=rankings,
                fold_results=fold_results,
                rolling_oos_validation=rolling_val,
                rolling_oos_window=oos_window,
                outer_mean_sharpe=outer_mean,
                outer_median_sharpe=outer_median,
            )

        logger.info(
            "Nested search accepted params=%s fold_oos_mean=%s fold_oos_median=%s "
            "rolling_oos_window=%s",
            final_inner.best_params.as_dict(),
            outer_mean,
            outer_median,
            oos_window,
        )
        return self._nested_outcome(
            best_params=final_inner.best_params,
            best_validation=final_inner.best_validation,
            tried=tried,
            accepted_count=accepted_count,
            rankings=rankings,
            fold_results=fold_results,
            rolling_oos_validation=rolling_val,
            rolling_oos_window=oos_window,
            outer_mean_sharpe=outer_mean,
            outer_median_sharpe=outer_median,
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
        seg = combined.iloc[warm:].reset_index(drop=True)
        return self.backtester.trim_warmup(full, warm, close=seg["close"])

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
        return_series: list = []

        n_tests = max(maker_n, len(approved), 1)
        for rev in approved:
            try:
                val, full = self.validator.validate(
                    df,
                    rev.params,
                    backtester=self.backtester,
                    n_tests=n_tests,
                )
            except Exception as exc:
                logger.warning("Validator failed for %s: %s", rev.params.as_dict(), exc)
                self.state_store.append_lesson(
                    f"Validator exception params={rev.params.as_dict()}: {exc}"
                )
                continue

            return_series.append(full.bar_returns)
            rankings.append(
                validation_ranking_row(
                    val,
                    rev.params,
                    source="llm",
                    checker_reason=rev.reason,
                )
            )

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

        best_params, best_val, accepted_count, pbo = apply_search_family_gates(
            validator=self.validator,
            state_store=self.state_store,
            best_params=best_params,
            best_val=best_val,
            accepted_count=accepted_count,
            rankings=rankings,
            return_series=return_series,
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
            pbo=pbo,
        )

    def _run_grid(self, df: pd.DataFrame) -> IntelligenceOutcome:
        ocfg = self.app_config.get("optimizer", {})
        opt = ParameterOptimizer(
            validator=self.validator,
            backtester=self.backtester,
            config=OptimizerConfig(
                long_windows=list(
                    ocfg.get("long_windows", OptimizerConfig().long_windows)
                ),
                short_windows=list(
                    ocfg.get("short_windows", OptimizerConfig().short_windows)
                ),
                max_hold_bars=list(
                    ocfg.get("max_hold_bars", OptimizerConfig().max_hold_bars)
                ),
            ),
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
            pbo=outcome.pbo,
        )
