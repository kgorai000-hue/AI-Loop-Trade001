"""Per-symbol orchestration: data → signal → risk → execution → persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .backtest import Backtester
from .connection import MT5Connection
from .data import DataFeed
from .execution import OrderExecutor, OrderRequest
from .intelligence import IntelligenceLoop
from .persistence import StateStore
from .risk import CostModel, RiskManager
from .strategy import RegressionStrategy, Signal, StrategyDecision, StrategyParams
from .validator import StrategyValidator

logger = logging.getLogger(__name__)


@dataclass
class SymbolConfig:
    name: str
    state_key: str
    timeframe: str = "M30"
    long_window: int = 240
    short_window: int = 48
    max_hold_bars: int = 16
    enabled: bool = True


class SymbolTrader:
    """End-to-end controller for a single tradable symbol."""

    def __init__(
        self,
        symbol_cfg: SymbolConfig,
        connection: MT5Connection,
        executor: OrderExecutor,
        risk: RiskManager,
        state_store: StateStore,
        app_config: dict[str, Any],
    ) -> None:
        self.cfg = symbol_cfg
        self.connection = connection
        self.executor = executor
        self.risk = risk
        self.store = state_store
        self.app_config = app_config

        params = self.store.get_params()
        # Seed from config if state has defaults only and config differs
        if params.long_window == 240 and symbol_cfg.long_window != 240:
            params = StrategyParams(
                long_window=symbol_cfg.long_window,
                short_window=symbol_cfg.short_window,
                max_hold_bars=symbol_cfg.max_hold_bars,
            )
        self.strategy = RegressionStrategy(params)
        self.feed = DataFeed(connection, symbol_cfg.name, symbol_cfg.timeframe)
        self._last_bar_time = None
        self._regime_counts = {"trend": 0, "mean_reversion": 0, "unknown": 0}

    @property
    def symbol(self) -> str:
        return self.cfg.name

    def cost_model(self) -> CostModel:
        info = self.connection.symbol_info(self.symbol)
        tick = self.feed.tick()
        min_bps = float(self.app_config.get("risk", {}).get("min_cost_bps", 10))
        return CostModel.from_symbol_info(info, tick=tick, min_cost_bps=min_bps)

    def fetch_history(self, months: int = 6) -> Optional[Any]:
        return self.feed.last_n_months(months=months)

    def evaluate(self) -> Optional[StrategyDecision]:
        params = self.strategy.params
        need = params.long_window + 5
        df = self.feed.copy_rates(need)
        if df is None or len(df) < params.long_window:
            logger.warning("%s: insufficient bars for evaluation", self.symbol)
            return None
        decision = self.strategy.decide(df)
        self._regime_counts[decision.regime] = self._regime_counts.get(decision.regime, 0) + 1
        total = sum(self._regime_counts.values()) or 1
        self.store.update_state(
            regime_probability={
                "trend": self._regime_counts.get("trend", 0) / total,
                "mean_reversion": self._regime_counts.get("mean_reversion", 0) / total,
            }
        )
        return decision

    def sync_account_state(self) -> None:
        """Persist equity/margin/position snapshot into STATE.md."""
        account = self.connection.account_info()
        if account is None:
            return
        equity = float(account.equity)
        margin = float(getattr(account, "margin", 0.0) or 0.0)
        state = self.store.read_state()
        peak = state.get("equity_peak")
        try:
            peak_f = float(peak) if peak is not None else equity
        except (TypeError, ValueError):
            peak_f = equity
        if equity > peak_f:
            peak_f = equity

        side = "flat"
        lots = 0.0
        try:
            import MetaTrader5 as mt5

            positions = mt5.positions_get(symbol=self.symbol)
            if positions:
                pos = positions[0]
                side = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
                lots = float(pos.volume)
        except Exception:
            pass

        self.store.update_state(
            equity=equity,
            margin=margin,
            equity_peak=peak_f,
            position={"side": side, "lots": lots},
        )

    def maybe_trade(self, decision: StrategyDecision) -> dict[str, Any]:
        """Size and place a limit order for the decision signal."""
        result: dict[str, Any] = {
            "symbol": self.symbol,
            "signal": int(decision.signal),
            "regime": decision.regime,
            "b_long": decision.b_long,
            "b_short": decision.b_short,
            "order": None,
        }
        if self.store.is_locked():
            result["order"] = {"ok": False, "message": "system locked by kill-switch"}
            logger.warning("%s trade blocked: kill-switch locked", self.symbol)
            return result

        if decision.signal == Signal.FLAT:
            result["order"] = {"ok": True, "message": "flat"}
            self.sync_account_state()
            return result

        info = self.connection.symbol_info(self.symbol)
        account = self.connection.account_info()
        tick = self.feed.tick()
        if info is None or account is None or tick is None:
            result["order"] = {"ok": False, "message": "missing market/account info"}
            return result

        price = float(tick.ask if decision.signal == Signal.LONG else tick.bid)
        equity = float(account.equity)
        lots = self.risk.position_lots(
            equity=equity,
            price=price,
            contract_size=float(getattr(info, "trade_contract_size", 1) or 1),
            volume_step=float(getattr(info, "volume_step", 0.01) or 0.01),
            volume_min=float(getattr(info, "volume_min", 0.01) or 0.01),
            volume_max=float(getattr(info, "volume_max", 5) or 5),
        )
        result["lots"] = lots
        if lots <= 0:
            result["order"] = {"ok": False, "message": "kelly lots=0"}
            return result

        # Cancel prior pending for symbol then place new limit
        self.executor.cancel_pending(self.symbol)
        order = self.executor.place_limit(
            OrderRequest(
                symbol=self.symbol,
                side=decision.signal,
                volume=lots,
                price=0.0,
                comment=f"lr_{decision.regime[:8]}",
            )
        )
        result["order"] = {
            "ok": order.ok,
            "dry_run": order.dry_run,
            "message": order.message,
            "retcode": order.retcode,
            "order_id": order.order,
            "request": order.request,
        }
        self.sync_account_state()
        return result

    def on_new_bar(self) -> Optional[dict[str, Any]]:
        """Run once per new closed bar (caller detects bar change)."""
        df = self.feed.copy_rates(3)
        if df is None or df.empty:
            return None
        # Use second-to-last as last closed bar if last may still be forming
        bar_time = df["time"].iloc[-2] if len(df) >= 2 else df["time"].iloc[-1]
        if self._last_bar_time is not None and bar_time <= self._last_bar_time:
            return None
        self._last_bar_time = bar_time

        decision = self.evaluate()
        if decision is None:
            return None
        decision.bar_time = bar_time
        out = self.maybe_trade(decision)
        out["bar_time"] = str(bar_time)
        logger.info(
            "%s bar=%s signal=%s regime=%s bL=%.6f bS=%.6f",
            self.symbol,
            bar_time,
            decision.signal.name,
            decision.regime,
            decision.b_long,
            decision.b_short,
        )
        return out

    def backtest_and_validate(self, months: int = 6) -> dict[str, Any]:
        df = self.fetch_history(months=months)
        if df is None:
            return {"ok": False, "error": "no history"}
        cost = self.cost_model()
        bt = Backtester(cost_model=cost)
        from .validator import ValidatorConfig

        vcfg = self.app_config.get("validator", {})
        validator = StrategyValidator(
            ValidatorConfig(
                max_drawdown=float(vcfg.get("max_drawdown", 0.10)),
                sharpe_min=float(vcfg.get("sharpe_min", 1.5)),
                sharpe_max=float(vcfg.get("sharpe_max", 3.0)),
                p_value_max=float(vcfg.get("p_value_max", 0.05)),
                oos_degradation_max=float(vcfg.get("oos_degradation_max", 0.30)),
                is_fraction=float(vcfg.get("is_fraction", 0.70)),
            )
        )
        params = self.strategy.params
        val, full = validator.validate(df, params, backtester=bt)
        self.store.update_state(
            last_metrics={
                "sharpe": val.sharpe,
                "max_drawdown": val.max_drawdown,
                "p_value": val.p_value,
                "ic": val.ic,
                "oos_degradation": val.oos_degradation,
            },
            accepted=val.accepted,
            params=params.as_dict(),
        )
        if not val.accepted:
            for reason in val.reasons[:3]:
                self.store.append_lesson(f"Backtest rejected: {reason}")
        return {
            "ok": True,
            "validation": val.as_dict(),
            "report": full.report.as_dict(),
            "params": params.as_dict(),
        }

    def optimize(self, months: int = 6) -> dict[str, Any]:
        df = self.fetch_history(months=months)
        if df is None:
            return {"ok": False, "error": "no history"}
        intel = IntelligenceLoop(
            self.app_config,
            state_store=self.store,
            cost_model=self.cost_model(),
        )
        outcome = intel.run(df)
        if outcome.best_params is not None and outcome.best_validation is not None:
            self.strategy.update_params(outcome.best_params)
            v = outcome.best_validation
            self.store.update_state(
                params=outcome.best_params.as_dict(),
                accepted=True,
                last_metrics={
                    "sharpe": v.sharpe,
                    "max_drawdown": v.max_drawdown,
                    "p_value": v.p_value,
                    "ic": v.ic,
                    "oos_degradation": v.oos_degradation,
                },
            )
            logger.info(
                "%s applied optimized params %s via %s",
                self.symbol,
                outcome.best_params.as_dict(),
                outcome.path,
            )
        else:
            self.store.append_lesson(
                "Weekend/optimize cycle found no validator-passing params "
                f"(path={outcome.path})"
            )
        return {
            "ok": True,
            "path": outcome.path,
            "tried": outcome.tried,
            "accepted_count": outcome.accepted_count,
            "maker_proposed": outcome.maker_proposed,
            "checker_rejected": outcome.checker_rejected,
            "best_params": outcome.best_params.as_dict() if outcome.best_params else None,
            "best_validation": outcome.best_validation.as_dict() if outcome.best_validation else None,
            "top": outcome.rankings[:5],
        }

    def metrics_degraded(self, sharpe_trigger: float = 0.20, ic_trigger: float = 0.20) -> bool:
        """Compare a fresh backtest vs stored last_metrics."""
        state = self.store.read_state()
        last = state.get("last_metrics") or {}
        prev_sharpe = last.get("sharpe")
        prev_ic = last.get("ic")
        if prev_sharpe is None:
            return True  # never validated → trigger optimize

        fresh = self.backtest_and_validate(
            months=int(self.app_config.get("validator", {}).get("lookback_months", 6))
        )
        if not fresh.get("ok"):
            return False
        cur_sharpe = fresh["validation"]["sharpe"]
        cur_ic = fresh["validation"]["ic"]

        deg_s = 0.0
        if abs(prev_sharpe) > 1e-9:
            deg_s = max(0.0, (prev_sharpe - cur_sharpe) / abs(prev_sharpe))
        deg_i = 0.0
        if prev_ic is not None and abs(prev_ic) > 1e-9:
            deg_i = max(0.0, (prev_ic - cur_ic) / abs(prev_ic))

        degraded = deg_s >= sharpe_trigger or deg_i >= ic_trigger or not fresh["validation"]["accepted"]
        if degraded:
            self.store.append_lesson(
                f"Metrics degraded sharpe_drop={deg_s:.2%} ic_drop={deg_i:.2%} "
                f"accepted={fresh['validation']['accepted']}"
            )
        return degraded
