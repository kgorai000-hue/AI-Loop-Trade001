"""Per-symbol orchestration: closed bars → target position → execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from .backtest import Backtester, SymbolSpec
from .connection import MT5Connection
from .data import DataFeed
from .execution import OrderExecutor
from .intelligence import IntelligenceLoop
from .persistence import StateStore
from .risk import CostModel, RiskManager
from .strategy import RegressionStrategy, Signal, StrategyDecision, StrategyParams
from .validator import StrategyValidator

logger = logging.getLogger(__name__)

TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


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

    MAGIC = 260717

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
        history = self.store.read_state().get("recent_pnls") or []
        if isinstance(history, list) and history:
            self.risk.load_trade_history(history)

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
        """Evaluate only completed bars; the forming MT5 bar is never included."""
        params = self.strategy.params
        need = params.long_window + 5
        df = self.feed.copy_closed_rates(need)
        if df is None or len(df) < params.long_window:
            logger.warning("%s: insufficient closed bars for evaluation", self.symbol)
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

    def _position_age_bars(self, bar_time: Optional[pd.Timestamp]) -> int:
        if bar_time is None:
            return 0
        positions = self.executor.managed_positions(self.symbol, magic=self.MAGIC)
        if positions is None:
            logger.warning("%s: position age unknown (positions_get failed)", self.symbol)
            return 0
        opened = [int(getattr(p, "time", 0) or 0) for p in positions]
        opened = [t for t in opened if t > 0]
        if not opened:
            return 0
        current = pd.Timestamp(bar_time)
        if current.tzinfo is None:
            current = current.tz_localize("UTC")
        opened_at = pd.Timestamp(datetime.fromtimestamp(min(opened), tz=timezone.utc))
        minutes = TIMEFRAME_MINUTES.get(self.cfg.timeframe.upper(), 30)
        return max(0, int((current - opened_at).total_seconds() // (minutes * 60)))

    def sync_account_state(self) -> None:
        """Persist account and managed-position state into STATE.md."""
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

        positions = self.executor.managed_positions(self.symbol, magic=self.MAGIC)
        if positions is None:
            logger.warning(
                "%s: skipping position state sync (positions_get failed)", self.symbol
            )
            self.store.update_state(equity=equity, margin=margin, equity_peak=peak_f)
            return

        if not positions:
            position_state = {"side": "flat", "lots": 0.0, "entry_time": None}
        else:
            import MetaTrader5 as mt5

            sides = {"long" if p.type == mt5.POSITION_TYPE_BUY else "short" for p in positions}
            position_state = {
                "side": next(iter(sides)) if len(sides) == 1 else "mixed",
                "lots": sum(float(p.volume) for p in positions),
                "entry_time": min(int(getattr(p, "time", 0) or 0) for p in positions) or None,
            }

        self.store.update_state(
            equity=equity,
            margin=margin,
            equity_peak=peak_f,
            position=position_state,
        )

    def maybe_trade(self, decision: StrategyDecision) -> dict[str, Any]:
        """Reconcile actual positions to one target position for the decision."""
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

        target_signal = decision.signal
        age_bars = self._position_age_bars(decision.bar_time)
        if age_bars >= self.strategy.params.max_hold_bars:
            target_signal = Signal.FLAT
            result["forced_flat"] = "max_hold_bars"
            result["position_age_bars"] = age_bars

        info = self.connection.symbol_info(self.symbol)
        if info is None:
            result["order"] = {"ok": False, "message": "missing symbol info"}
            return result
        volume_step = float(getattr(info, "volume_step", 0.01) or 0.01)

        if target_signal == Signal.FLAT:
            reconciliation = self.executor.reconcile_target(
                symbol=self.symbol,
                side=Signal.FLAT,
                volume=0.0,
                volume_step=volume_step,
                magic=self.MAGIC,
            )
            result["order"] = reconciliation.as_dict()
            self._record_closed_trades(reconciliation)
            self.sync_account_state()
            return result

        account = self.connection.account_info()
        tick = self.feed.tick()
        if account is None or tick is None:
            result["order"] = {"ok": False, "message": "missing market/account info"}
            return result

        price = float(tick.ask if target_signal == Signal.LONG else tick.bid)
        point = float(getattr(info, "point", 0.01) or 0.01)
        tick_size = float(getattr(info, "trade_tick_size", 0) or 0) or point
        tick_value = float(getattr(info, "trade_tick_value", 0) or 0)
        digits = int(getattr(info, "digits", 2) or 2)
        free_margin = float(getattr(account, "margin_free", 0) or 0)
        margin_per_lot = self._margin_per_lot(target_signal, price)
        open_risk = self._open_risk_reserved(price=price, tick_size=tick_size, tick_value=tick_value)

        decision_lots = self.risk.position_lots(
            equity=float(account.equity),
            price=price,
            contract_size=float(getattr(info, "trade_contract_size", 1) or 1),
            volume_step=volume_step,
            volume_min=float(getattr(info, "volume_min", 0.01) or 0.01),
            volume_max=float(getattr(info, "volume_max", 5) or 5),
            tick_size=tick_size,
            tick_value=tick_value,
            point=point,
            side_long=target_signal == Signal.LONG,
            digits=digits,
            free_margin=free_margin if free_margin > 0 else None,
            margin_per_lot=margin_per_lot,
            open_risk=open_risk,
        )
        lots = decision_lots.lots
        result["lots"] = lots
        result["lot_decision"] = {
            "stop_loss": decision_lots.stop_loss,
            "stop_distance": decision_lots.stop_distance,
            "risk_capital": decision_lots.risk_capital,
            "risk_per_lot": decision_lots.risk_per_lot,
            "open_risk_reserved": decision_lots.open_risk_reserved,
            "margin_capped": decision_lots.margin_capped,
            "message": decision_lots.message,
        }
        if lots <= 0:
            result["order"] = {"ok": False, "message": f"lots=0 ({decision_lots.message})"}
            return result
        if decision_lots.stop_loss is None:
            result["order"] = {"ok": False, "message": "stop loss required but not computed"}
            return result

        rebalance_band = float(self.app_config.get("risk", {}).get("rebalance_band", 0.15))
        reconciliation = self.executor.reconcile_target(
            symbol=self.symbol,
            side=target_signal,
            volume=lots,
            volume_step=volume_step,
            comment=f"lr_{decision.regime[:8]}",
            magic=self.MAGIC,
            sl=decision_lots.stop_loss,
            rebalance_band=rebalance_band,
        )
        result["order"] = reconciliation.as_dict()
        self._record_closed_trades(reconciliation)
        self.sync_account_state()
        return result

    def _margin_per_lot(self, side: Signal, price: float) -> Optional[float]:
        try:
            import MetaTrader5 as mt5

            order_type = mt5.ORDER_TYPE_BUY if side == Signal.LONG else mt5.ORDER_TYPE_SELL
            margin = mt5.order_calc_margin(order_type, self.symbol, 1.0, float(price))
            if margin is None:
                return None
            value = float(margin)
            return value if value > 0 else None
        except Exception:
            logger.debug("%s order_calc_margin unavailable", self.symbol, exc_info=True)
            return None

    def _open_risk_reserved(self, *, price: float, tick_size: float, tick_value: float) -> float:
        positions = self.executor.managed_positions(self.symbol, magic=self.MAGIC)
        if positions is None:
            logger.warning("%s: open risk unknown (positions_get failed)", self.symbol)
            return float("inf")
        if not positions:
            return 0.0
        import MetaTrader5 as mt5

        info = self.connection.symbol_info(self.symbol)
        point = float(getattr(info, "point", 0.01) or 0.01) if info is not None else 0.01
        fallback = self.risk.stop_distance_price(price, point) * max(
            1.0, float(self.risk.gap_buffer_mult)
        )
        total = 0.0
        for p in positions:
            side_long = p.type == mt5.POSITION_TYPE_BUY
            sl_raw = float(getattr(p, "sl", 0) or 0)
            sl = sl_raw if sl_raw > 0 else None
            total += self.risk.estimate_position_open_risk(
                volume=float(p.volume),
                current_price=price,
                side_long=side_long,
                stop_loss=sl,
                tick_size=tick_size,
                tick_value=tick_value,
                fallback_stop_distance=fallback,
            )
        return total

    def _record_closed_trades(self, reconciliation: Any) -> None:
        recorded = False
        for order in getattr(reconciliation, "orders", []) or []:
            pnl = getattr(order, "closed_pnl", None)
            if pnl is None or getattr(order, "dry_run", False) or not getattr(order, "ok", False):
                continue
            self.risk.record_trade(float(pnl))
            recorded = True
        if recorded:
            self.store.update_state(recent_pnls=list(self.risk.recent_pnls))

    def on_new_bar(self) -> Optional[dict[str, Any]]:
        """Run once for each newly completed bar."""
        df = self.feed.copy_closed_rates(1)
        if df is None or df.empty:
            return None
        bar_time = df["time"].iloc[-1]
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
        symbol_spec = None
        initial_equity = None
        info = self.connection.symbol_info(self.symbol)
        if info is not None:
            symbol_spec = SymbolSpec.from_mt5_info(
                info, margin_per_lot=self._margin_per_lot(Signal.LONG, float(getattr(info, "bid", 0) or 0) or 1.0)
            )
        account = self.connection.account_info()
        if account is not None:
            eq = float(getattr(account, "equity", 0) or 0)
            if eq > 0:
                initial_equity = eq
        bt = Backtester.from_app_config(
            self.app_config,
            cost_model=cost,
            risk=self.risk,
            symbol_spec=symbol_spec,
            initial_equity=initial_equity,
        )
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
            return True

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
