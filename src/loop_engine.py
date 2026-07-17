"""Resident loop: M30 bar polling + weekend review / re-optimization sub-loop."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .connection import MT5Connection
from .execution import OrderExecutor
from .kill_switch import KillSwitchMonitor
from .persistence import StateStore
from .risk import RiskManager
from .symbol_trader import SymbolConfig, SymbolTrader

logger = logging.getLogger(__name__)


class LoopEngine:
    """
    Holds a list of SymbolTrader instances (multi-symbol ready).
    - Polls every `poll_seconds` for new closed bars.
    - On configured weekday/hour, runs review sub-loop (metric check → optimize).
    - KillSwitchMonitor thread flattens and locks on account DD breach.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        mt5_cfg = config.get("mt5", {})
        self.connection = MT5Connection(
            login=int(mt5_cfg.get("login") or 0),
            password=str(mt5_cfg.get("password") or ""),
            server=str(mt5_cfg.get("server") or "FxPro-Demo"),
            path=str(mt5_cfg.get("path") or ""),
            timeout_ms=int(mt5_cfg.get("timeout_ms") or 10000),
            reconnect_attempts=int(mt5_cfg.get("reconnect_attempts") or 3),
            reconnect_delay_sec=float(mt5_cfg.get("reconnect_delay_sec") or 2.0),
        )
        self.executor = OrderExecutor(
            self.connection,
            execute=bool(config.get("EXECUTE", False)),
            account_type=str(config.get("account_type", "demo")),
            allow_live=bool(config.get("allow_live", False)),
        )
        risk_cfg = config.get("risk", {})
        self.risk = RiskManager.from_config(risk_cfg)
        loop_cfg = config.get("loop", {})
        self.poll_seconds = int(loop_cfg.get("poll_seconds", 30))
        self.review_weekday = int(loop_cfg.get("review_weekday", 5))
        self.review_hour_utc = int(loop_cfg.get("review_hour_utc", 6))
        self._last_review_date: Optional[str] = None

        state_dir = config.get("paths", {}).get("state_dir", "state")
        strat = config.get("strategy", {})
        self.traders: list[SymbolTrader] = []
        for sc in config.get("symbols", []):
            if not sc.get("enabled", True):
                continue
            sym_cfg = SymbolConfig(
                name=str(sc["name"]),
                state_key=str(sc.get("state_key") or sc["name"].lstrip("#")),
                timeframe=str(sc.get("timeframe", "M30")),
                long_window=int(sc.get("long_window", strat.get("long_window", 240))),
                short_window=int(sc.get("short_window", strat.get("short_window", 48))),
                max_hold_bars=int(sc.get("max_hold_bars", strat.get("max_hold_bars", 16))),
                enabled=True,
            )
            store = StateStore(state_dir, sym_cfg.state_key)
            trader = SymbolTrader(
                symbol_cfg=sym_cfg,
                connection=self.connection,
                executor=self.executor,
                risk=self.risk,
                state_store=store,
                app_config=config,
            )
            self.traders.append(trader)

        if not self.traders:
            raise ValueError("No enabled symbols in config")

        ks = config.get("kill_switch", {})
        self.kill_switch = KillSwitchMonitor(
            connection=self.connection,
            executor=self.executor,
            state_stores=[t.store for t in self.traders],
            symbols=[t.symbol for t in self.traders],
            max_drawdown=float(ks.get("max_drawdown", 0.10)),
            poll_seconds=float(ks.get("poll_seconds", 15)),
        )

    def start_connection(self) -> bool:
        self.connection.start()
        return self.connection.connect()

    def stop(self) -> None:
        self.kill_switch.stop()
        self.connection.shutdown()

    def run_once_all(self) -> list[dict[str, Any]]:
        results = []
        for t in self.traders:
            decision = t.evaluate()
            if decision is None:
                results.append({"symbol": t.symbol, "error": "no decision"})
                continue
            results.append(t.maybe_trade(decision))
        return results

    def poll_bars(self) -> list[dict[str, Any]]:
        if not self.connection.ensure():
            logger.error("MT5 reconnect failed; skipping poll cycle")
            return []
        events = []
        for t in self.traders:
            try:
                if t.store.is_locked():
                    continue
                ev = t.on_new_bar()
                if ev:
                    events.append(ev)
            except Exception:
                logger.exception("Error on_new_bar for %s", t.symbol)
        return events

    def should_review(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if now.weekday() != self.review_weekday:
            return False
        if now.hour < self.review_hour_utc:
            return False
        day_key = now.strftime("%Y-%m-%d")
        if self._last_review_date == day_key:
            return False
        return True

    def review_subloop(self) -> list[dict[str, Any]]:
        """Weekend review: if Sharpe/IC degraded, re-optimize and apply if validated."""
        opt_cfg = self.config.get("optimizer", {})
        sharpe_trig = float(opt_cfg.get("sharpe_degrade_trigger", 0.20))
        ic_trig = float(opt_cfg.get("ic_degrade_trigger", 0.20))
        months = int(self.config.get("validator", {}).get("lookback_months", 6))

        outcomes = []
        for t in self.traders:
            if t.store.is_locked():
                outcomes.append({"symbol": t.symbol, "ok": False, "error": "locked"})
                continue
            logger.info("Review sub-loop starting for %s", t.symbol)
            try:
                degraded = t.metrics_degraded(sharpe_trigger=sharpe_trig, ic_trigger=ic_trig)
                if degraded:
                    logger.info("%s metrics degraded → optimizing", t.symbol)
                    out = t.optimize(months=months)
                else:
                    logger.info("%s metrics stable → skip optimize", t.symbol)
                    out = {"ok": True, "skipped": True, "reason": "metrics_stable"}
                out["symbol"] = t.symbol
                outcomes.append(out)
            except Exception as exc:
                logger.exception("Review failed for %s", t.symbol)
                outcomes.append({"symbol": t.symbol, "ok": False, "error": str(exc)})

        self._last_review_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return outcomes

    def run_forever(self) -> None:
        if not self.start_connection():
            raise RuntimeError("Failed to connect to MT5")
        self.kill_switch.start()
        logger.info(
            "LoopEngine started: symbols=%s EXECUTE=%s poll=%ss",
            [t.symbol for t in self.traders],
            self.config.get("EXECUTE"),
            self.poll_seconds,
        )
        try:
            while True:
                self.poll_bars()
                if self.should_review():
                    self.review_subloop()
                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            logger.info("LoopEngine interrupted by user")
        finally:
            self.stop()
