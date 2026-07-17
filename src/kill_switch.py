"""Account drawdown kill-switch monitor (daemon thread)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class KillSwitchMonitor:
    """
    Poll account equity; if drawdown from peak >= max_drawdown, flatten and lock.
    Unlock is manual only (STATE.locked = false).
    """

    def __init__(
        self,
        connection: Any,
        executor: Any,
        state_stores: list[Any],
        symbols: list[str],
        max_drawdown: float = 0.10,
        poll_seconds: float = 15.0,
        on_trigger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.connection = connection
        self.executor = executor
        self.state_stores = state_stores
        self.symbols = symbols
        self.max_drawdown = float(max_drawdown)
        self.poll_seconds = float(poll_seconds)
        self.on_trigger = on_trigger
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._triggered = False
        self._equity_peak: Optional[float] = None

    @property
    def triggered(self) -> bool:
        return self._triggered

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="KillSwitchMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "KillSwitchMonitor started (max_dd=%.2f%% poll=%ss)",
            self.max_drawdown * 100,
            self.poll_seconds,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.poll_seconds + 2)
        self._thread = None

    def _any_locked(self) -> bool:
        return any(s.is_locked() for s in self.state_stores)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("KillSwitchMonitor tick failed")
            self._stop.wait(self.poll_seconds)

    def _tick(self) -> None:
        if self._triggered or self._any_locked():
            self._triggered = True
            return

        if not self.connection.ensure():
            return

        info = self.connection.account_info()
        if info is None:
            return

        equity = float(info.equity)
        margin = float(getattr(info, "margin", 0.0) or 0.0)

        # Seed peak from STATE if present
        if self._equity_peak is None:
            peaks = []
            for store in self.state_stores:
                st = store.read_state()
                ep = st.get("equity_peak")
                if ep is not None:
                    try:
                        peaks.append(float(ep))
                    except (TypeError, ValueError):
                        pass
            self._equity_peak = max(peaks) if peaks else equity

        if equity > self._equity_peak:
            self._equity_peak = equity

        for store in self.state_stores:
            store.update_state(equity=equity, margin=margin, equity_peak=self._equity_peak)

        if self._equity_peak <= 0:
            return

        dd = (self._equity_peak - equity) / self._equity_peak
        if dd < self.max_drawdown:
            return

        reason = (
            f"equity drawdown {dd:.2%} >= {self.max_drawdown:.2%} "
            f"(equity={equity:.2f} peak={self._equity_peak:.2f})"
        )
        logger.critical("KILL SWITCH: %s", reason)
        self._fire(reason)

    def _fire(self, reason: str) -> None:
        self._triggered = True
        # Flatten all symbols (bypass dry-run for safety when EXECUTE=true;
        # close_all still respects can_execute for order_send)
        for symbol in self.symbols:
            try:
                result = self.executor.close_all(symbol)
                logger.critical("Flatten %s → %s", symbol, getattr(result, "message", result))
            except Exception:
                logger.exception("Failed to flatten %s", symbol)

        for store in self.state_stores:
            store.set_locked(True, reason=reason)

        if self.on_trigger:
            try:
                self.on_trigger(reason)
            except Exception:
                logger.exception("on_trigger callback failed")
