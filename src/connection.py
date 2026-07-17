"""MetaTrader5 connection wrapper for FxPro terminals."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class MT5Connection:
    """Manage initialize / login / shutdown for a single MT5 terminal."""

    def __init__(
        self,
        login: int = 0,
        password: str = "",
        server: str = "FxPro-Demo",
        path: str = "",
        timeout_ms: int = 10000,
        reconnect_attempts: int = 3,
        reconnect_delay_sec: float = 2.0,
    ) -> None:
        self.login = int(login) if login else 0
        self.password = password or ""
        self.server = server or ""
        self.path = path or ""
        self.timeout_ms = int(timeout_ms)
        self.reconnect_attempts = max(1, int(reconnect_attempts))
        self.reconnect_delay_sec = float(reconnect_delay_sec)
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and self._terminal_alive()

    def _terminal_alive(self) -> bool:
        try:
            info = mt5.terminal_info()
            return info is not None
        except Exception:
            return False

    def connect(self) -> bool:
        """Initialize the terminal and optionally log in."""
        # Clean slate if a prior session is half-open
        try:
            mt5.shutdown()
        except Exception:
            pass

        kwargs: dict[str, Any] = {"timeout": self.timeout_ms}
        if self.path:
            kwargs["path"] = self.path
        if self.login:
            kwargs["login"] = self.login
            kwargs["password"] = self.password
            kwargs["server"] = self.server

        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            logger.error("MT5 initialize failed: %s", err)
            self._connected = False
            return False

        info = mt5.account_info()
        if info is None:
            logger.error("MT5 account_info unavailable after initialize: %s", mt5.last_error())
            mt5.shutdown()
            self._connected = False
            return False

        self._connected = True
        logger.info(
            "MT5 connected: login=%s server=%s balance=%.2f",
            info.login,
            info.server,
            info.balance,
        )
        return True

    def ensure(self) -> bool:
        """Return True if connected; otherwise reconnect with retries."""
        if self._connected and self._terminal_alive():
            return True

        logger.warning("MT5 connection lost or not initialized; reconnecting")
        self._connected = False
        for attempt in range(1, self.reconnect_attempts + 1):
            if self.connect():
                return True
            logger.warning(
                "MT5 reconnect attempt %d/%d failed",
                attempt,
                self.reconnect_attempts,
            )
            if attempt < self.reconnect_attempts:
                time.sleep(self.reconnect_delay_sec * attempt)
        return False

    def account_info(self) -> Optional[Any]:
        if not self.ensure():
            return None
        return mt5.account_info()

    def symbol_info(self, symbol: str) -> Optional[Any]:
        if not self.ensure():
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            if not mt5.symbol_select(symbol, True):
                logger.warning("symbol_select failed for %s: %s", symbol, mt5.last_error())
                return None
            info = mt5.symbol_info(symbol)
        return info

    def shutdown(self) -> None:
        if self._connected or self._terminal_alive():
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 shutdown complete")

    def __enter__(self) -> "MT5Connection":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
