"""MetaTrader5 connection wrapper with a dedicated API worker thread."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _MT5Job:
    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None


class MT5Connection:
    """Manage initialize / login / shutdown on a single MT5 worker thread.

    All MetaTrader5 API calls must go through :meth:`invoke` (or helpers that
    use it). The kill-switch thread and the trading loop therefore cannot
    interleave ``shutdown``/``initialize`` with ``order_send``.
    """

    def __init__(
        self,
        login: int = 0,
        password: str = "",
        server: str = "FxPro-Demo",
        path: str = "",
        timeout_ms: int = 10000,
        reconnect_attempts: int = 3,
        reconnect_delay_sec: float = 2.0,
        invoke_timeout_sec: float = 120.0,
    ) -> None:
        self.login = int(login) if login else 0
        self.password = password or ""
        self.server = server or ""
        self.path = path or ""
        self.timeout_ms = int(timeout_ms)
        self.reconnect_attempts = max(1, int(reconnect_attempts))
        self.reconnect_delay_sec = float(reconnect_delay_sec)
        self.invoke_timeout_sec = float(invoke_timeout_sec)
        self._connected = False

        self._jobs: queue.Queue[Optional[_MT5Job]] = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._life = threading.Lock()

    @property
    def connected(self) -> bool:
        try:
            return bool(self.invoke(self._connected_unlocked))
        except Exception:
            return False

    def start(self) -> None:
        """Start the dedicated MT5 worker thread (idempotent)."""
        with self._life:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="MT5Worker",
                daemon=True,
            )
            self._worker.start()
            logger.info("MT5 worker thread started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker after draining; does not call mt5.shutdown()."""
        with self._life:
            worker = self._worker
            if worker is None:
                return
            self._stop.set()
            self._jobs.put(None)
            worker.join(timeout=timeout)
            if worker.is_alive():
                logger.warning("MT5 worker did not stop within %.1fs", timeout)
            else:
                logger.info("MT5 worker thread stopped")
            self._worker = None

    def invoke(self, fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        """Run ``fn(*args, **kwargs)`` on the MT5 worker thread and return its result."""
        self.start()
        worker = self._worker
        if worker is not None and threading.current_thread() is worker:
            return fn(*args, **kwargs)

        job = _MT5Job(fn=fn, args=args, kwargs=kwargs)
        self._jobs.put(job)
        if not job.done.wait(timeout=self.invoke_timeout_sec):
            raise TimeoutError(
                f"MT5 invoke timed out after {self.invoke_timeout_sec:.1f}s "
                f"({getattr(fn, '__name__', repr(fn))})"
            )
        if job.error is not None:
            raise job.error
        return job.result

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                if self._stop.is_set():
                    break
                continue
            try:
                job.result = job.fn(*job.args, **job.kwargs)
            except BaseException as exc:  # noqa: BLE001 — surface to caller
                job.error = exc
            finally:
                job.done.set()

    def _connected_unlocked(self) -> bool:
        return self._connected and self._terminal_alive_unlocked()

    def _terminal_alive_unlocked(self) -> bool:
        try:
            info = mt5.terminal_info()
            return info is not None
        except Exception:
            return False

    def _connect_unlocked(self) -> bool:
        """Initialize terminal. Must run on the worker thread."""
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
            logger.error(
                "MT5 account_info unavailable after initialize: %s", mt5.last_error()
            )
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

    def _ensure_unlocked(self) -> bool:
        """Reconnect if needed. Must run on the worker thread."""
        if self._connected and self._terminal_alive_unlocked():
            return True

        logger.warning("MT5 connection lost or not initialized; reconnecting")
        self._connected = False
        for attempt in range(1, self.reconnect_attempts + 1):
            if self._connect_unlocked():
                return True
            logger.warning(
                "MT5 reconnect attempt %d/%d failed",
                attempt,
                self.reconnect_attempts,
            )
            if attempt < self.reconnect_attempts:
                # Block the worker intentionally: no other MT5 op mid-reconnect.
                time.sleep(self.reconnect_delay_sec * attempt)
        return False

    def _shutdown_unlocked(self) -> None:
        if self._connected or self._terminal_alive_unlocked():
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 shutdown complete")

    def connect(self) -> bool:
        """Initialize the terminal and optionally log in."""
        return bool(self.invoke(self._connect_unlocked))

    def ensure(self) -> bool:
        """Return True if connected; otherwise reconnect with retries."""
        return bool(self.invoke(self._ensure_unlocked))

    def account_info(self) -> Optional[Any]:
        def _op() -> Optional[Any]:
            if not self._ensure_unlocked():
                return None
            return mt5.account_info()

        return self.invoke(_op)

    def symbol_info(self, symbol: str) -> Optional[Any]:
        def _op() -> Optional[Any]:
            if not self._ensure_unlocked():
                return None
            info = mt5.symbol_info(symbol)
            if info is None:
                if not mt5.symbol_select(symbol, True):
                    logger.warning(
                        "symbol_select failed for %s: %s", symbol, mt5.last_error()
                    )
                    return None
                info = mt5.symbol_info(symbol)
            return info

        return self.invoke(_op)

    def symbol_info_tick(self, symbol: str) -> Optional[Any]:
        def _op() -> Optional[Any]:
            if not self._ensure_unlocked():
                return None
            return mt5.symbol_info_tick(symbol)

        return self.invoke(_op)

    def positions_get(self, *, symbol: Optional[str] = None) -> Optional[tuple[Any, ...]]:
        def _op() -> Optional[tuple[Any, ...]]:
            if not self._ensure_unlocked():
                return None
            if symbol is None:
                return mt5.positions_get()
            return mt5.positions_get(symbol=symbol)

        return self.invoke(_op)

    def orders_get(self, *, symbol: Optional[str] = None) -> Optional[tuple[Any, ...]]:
        def _op() -> Optional[tuple[Any, ...]]:
            if not self._ensure_unlocked():
                return None
            if symbol is None:
                return mt5.orders_get()
            return mt5.orders_get(symbol=symbol)

        return self.invoke(_op)

    def order_send(self, request: dict[str, Any]) -> Any:
        def _op() -> Any:
            if not self._ensure_unlocked():
                return None
            return mt5.order_send(request)

        return self.invoke(_op)

    def last_error(self) -> Any:
        return self.invoke(mt5.last_error)

    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> Any:
        def _op() -> Any:
            if not self._ensure_unlocked():
                return None
            return mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)

        return self.invoke(_op)

    def copy_rates_range(
        self, symbol: str, timeframe: int, date_from: Any, date_to: Any
    ) -> Any:
        def _op() -> Any:
            if not self._ensure_unlocked():
                return None
            return mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

        return self.invoke(_op)

    def order_calc_margin(
        self, order_type: int, symbol: str, volume: float, price: float
    ) -> Optional[float]:
        def _op() -> Optional[float]:
            if not self._ensure_unlocked():
                return None
            margin = mt5.order_calc_margin(order_type, symbol, volume, price)
            if margin is None:
                return None
            value = float(margin)
            return value if value > 0 else None

        return self.invoke(_op)

    def shutdown(self) -> None:
        """Shut down the MT5 session on the worker, then stop the worker."""
        try:
            if self._worker is not None and self._worker.is_alive():
                self.invoke(self._shutdown_unlocked)
            else:
                # Worker already gone; best-effort local shutdown.
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                self._connected = False
        finally:
            self.stop()

    def __enter__(self) -> "MT5Connection":
        self.start()
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
