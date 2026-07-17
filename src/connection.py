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


class MT5InvokeTimeout(TimeoutError):
    """Caller waited too long; the worker job may still run or was abandoned.

    Treat order_send timeouts as *result unknown*, never as a clean failure
    that is safe to auto-retry.
    """

    def __init__(
        self,
        message: str,
        *,
        fn_name: str,
        abandoned: bool = True,
        generation: int = 0,
    ) -> None:
        super().__init__(message)
        self.fn_name = fn_name
        self.abandoned = abandoned
        self.generation = generation


@dataclass
class _MT5Job:
    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None
    abandoned: bool = False
    generation: int = 0


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
        self._generation = 0

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
        """Run ``fn(*args, **kwargs)`` on the MT5 worker thread and return its result.

        On timeout the specific job is marked ``abandoned``. If it has not started,
        the worker skips it. If ``order_send`` is already running it cannot be
        cancelled — callers must treat that as result-unknown and reconcile before
        retrying (never auto-resend the same order).
        """
        self.start()
        worker = self._worker
        if worker is not None and threading.current_thread() is worker:
            return fn(*args, **kwargs)

        generation = self._generation
        job = _MT5Job(fn=fn, args=args, kwargs=kwargs, generation=generation)
        self._jobs.put(job)
        fn_name = getattr(fn, "__name__", repr(fn))
        if not job.done.wait(timeout=self.invoke_timeout_sec):
            job.abandoned = True
            raise MT5InvokeTimeout(
                f"MT5 invoke timed out after {self.invoke_timeout_sec:.1f}s "
                f"({fn_name}); job abandoned — treat order results as unknown "
                f"until reconciled",
                fn_name=str(fn_name),
                abandoned=True,
                generation=generation,
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
            fn_name = getattr(job.fn, "__name__", repr(job.fn))
            # Abandoned before start: do not execute (avoids late order_send after
            # the caller already timed out). Already-running jobs are not cancelled.
            if job.abandoned:
                logger.warning(
                    "Skipping abandoned MT5 job fn=%s gen=%s",
                    fn_name,
                    job.generation,
                )
                job.done.set()
                continue
            try:
                job.result = job.fn(*job.args, **job.kwargs)
            except BaseException as exc:  # noqa: BLE001 — surface to caller
                job.error = exc
            finally:
                if job.abandoned:
                    logger.warning(
                        "MT5 job finished after caller timeout fn=%s error=%s",
                        fn_name,
                        job.error,
                    )
                job.done.set()

    def bump_generation(self) -> int:
        """Invalidate not-yet-started queued jobs (optional manual fence)."""
        self._generation += 1
        return self._generation

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

    def order_check(self, request: dict[str, Any]) -> Any:
        """Validate a trade request (margin, volume, stops, permissions, market).

        Does not place an order. Callers must treat a failed / missing result as
        fail-closed and must not call :meth:`order_send`.
        """

        def _op() -> Any:
            if not self._ensure_unlocked():
                return None
            return mt5.order_check(request)

        return self.invoke(_op)

    def order_send(self, request: dict[str, Any]) -> Any:
        def _op() -> Any:
            if not self._ensure_unlocked():
                return None
            return mt5.order_send(request)

        return self.invoke(_op)

    def history_deals_get(
        self,
        date_from: Any = None,
        date_to: Any = None,
        *,
        group: Optional[str] = None,
        ticket: Optional[int] = None,
        position: Optional[int] = None,
        order: Optional[int] = None,
    ) -> Optional[tuple[Any, ...]]:
        """Fetch deal history.

        Prefer ``ticket`` / ``order`` / ``position`` lookups when available.
        Date-range form remains for intent reconciliation scans.
        """

        def _op() -> Optional[tuple[Any, ...]]:
            if not self._ensure_unlocked():
                return None
            if ticket is not None:
                return mt5.history_deals_get(ticket=int(ticket))
            if order is not None:
                return mt5.history_deals_get(order=int(order))
            if position is not None:
                return mt5.history_deals_get(position=int(position))
            if date_from is None or date_to is None:
                return None
            if group is None:
                return mt5.history_deals_get(date_from, date_to)
            return mt5.history_deals_get(date_from, date_to, group=group)

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

    def order_calc_profit(
        self,
        order_type: int,
        symbol: str,
        volume: float,
        price_open: float,
        price_close: float,
    ) -> Optional[float]:
        """Deposit-currency PnL for a hypothetical fill; None if MT5 cannot compute."""

        def _op() -> Optional[float]:
            if not self._ensure_unlocked():
                return None
            profit = mt5.order_calc_profit(
                order_type, symbol, volume, price_open, price_close
            )
            if profit is None:
                return None
            return float(profit)

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
