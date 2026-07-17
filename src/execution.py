"""Order execution and target-position reconciliation with safety guards."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import MetaTrader5 as mt5

from .connection import MT5Connection
from .strategy import Signal

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    symbol: str
    side: Signal
    volume: float
    price: float
    comment: str = "lr_loop"
    deviation: int = 20
    magic: int = 260717


@dataclass
class OrderResult:
    ok: bool
    retcode: Optional[int] = None
    order: Optional[int] = None
    deal: Optional[int] = None
    message: str = ""
    dry_run: bool = False
    request: Optional[dict] = None


@dataclass
class ReconcileResult:
    ok: bool
    action: str
    message: str = ""
    dry_run: bool = False
    orders: list[OrderResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "message": self.message,
            "dry_run": self.dry_run,
            "orders": [
                {
                    "ok": o.ok,
                    "retcode": o.retcode,
                    "order": o.order,
                    "deal": o.deal,
                    "message": o.message,
                    "dry_run": o.dry_run,
                    "request": o.request,
                }
                for o in self.orders
            ],
        }


class OrderExecutor:
    """Execute only the delta between actual and desired strategy positions."""

    def __init__(
        self,
        connection: MT5Connection,
        execute: bool = False,
        account_type: str = "demo",
        allow_live: bool = False,
    ) -> None:
        self.connection = connection
        self.execute = bool(execute)
        self.account_type = (account_type or "demo").lower()
        self.allow_live = bool(allow_live)

    def can_execute(self) -> tuple[bool, str]:
        if not self.execute:
            return False, "EXECUTE=false"
        if self.account_type == "live" and not self.allow_live:
            return False, "live trading blocked (allow_live=false)"
        if self.account_type not in ("demo", "live"):
            return False, f"unknown account_type={self.account_type}"
        info = self.connection.account_info()
        if info is not None:
            trade_mode = getattr(info, "trade_mode", None)
            if trade_mode == 2 and (self.account_type != "live" or not self.allow_live):
                return False, "MT5 account is REAL but config forbids live"
        return True, "ok"

    def managed_positions(self, symbol: str, magic: int = 260717) -> list[Any]:
        if not self.connection.ensure():
            return []
        positions = mt5.positions_get(symbol=symbol) or []
        if magic == 0:
            return list(positions)
        return [p for p in positions if int(getattr(p, "magic", 0) or 0) == magic]

    def managed_pending(self, symbol: str, magic: int = 260717) -> list[Any]:
        if not self.connection.ensure():
            return []
        orders = mt5.orders_get(symbol=symbol) or []
        if magic == 0:
            return list(orders)
        return [o for o in orders if int(getattr(o, "magic", 0) or 0) == magic]

    @staticmethod
    def _pending_side(order: Any) -> Optional[Signal]:
        order_type = int(getattr(order, "type", -1))
        buy_types = {
            int(mt5.ORDER_TYPE_BUY_LIMIT),
            int(getattr(mt5, "ORDER_TYPE_BUY_STOP", -1)),
            int(getattr(mt5, "ORDER_TYPE_BUY", -1)),
        }
        sell_types = {
            int(mt5.ORDER_TYPE_SELL_LIMIT),
            int(getattr(mt5, "ORDER_TYPE_SELL_STOP", -1)),
            int(getattr(mt5, "ORDER_TYPE_SELL", -1)),
        }
        if order_type in buy_types:
            return Signal.LONG
        if order_type in sell_types:
            return Signal.SHORT
        return None

    @staticmethod
    def _pending_volume(order: Any) -> float:
        for attr in ("volume_current", "volume_initial", "volume"):
            value = getattr(order, attr, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _limit_price(self, symbol: str, side: Signal) -> Optional[float]:
        tick = mt5.symbol_info_tick(symbol)
        info = self.connection.symbol_info(symbol)
        if tick is None or info is None:
            return None
        digits = int(getattr(info, "digits", 2) or 2)
        point = float(getattr(info, "point", 0.01) or 0.01)
        bid = float(tick.bid)
        ask = float(tick.ask)
        price = bid if side == Signal.LONG else ask
        if side == Signal.LONG and price >= ask:
            price = ask - point
        if side == Signal.SHORT and price <= bid:
            price = bid + point
        return float(round(price, digits))

    def place_limit(self, req: OrderRequest) -> OrderResult:
        allowed, reason = self.can_execute()
        side = req.side
        if side == Signal.FLAT or req.volume <= 0:
            return OrderResult(ok=False, message="flat or zero volume")
        if not self.connection.ensure():
            return OrderResult(ok=False, message="MT5 not connected")

        info = self.connection.symbol_info(req.symbol)
        if info is None:
            return OrderResult(ok=False, message=f"symbol unavailable: {req.symbol}")
        price = req.price if req.price > 0 else self._limit_price(req.symbol, side)
        if price is None or price <= 0:
            return OrderResult(ok=False, message="could not determine limit price")

        order_type = mt5.ORDER_TYPE_BUY_LIMIT if side == Signal.LONG else mt5.ORDER_TYPE_SELL_LIMIT
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": req.symbol,
            "volume": float(req.volume),
            "type": order_type,
            "price": float(price),
            "deviation": int(req.deviation),
            "magic": int(req.magic),
            "comment": req.comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(info),
        }

        if not allowed:
            logger.info("DRY-RUN order skipped (%s): %s", reason, request)
            return OrderResult(ok=True, message=f"dry-run: {reason}", dry_run=True, request=request)
        return self._send(request, "order")

    def cancel_pending(self, symbol: str, magic: int = 260717) -> int:
        """Cancel pending orders only after the same account safety gate as entry."""
        allowed, reason = self.can_execute()
        if not allowed:
            logger.info("Pending cancel skipped (%s) symbol=%s magic=%s", reason, symbol, magic)
            return 0
        if not self.connection.ensure():
            return 0
        orders = mt5.orders_get(symbol=symbol) or []
        cancelled = 0
        for order in orders:
            if magic != 0 and int(getattr(order, "magic", 0) or 0) != magic:
                continue
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order.ticket),
                "symbol": symbol,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                cancelled += 1
            else:
                logger.warning("Failed to cancel pending order=%s", order.ticket)
        return cancelled

    def close_position_market(self, position: Any, magic: int = 260717) -> OrderResult:
        """Close one exact MT5 position ticket using a market deal."""
        allowed, reason = self.can_execute()
        symbol = str(position.symbol)
        if not self.connection.ensure():
            return OrderResult(ok=False, message="not connected")
        tick = mt5.symbol_info_tick(symbol)
        info = self.connection.symbol_info(symbol)
        if tick is None or info is None:
            return OrderResult(ok=False, message="missing tick/info")

        is_buy = position.type == mt5.POSITION_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(position.volume),
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": int(position.ticket),
            "price": float(tick.bid if is_buy else tick.ask),
            "deviation": 50,
            "magic": int(magic),
            "comment": "lr_flat",
            "type_filling": self._filling_mode(info),
        }
        if not allowed:
            logger.info("DRY-RUN close skipped (%s): %s", reason, request)
            return OrderResult(ok=True, message=f"dry-run close: {reason}", dry_run=True, request=request)
        return self._send(request, "close")

    def close_managed_positions(self, symbol: str, magic: int = 260717) -> list[OrderResult]:
        return [self.close_position_market(p, magic=magic) for p in self.managed_positions(symbol, magic)]

    def reconcile_target(
        self,
        *,
        symbol: str,
        side: Signal,
        volume: float,
        volume_step: float = 0.01,
        comment: str = "lr_loop",
        magic: int = 260717,
    ) -> ReconcileResult:
        """Move managed positions toward one desired side/volume without stacking.

        Reversals and resizes use close-then-open. Matching pending limit
        orders are left alone (`await_fill`) so unfilled entries are not
        cancelled and re-placed every bar. Partial-fill retries remain a
        later MT5 integration step.
        """
        positions = self.managed_positions(symbol, magic=magic)
        pending = self.managed_pending(symbol, magic=magic)
        current_sides = {
            Signal.LONG if p.type == mt5.POSITION_TYPE_BUY else Signal.SHORT for p in positions
        }
        current_volume = sum(float(p.volume) for p in positions)
        tolerance = max(float(volume_step), 1e-9) / 2.0

        if side == Signal.FLAT or volume <= 0:
            self.cancel_pending(symbol, magic=magic)
            closes = self.close_managed_positions(symbol, magic=magic)
            return ReconcileResult(
                ok=all(r.ok for r in closes),
                action="flatten" if positions or pending else "hold_flat",
                message="target flat",
                dry_run=any(r.dry_run for r in closes),
                orders=closes,
            )

        matches = (
            len(current_sides) == 1
            and side in current_sides
            and abs(current_volume - float(volume)) <= tolerance
        )
        if matches:
            # Position already filled — drop any leftover working orders.
            self.cancel_pending(symbol, magic=magic)
            return ReconcileResult(ok=True, action="hold", message="target already satisfied")

        # Flat (or wrong size) but a matching limit is already working → wait.
        if not positions and pending:
            pending_sides = {self._pending_side(o) for o in pending}
            pending_sides.discard(None)
            pending_volume = sum(self._pending_volume(o) for o in pending)
            pending_matches = (
                len(pending_sides) == 1
                and side in pending_sides
                and abs(pending_volume - float(volume)) <= tolerance
            )
            if pending_matches:
                logger.info(
                    "Awaiting fill symbol=%s side=%s volume=%s pending=%s",
                    symbol,
                    side.name,
                    volume,
                    len(pending),
                )
                return ReconcileResult(
                    ok=True,
                    action="await_fill",
                    message="matching pending limit already working",
                )

        self.cancel_pending(symbol, magic=magic)
        closes = self.close_managed_positions(symbol, magic=magic)
        if any(not r.ok for r in closes):
            return ReconcileResult(
                ok=False,
                action="close_failed",
                message="existing position close failed; new entry blocked",
                dry_run=any(r.dry_run for r in closes),
                orders=closes,
            )

        entry = self.place_limit(
            OrderRequest(
                symbol=symbol,
                side=side,
                volume=float(volume),
                price=0.0,
                comment=comment,
                magic=magic,
            )
        )
        if entry.ok:
            entry.message = (
                entry.message
                if entry.dry_run
                else f"limit entry submitted; awaiting fill ({entry.message})"
            )
        return ReconcileResult(
            ok=entry.ok,
            action="open" if not positions else "reverse_or_resize",
            message=entry.message,
            dry_run=entry.dry_run or any(r.dry_run for r in closes),
            orders=closes + [entry],
        )

    def close_position_limit(self, symbol: str, magic: int = 260717) -> Optional[OrderResult]:
        """Backward-compatible wrapper; now closes an exact position at market."""
        positions = self.managed_positions(symbol, magic=magic)
        if not positions:
            return OrderResult(ok=True, message="no position")
        return self.close_position_market(positions[0], magic=magic)

    def close_all(self, symbol: str, magic: int = 260717) -> OrderResult:
        """Kill-switch flatten for all positions on a symbol."""
        self.cancel_pending(symbol, magic=0)
        positions = self.managed_positions(symbol, magic=0)
        if not positions:
            return OrderResult(ok=True, message="no position")
        results = [self.close_position_market(p, magic=magic) for p in positions]
        failures = [r for r in results if not r.ok]
        return failures[-1] if failures else results[-1]

    @staticmethod
    def _send(request: dict[str, Any], operation: str) -> OrderResult:
        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            logger.error("%s order_send returned None: %s", operation, err)
            return OrderResult(ok=False, message=str(err), request=request)
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        message = getattr(result, "comment", "") or str(result.retcode)
        if not ok:
            logger.warning("%s rejected retcode=%s comment=%s", operation, result.retcode, message)
        return OrderResult(
            ok=ok,
            retcode=int(result.retcode),
            order=int(result.order) if getattr(result, "order", 0) else None,
            deal=int(result.deal) if getattr(result, "deal", 0) else None,
            message=message,
            request=request,
        )

    @staticmethod
    def _filling_mode(info: Any) -> int:
        filling = getattr(info, "filling_mode", None)
        try:
            mode = int(filling) if filling is not None else 0
        except (TypeError, ValueError):
            mode = 0
        if mode & 2:
            return mt5.ORDER_FILLING_IOC
        if mode & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN
