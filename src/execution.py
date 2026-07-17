"""Limit-order execution with EXECUTE / account-type safety guards."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


class OrderExecutor:
    """
    Prefer BUY_LIMIT / SELL_LIMIT to reduce slippage.
    Orders are only sent when EXECUTE is true and account_type rules pass.
    """

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
        # Soft check against actual MT5 account trade mode if available
        info = self.connection.account_info()
        if info is not None:
            trade_mode = getattr(info, "trade_mode", None)
            # ACCOUNT_TRADE_MODE_DEMO = 0, CONTEST = 1, REAL = 2
            if trade_mode == 2 and (self.account_type != "live" or not self.allow_live):
                return False, "MT5 account is REAL but config forbids live"
        return True, "ok"

    def _limit_price(self, symbol: str, side: Signal) -> Optional[float]:
        tick = mt5.symbol_info_tick(symbol)
        info = self.connection.symbol_info(symbol)
        if tick is None or info is None:
            return None
        digits = int(getattr(info, "digits", 2) or 2)
        point = float(getattr(info, "point", 0.01) or 0.01)
        # Place limit slightly inside spread toward mid to improve fill odds
        bid = float(tick.bid)
        ask = float(tick.ask)
        if side == Signal.LONG:
            # Buy limit below ask
            price = bid
        else:
            # Sell limit above bid
            price = ask
        # Nudge 1 point toward market if needed for validity
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
        filling = self._filling_mode(info)

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
            "type_filling": filling,
        }

        if not allowed:
            logger.info("DRY-RUN order skipped (%s): %s", reason, request)
            return OrderResult(
                ok=True,
                message=f"dry-run: {reason}",
                dry_run=True,
                request=request,
            )

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            logger.error("order_send returned None: %s", err)
            return OrderResult(ok=False, message=str(err), request=request)

        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        msg = getattr(result, "comment", "") or str(result.retcode)
        if not ok:
            logger.warning("order rejected retcode=%s comment=%s", result.retcode, msg)
        else:
            logger.info(
                "order placed order=%s deal=%s retcode=%s",
                result.order,
                result.deal,
                result.retcode,
            )
        return OrderResult(
            ok=ok,
            retcode=int(result.retcode),
            order=int(result.order) if result.order else None,
            deal=int(result.deal) if result.deal else None,
            message=msg,
            dry_run=False,
            request=request,
        )

    def cancel_pending(self, symbol: str, magic: int = 260717) -> int:
        if not self.connection.ensure():
            return 0
        orders = mt5.orders_get(symbol=symbol)
        if not orders:
            return 0
        cancelled = 0
        for o in orders:
            if o.magic != magic:
                continue
            req = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": o.ticket,
                "symbol": symbol,
            }
            if not self.execute:
                logger.info("DRY-RUN cancel order %s", o.ticket)
                cancelled += 1
                continue
            r = mt5.order_send(req)
            if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                cancelled += 1
        return cancelled

    def close_position_limit(self, symbol: str, magic: int = 260717) -> Optional[OrderResult]:
        """Flatten by opposite limit if a position exists."""
        if not self.connection.ensure():
            return OrderResult(ok=False, message="not connected")
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return OrderResult(ok=True, message="no position")
        pos = None
        for p in positions:
            if p.magic == magic or magic == 0:
                pos = p
                break
        if pos is None:
            pos = positions[0]

        side = Signal.SHORT if pos.type == mt5.POSITION_TYPE_BUY else Signal.LONG
        return self.place_limit(
            OrderRequest(
                symbol=symbol,
                side=side,
                volume=float(pos.volume),
                price=0.0,
                comment="lr_flat",
                magic=magic,
            )
        )

    def close_all(self, symbol: str, magic: int = 260717) -> OrderResult:
        """
        Kill-switch flatten: cancel pendings, then close positions via market deal.
        When EXECUTE=false, logs dry-run but still cancels/records intent.
        """
        if not self.connection.ensure():
            return OrderResult(ok=False, message="not connected")

        self.cancel_pending(symbol, magic=magic)
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return OrderResult(ok=True, message="no position")

        last: Optional[OrderResult] = None
        for pos in positions:
            # Kill-switch closes every position on the symbol (ignore magic filter)
            tick = mt5.symbol_info_tick(symbol)
            info = self.connection.symbol_info(symbol)
            if tick is None or info is None:
                last = OrderResult(ok=False, message="missing tick/info")
                continue

            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
            price = float(tick.bid if is_buy else tick.ask)
            filling = self._filling_mode(info)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(pos.volume),
                "type": order_type,
                "position": int(pos.ticket),
                "price": price,
                "deviation": 50,
                "magic": int(magic),
                "comment": "kill_flat",
                "type_filling": filling,
            }

            allowed, reason = self.can_execute()
            if not allowed:
                logger.critical("DRY-RUN kill flatten (%s): %s", reason, request)
                last = OrderResult(
                    ok=True,
                    message=f"dry-run kill flatten: {reason}",
                    dry_run=True,
                    request=request,
                )
                continue

            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                last = OrderResult(ok=False, message=str(err), request=request)
                continue
            ok = result.retcode == mt5.TRADE_RETCODE_DONE
            last = OrderResult(
                ok=ok,
                retcode=int(result.retcode),
                order=int(result.order) if result.order else None,
                deal=int(result.deal) if result.deal else None,
                message=getattr(result, "comment", "") or str(result.retcode),
                request=request,
            )
            if ok:
                logger.critical(
                    "Kill flatten closed position=%s volume=%s",
                    pos.ticket,
                    pos.volume,
                )
        return last or OrderResult(ok=True, message="flattened")

    @staticmethod
    def _filling_mode(info: Any) -> int:
        filling = getattr(info, "filling_mode", None)
        # Prefer IOC / RETURN commonly supported; fallback FOK
        # SYMBOL_FILLING_FOK=1, IOC=2, RETURN=4 (bitmask)
        try:
            mode = int(filling) if filling is not None else 0
        except (TypeError, ValueError):
            mode = 0
        if mode & 2:
            return mt5.ORDER_FILLING_IOC
        if mode & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN
