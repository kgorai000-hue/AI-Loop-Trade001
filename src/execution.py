"""Order execution and target-position reconciliation with safety guards."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol

import MetaTrader5 as mt5

from .connection import MT5Connection, MT5InvokeTimeout
from .strategy import Signal

logger = logging.getLogger(__name__)


class IntentStateStore(Protocol):
    def read_state(self) -> dict[str, Any]: ...
    def update_state(self, **kwargs: Any) -> dict[str, Any]: ...


def make_intent_id() -> str:
    """10-hex intent id (fits MT5 comment with a 1-char prefix)."""
    return f"{time.time_ns() % (16**10):010x}"


def intent_comment(intent_id: str) -> str:
    return f"i{intent_id}"[:31]


@dataclass
class OrderRequest:
    symbol: str
    side: Signal
    volume: float
    price: float
    comment: str = "lr_loop"
    deviation: int = 20
    magic: int = 260717
    sl: Optional[float] = None


@dataclass
class OrderResult:
    ok: bool
    retcode: Optional[int] = None
    order: Optional[int] = None
    deal: Optional[int] = None
    message: str = ""
    dry_run: bool = False
    request: Optional[dict] = None
    closed_pnl: Optional[float] = None
    partial: bool = False  # DONE_PARTIAL / incomplete fill — not a finished close
    unknown: bool = False  # invoke timeout — result unknown; do not auto-retry
    intent_id: Optional[str] = None


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
                    "closed_pnl": o.closed_pnl,
                    "partial": o.partial,
                    "unknown": o.unknown,
                    "intent_id": o.intent_id,
                }
                for o in self.orders
            ],
        }


@dataclass
class CancelPendingResult:
    """Outcome of cancelling managed pending orders, including post-check."""

    ok: bool
    message: str = ""
    dry_run: bool = False
    skipped: bool = False
    fetch_failed: bool = False
    attempted: list[int] = field(default_factory=list)
    cancelled: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    remaining: list[int] = field(default_factory=list)
    unknown: bool = False
    intent_id: Optional[str] = None


class OrderExecutor:
    """Execute only the delta between actual and desired strategy positions."""

    def __init__(
        self,
        connection: MT5Connection,
        execute: bool = False,
        account_type: str = "demo",
        allow_live: bool = False,
        *,
        intent_settle_sec: Optional[float] = None,
    ) -> None:
        self.connection = connection
        self.execute = bool(execute)
        self.account_type = (account_type or "demo").lower()
        self.allow_live = bool(allow_live)
        # Wait at least one invoke timeout (+buffer) before treating "not found" as absent.
        default_settle = float(getattr(connection, "invoke_timeout_sec", 120.0) or 120.0) + 60.0
        self.intent_settle_sec = float(
            intent_settle_sec if intent_settle_sec is not None else default_settle
        )
        self._intent_lock = threading.RLock()
        self._intents: dict[str, dict[str, Any]] = {}

    def can_execute(self) -> tuple[bool, str]:
        if not self.execute:
            return False, "EXECUTE=false"
        if self.account_type == "live" and not self.allow_live:
            return False, "live trading blocked (allow_live=false)"
        if self.account_type not in ("demo", "live"):
            return False, f"unknown account_type={self.account_type}"

        info = self.connection.account_info()
        if info is None:
            return False, "account_info unavailable; fail closed"

        trade_mode = getattr(info, "trade_mode", None)
        if trade_mode is None:
            return False, "account trade_mode missing; fail closed"
        try:
            trade_mode_i = int(trade_mode)
        except (TypeError, ValueError):
            return False, f"account trade_mode invalid={trade_mode!r}; fail closed"

        # MT5: DEMO=0, CONTEST=1, REAL=2
        demo_mode = int(getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0))
        contest_mode = int(getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 1))
        real_mode = int(getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2))
        known = {demo_mode, contest_mode, real_mode}
        if trade_mode_i not in known:
            return False, f"account trade_mode unknown={trade_mode_i}; fail closed"

        if trade_mode_i == real_mode and (
            self.account_type != "live" or not self.allow_live
        ):
            return False, "MT5 account is REAL but config forbids live"
        return True, "ok"

    # --- order intent tracking (timeout = unknown, not failure) ---

    def _sync_intents_from_store(self, state_store: Optional[IntentStateStore]) -> None:
        if state_store is None:
            return
        raw = state_store.read_state().get("order_intents") or []
        if not isinstance(raw, list):
            return
        with self._intent_lock:
            for item in raw:
                if not isinstance(item, dict):
                    continue
                intent_id = str(item.get("id") or "")
                if not intent_id:
                    continue
                self._intents[intent_id] = dict(item)

    def _persist_intents(self, state_store: Optional[IntentStateStore], symbol: str) -> None:
        if state_store is None:
            return
        with self._intent_lock:
            # Keep other symbols' intents from disk, replace this symbol's set.
            existing = state_store.read_state().get("order_intents") or []
            if not isinstance(existing, list):
                existing = []
            kept = [
                item
                for item in existing
                if isinstance(item, dict) and str(item.get("symbol") or "") != symbol
            ]
            ours = [
                dict(v)
                for v in self._intents.values()
                if str(v.get("symbol") or "") == symbol
                and str(v.get("status") or "") == "unknown"
            ]
            state_store.update_state(order_intents=kept + ours)

    def _record_unknown_intent(
        self,
        *,
        intent_id: str,
        symbol: str,
        kind: str,
        comment: str,
        magic: int,
        state_store: Optional[IntentStateStore],
        position_ticket: Optional[int] = None,
    ) -> None:
        intent = {
            "id": intent_id,
            "symbol": symbol,
            "kind": kind,
            "status": "unknown",
            "comment": comment[:31],
            "magic": int(magic),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "position_ticket": position_ticket,
        }
        with self._intent_lock:
            self._intents[intent_id] = intent
        self._persist_intents(state_store, symbol)
        logger.error(
            "Order result UNKNOWN after timeout intent=%s kind=%s symbol=%s — "
            "reorder blocked until reconciled",
            intent_id,
            kind,
            symbol,
        )

    def _clear_intent(
        self,
        intent_id: str,
        *,
        symbol: str,
        state_store: Optional[IntentStateStore],
        reason: str,
    ) -> None:
        with self._intent_lock:
            self._intents.pop(intent_id, None)
        self._persist_intents(state_store, symbol)
        logger.info("Order intent cleared id=%s reason=%s", intent_id, reason)

    def unknown_intents(
        self, symbol: str, state_store: Optional[IntentStateStore] = None
    ) -> list[dict[str, Any]]:
        self._sync_intents_from_store(state_store)
        with self._intent_lock:
            return [
                dict(v)
                for v in self._intents.values()
                if str(v.get("symbol") or "") == symbol
                and str(v.get("status") or "") == "unknown"
            ]

    @staticmethod
    def _comment_has_intent(comment: Any, intent_id: str) -> bool:
        text = str(comment or "")
        return bool(intent_id) and intent_id in text

    def _intent_age_sec(self, intent: dict[str, Any]) -> float:
        created = intent.get("created_at")
        if not created:
            return 0.0
        try:
            ts = datetime.fromisoformat(str(created))
        except ValueError:
            return 0.0
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())

    def _find_intent_evidence(
        self, intent: dict[str, Any], *, symbol: str, magic: int
    ) -> Optional[str]:
        """Return a short reason if pending/position/deal evidence is found."""
        intent_id = str(intent.get("id") or "")
        kind = str(intent.get("kind") or "")
        ticket = intent.get("position_ticket")

        positions = self._positions_get(symbol)
        orders = self._orders_get(symbol)
        if positions is None or orders is None:
            return None

        for order in orders:
            if magic and int(getattr(order, "magic", 0) or 0) != magic:
                continue
            if self._comment_has_intent(getattr(order, "comment", ""), intent_id):
                return f"pending ticket={getattr(order, 'ticket', None)}"

        for pos in positions:
            if magic and int(getattr(pos, "magic", 0) or 0) != magic:
                continue
            if self._comment_has_intent(getattr(pos, "comment", ""), intent_id):
                return f"position ticket={getattr(pos, 'ticket', None)}"

        if kind == "close" and ticket is not None:
            still = any(int(getattr(p, "ticket", 0)) == int(ticket) for p in positions)
            if not still:
                return f"close confirmed; position {ticket} gone"

        deals = self._history_deals_for_intent(symbol, intent_id)
        if deals:
            return f"deal ticket={getattr(deals[0], 'ticket', None)}"
        return None

    def _history_deals_for_intent(self, symbol: str, intent_id: str) -> list[Any]:
        history_fn = getattr(self.connection, "history_deals_get", None)
        if history_fn is None:
            return []
        date_to = datetime.now(timezone.utc)
        date_from = date_to - timedelta(days=2)
        try:
            raw = history_fn(date_from, date_to)
        except MT5InvokeTimeout:
            logger.warning("history_deals_get timed out while reconciling intent=%s", intent_id)
            return []
        except Exception:
            logger.exception("history_deals_get failed for intent=%s", intent_id)
            return []
        if not raw:
            return []
        matched = []
        for deal in raw:
            deal_symbol = str(getattr(deal, "symbol", "") or "")
            if deal_symbol and deal_symbol != symbol:
                continue
            if self._comment_has_intent(getattr(deal, "comment", ""), intent_id):
                matched.append(deal)
        return matched

    def _reconcile_unknown_intents(
        self,
        symbol: str,
        magic: int,
        state_store: Optional[IntentStateStore],
    ) -> Optional[ReconcileResult]:
        """Block new orders while any order intent remains result-unknown."""
        intents = self.unknown_intents(symbol, state_store)
        if not intents:
            return None

        remaining: list[str] = []
        for intent in intents:
            intent_id = str(intent.get("id") or "")
            evidence = self._find_intent_evidence(intent, symbol=symbol, magic=magic)
            if evidence:
                self._clear_intent(
                    intent_id,
                    symbol=symbol,
                    state_store=state_store,
                    reason=f"matched ({evidence})",
                )
                continue
            age = self._intent_age_sec(intent)
            if age >= self.intent_settle_sec:
                # No broker evidence after settle window — treat as never-landed
                # (typical when the queued job was skipped after abandon).
                self._clear_intent(
                    intent_id,
                    symbol=symbol,
                    state_store=state_store,
                    reason=f"absent after {age:.0f}s settle",
                )
                continue
            remaining.append(intent_id)

        if remaining:
            return ReconcileResult(
                ok=False,
                action="intent_unknown",
                message=(
                    "unresolved order intent(s) after invoke timeout; "
                    f"reorder blocked until reconciled ids={remaining}"
                ),
            )
        return None

    def _positions_get(self, symbol: str) -> Optional[list[Any]]:
        """MT5 positions query: None = error, [] = success/empty, else rows."""
        raw = self.connection.positions_get(symbol=symbol)
        if raw is None:
            logger.error(
                "positions_get failed symbol=%s: %s",
                symbol,
                self.connection.last_error(),
            )
            return None
        return list(raw)

    def _orders_get(self, symbol: str) -> Optional[list[Any]]:
        """MT5 orders query: None = error, [] = success/empty, else rows."""
        raw = self.connection.orders_get(symbol=symbol)
        if raw is None:
            logger.error(
                "orders_get failed symbol=%s: %s",
                symbol,
                self.connection.last_error(),
            )
            return None
        return list(raw)

    def managed_positions(self, symbol: str, magic: int = 260717) -> Optional[list[Any]]:
        """Return managed positions, or None when the MT5 query failed."""
        positions = self._positions_get(symbol)
        if positions is None:
            return None
        if magic == 0:
            return positions
        return [p for p in positions if int(getattr(p, "magic", 0) or 0) == magic]

    def managed_pending(self, symbol: str, magic: int = 260717) -> Optional[list[Any]]:
        """Return managed pending orders, or None when the MT5 query failed."""
        orders = self._orders_get(symbol)
        if orders is None:
            return None
        if magic == 0:
            return orders
        return [o for o in orders if int(getattr(o, "magic", 0) or 0) == magic]

    @staticmethod
    def _fetch_failed(what: str, symbol: str) -> ReconcileResult:
        return ReconcileResult(
            ok=False,
            action="fetch_failed",
            message=f"{what} query failed for {symbol}; trading halted",
        )

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
        tick = self.connection.symbol_info_tick(symbol)
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

    def place_limit(
        self,
        req: OrderRequest,
        *,
        state_store: Optional[IntentStateStore] = None,
    ) -> OrderResult:
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

        intent_id = make_intent_id()
        stamped = intent_comment(intent_id)
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if side == Signal.LONG else mt5.ORDER_TYPE_SELL_LIMIT
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": req.symbol,
            "volume": float(req.volume),
            "type": order_type,
            "price": float(price),
            "deviation": int(req.deviation),
            "magic": int(req.magic),
            "comment": stamped,
            "type_time": mt5.ORDER_TIME_GTC,
            # Pending orders must use RETURN per MT5; symbol IOC/FOK flags apply to deals only.
            "type_filling": self._pending_filling_mode(),
        }
        if req.sl is not None and float(req.sl) > 0:
            sl = float(req.sl)
            stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
            point = float(getattr(info, "point", 0) or 0)
            min_dist = stops_level * point if stops_level > 0 and point > 0 else 0.0
            if side == Signal.LONG and sl >= price:
                return OrderResult(ok=False, message=f"long SL {sl} must be below price {price}")
            if side == Signal.SHORT and sl <= price:
                return OrderResult(ok=False, message=f"short SL {sl} must be above price {price}")
            if min_dist > 0 and abs(price - sl) < min_dist:
                return OrderResult(
                    ok=False,
                    message=f"SL distance {abs(price - sl)} < stops_level min {min_dist}",
                )
            request["sl"] = sl

        if not allowed:
            logger.info("DRY-RUN order skipped (%s): %s", reason, request)
            return OrderResult(
                ok=True,
                message=f"dry-run: {reason}",
                dry_run=True,
                request=request,
                intent_id=intent_id,
            )
        return self._send(
            request,
            "order",
            intent_id=intent_id,
            kind="entry",
            state_store=state_store,
        )

    def cancel_pending(
        self,
        symbol: str,
        magic: int = 260717,
        *,
        state_store: Optional[IntentStateStore] = None,
    ) -> CancelPendingResult:
        """Cancel managed pending orders and re-verify none remain.

        Success requires a post-cancel ``orders_get`` showing zero managed
        pending for ``magic``. Failures stop callers from opening / claiming flat.
        """
        allowed, reason = self.can_execute()
        orders = self._orders_get(symbol)
        if orders is None:
            return CancelPendingResult(
                ok=False,
                fetch_failed=True,
                message=f"orders_get failed for {symbol}",
            )

        targets = [
            o
            for o in orders
            if magic == 0 or int(getattr(o, "magic", 0) or 0) == magic
        ]
        attempted = [int(o.ticket) for o in targets]
        if not targets:
            return CancelPendingResult(ok=True, message="no pending")

        if not allowed:
            logger.info(
                "Pending cancel blocked (%s) symbol=%s tickets=%s",
                reason,
                symbol,
                attempted,
            )
            return CancelPendingResult(
                ok=False,
                skipped=True,
                dry_run=True,
                message=f"cancel blocked ({reason}); pending remain {attempted}",
                attempted=attempted,
                remaining=attempted,
            )

        cancelled: list[int] = []
        failed: list[int] = []
        for order in targets:
            ticket = int(order.ticket)
            intent_id = make_intent_id()
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
                "symbol": symbol,
                "comment": intent_comment(intent_id),
                "magic": int(getattr(order, "magic", magic) or magic),
            }
            send_result = self._send(
                request,
                "cancel",
                intent_id=intent_id,
                kind="cancel",
                state_store=state_store,
            )
            if send_result.unknown:
                return CancelPendingResult(
                    ok=False,
                    unknown=True,
                    intent_id=intent_id,
                    message=(
                        f"cancel result unknown after timeout order={ticket} "
                        f"intent={intent_id}; reorder blocked"
                    ),
                    attempted=attempted,
                    cancelled=cancelled,
                    failed=failed + [ticket],
                    remaining=attempted,
                )
            if send_result.ok:
                cancelled.append(ticket)
            else:
                failed.append(ticket)
                detail = send_result.message or send_result.retcode
                logger.warning(
                    "Cancel failed order=%s retcode=%s detail=%s",
                    ticket,
                    send_result.retcode,
                    detail,
                )

        remaining_orders = self.managed_pending(symbol, magic=magic)
        if remaining_orders is None:
            return CancelPendingResult(
                ok=False,
                fetch_failed=True,
                message=f"orders_get re-check failed for {symbol}",
                attempted=attempted,
                cancelled=cancelled,
                failed=failed,
            )

        remaining = [int(o.ticket) for o in remaining_orders]
        if remaining or failed:
            message = (
                f"cancel incomplete attempted={attempted} cancelled={cancelled} "
                f"failed={failed} remaining={remaining}"
            )
            logger.error("%s symbol=%s", message, symbol)
            return CancelPendingResult(
                ok=False,
                message=message,
                attempted=attempted,
                cancelled=cancelled,
                failed=failed,
                remaining=remaining,
            )

        return CancelPendingResult(
            ok=True,
            message=f"cancelled {cancelled}" if cancelled else "no pending",
            attempted=attempted,
            cancelled=cancelled,
        )

    def _require_pending_cleared(
        self,
        symbol: str,
        magic: int = 260717,
        *,
        state_store: Optional[IntentStateStore] = None,
    ) -> Optional[ReconcileResult]:
        """Cancel managed pending; return a failure result when not fully clear."""
        result = self.cancel_pending(symbol, magic=magic, state_store=state_store)
        if result.fetch_failed:
            return self._fetch_failed("orders", symbol)
        if result.unknown:
            return ReconcileResult(
                ok=False,
                action="intent_unknown",
                message=result.message,
            )
        if not result.ok:
            return ReconcileResult(
                ok=False,
                action="cancel_failed",
                message=result.message,
                dry_run=result.dry_run,
            )
        return None

    @staticmethod
    def _deal_realized_pnl(deal: Any) -> float:
        """Realized cashflow on one MT5 deal (not open-position MTM)."""
        profit = float(getattr(deal, "profit", 0.0) or 0.0)
        swap = float(getattr(deal, "swap", 0.0) or 0.0)
        commission = float(getattr(deal, "commission", 0.0) or 0.0)
        fee = float(getattr(deal, "fee", 0.0) or 0.0)
        return profit + swap + commission + fee

    @staticmethod
    def _is_close_deal_entry(entry: Any) -> bool:
        try:
            entry_i = int(entry)
        except (TypeError, ValueError):
            return False
        out = int(getattr(mt5, "DEAL_ENTRY_OUT", 1))
        out_by = int(getattr(mt5, "DEAL_ENTRY_OUT_BY", 3))
        inout = int(getattr(mt5, "DEAL_ENTRY_INOUT", 2))
        return entry_i in (out, out_by, inout)

    def _history_deals_lookup(
        self,
        *,
        ticket: Optional[int] = None,
        order: Optional[int] = None,
        position: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> list[Any]:
        history_fn = getattr(self.connection, "history_deals_get", None)
        if history_fn is None:
            return []
        try:
            if ticket is not None:
                raw = history_fn(ticket=int(ticket))
            elif order is not None:
                raw = history_fn(order=int(order))
            elif position is not None:
                raw = history_fn(position=int(position))
            elif date_from is not None and date_to is not None:
                raw = history_fn(date_from, date_to)
            else:
                return []
        except MT5InvokeTimeout:
            logger.warning("history_deals_get timed out during realized-PnL lookup")
            return []
        except Exception:
            logger.exception("history_deals_get failed during realized-PnL lookup")
            return []
        if not raw:
            return []
        return list(raw)

    def _realized_close_pnl(
        self,
        *,
        deal_ticket: Optional[int],
        order_ticket: Optional[int],
        position_ticket: int,
        intent_id: Optional[str],
        symbol: str,
        attempts: int = 3,
        retry_delay_sec: float = 0.15,
    ) -> Optional[float]:
        """Fetch realized close PnL from deal history after a successful close.

        Returns ``None`` when the close deal cannot be confirmed — callers must
        not feed unconfirmed MTM estimates into Kelly learning.
        """
        for attempt in range(max(1, int(attempts))):
            deals: list[Any] = []

            if deal_ticket:
                deals = self._history_deals_lookup(ticket=int(deal_ticket))

            if not deals and order_ticket:
                deals = [
                    d
                    for d in self._history_deals_lookup(order=int(order_ticket))
                    if self._is_close_deal_entry(getattr(d, "entry", None))
                ]

            if not deals and intent_id:
                # Intent-scoped fallback only — never sum unrelated prior OUT deals.
                pos_deals = self._history_deals_lookup(position=int(position_ticket))
                if not pos_deals:
                    date_to = datetime.now(timezone.utc)
                    date_from = date_to - timedelta(days=2)
                    pos_deals = [
                        d
                        for d in self._history_deals_lookup(
                            date_from=date_from, date_to=date_to
                        )
                        if str(getattr(d, "symbol", "") or "") in ("", symbol)
                        and int(getattr(d, "position_id", 0) or 0)
                        in (0, int(position_ticket))
                    ]
                deals = [
                    d
                    for d in pos_deals
                    if self._is_close_deal_entry(getattr(d, "entry", None))
                    and self._comment_has_intent(getattr(d, "comment", ""), intent_id)
                ]

            if deals:
                if deal_ticket:
                    exact = [
                        d
                        for d in deals
                        if int(getattr(d, "ticket", 0) or 0) == int(deal_ticket)
                    ]
                    if exact:
                        deals = exact
                total = sum(self._deal_realized_pnl(d) for d in deals)
                logger.info(
                    "Realized close PnL=%.4f from %d deal(s) position=%s deal=%s",
                    total,
                    len(deals),
                    position_ticket,
                    deal_ticket,
                )
                return float(total)

            if attempt + 1 < attempts:
                time.sleep(max(0.0, float(retry_delay_sec)))

        logger.warning(
            "Close deal history unconfirmed for position=%s deal=%s order=%s "
            "intent=%s — leaving closed_pnl unset (Kelly skip)",
            position_ticket,
            deal_ticket,
            order_ticket,
            intent_id,
        )
        return None

    def close_position_market(
        self,
        position: Any,
        magic: int = 260717,
        volume: Optional[float] = None,
        *,
        state_store: Optional[IntentStateStore] = None,
    ) -> OrderResult:
        """Close one MT5 position ticket at market (full or partial volume)."""
        allowed, reason = self.can_execute()
        symbol = str(position.symbol)
        if not self.connection.ensure():
            return OrderResult(ok=False, message="not connected")
        tick = self.connection.symbol_info_tick(symbol)
        info = self.connection.symbol_info(symbol)
        if tick is None or info is None:
            return OrderResult(ok=False, message="missing tick/info")

        pos_volume = float(position.volume)
        close_volume = pos_volume if volume is None else min(pos_volume, float(volume))
        if close_volume <= 0:
            return OrderResult(ok=False, message="close volume <= 0")

        intent_id = make_intent_id()
        stamped = intent_comment(intent_id)
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(close_volume),
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": int(position.ticket),
            "price": float(tick.bid if is_buy else tick.ask),
            "deviation": 50,
            "magic": int(magic),
            "comment": stamped,
            "type_filling": self._deal_filling_mode(info),
        }
        if not allowed:
            logger.info("DRY-RUN close skipped (%s): %s", reason, request)
            # No realized deal — do not invent MTM PnL for Kelly.
            return OrderResult(
                ok=True,
                message=f"dry-run close: {reason}",
                dry_run=True,
                request=request,
                closed_pnl=None,
                intent_id=intent_id,
            )
        result = self._send(
            request,
            "close",
            intent_id=intent_id,
            kind="close",
            state_store=state_store,
            position_ticket=int(position.ticket),
        )
        if result.ok:
            result.closed_pnl = self._realized_close_pnl(
                deal_ticket=result.deal,
                order_ticket=result.order,
                position_ticket=int(position.ticket),
                intent_id=intent_id,
                symbol=symbol,
            )
        return result

    def close_managed_positions(
        self,
        symbol: str,
        magic: int = 260717,
        *,
        max_rounds: int = 3,
        state_store: Optional[IntentStateStore] = None,
    ) -> Optional[list[OrderResult]]:
        """Close managed positions, retrying leftovers after partial fills.

        Returns the list of close attempts, or None when a position query failed.
        Callers must still re-fetch and confirm flat before opening the opposite side.

        Result-unknown (invoke timeout) stops further closes — never auto-retry the
        same close until the intent is reconciled.
        """
        results: list[OrderResult] = []
        rounds = max(1, int(max_rounds))
        for _ in range(rounds):
            if any(r.unknown for r in results):
                return results
            positions = self.managed_positions(symbol, magic)
            if positions is None:
                return None
            if not positions:
                return results
            for position in positions:
                result = self.close_position_market(
                    position, magic=magic, state_store=state_store
                )
                results.append(result)
                if result.unknown:
                    return results
        return results

    def _require_managed_flat(
        self,
        symbol: str,
        *,
        magic: int,
        dry: bool,
        orders: list[OrderResult],
        action: str = "close_incomplete",
    ) -> Optional[ReconcileResult]:
        """Fail closed if managed positions remain after close attempts."""
        if any(r.unknown for r in orders):
            unknown_ids = [r.intent_id for r in orders if r.unknown and r.intent_id]
            return ReconcileResult(
                ok=False,
                action="intent_unknown",
                message=(
                    "close result unknown after invoke timeout; "
                    f"reorder blocked until reconciled ids={unknown_ids}"
                ),
                dry_run=dry,
                orders=orders,
            )
        positions_after = self.managed_positions(symbol, magic=magic)
        if positions_after is None:
            return self._fetch_failed("positions", symbol)
        if positions_after and not dry:
            tickets = [int(p.ticket) for p in positions_after]
            vols = [float(p.volume) for p in positions_after]
            return ReconcileResult(
                ok=False,
                action=action,
                message=(
                    "managed positions remain after close; new entry blocked "
                    f"tickets={tickets} volumes={vols}"
                ),
                dry_run=dry,
                orders=orders,
            )
        return None

    def reconcile_target(
        self,
        *,
        symbol: str,
        side: Signal,
        volume: float,
        volume_step: float = 0.01,
        comment: str = "lr_loop",
        magic: int = 260717,
        sl: Optional[float] = None,
        rebalance_band: float = 0.15,
        state_store: Optional[IntentStateStore] = None,
    ) -> ReconcileResult:
        """Move managed positions toward one desired side/volume without stacking.

        Same-side exposure uses entry-sized lots within ``rebalance_band`` (relative).
        Outside the band, only the delta is topped up or trimmed — not a full
        close/re-open. Target match is filled + same-side pending (RETURN partials).

        After an ``order_send`` invoke timeout, intents stay ``unknown`` until
        matched via orders/positions/deals (or settled absent). New orders are
        blocked until then — timeouts must not auto-retry.
        """
        del comment  # live comments are intent-stamped; regime text is not broker-safe alone
        block = self._reconcile_unknown_intents(symbol, magic, state_store)
        if block is not None:
            return block

        positions = self.managed_positions(symbol, magic=magic)
        pending = self.managed_pending(symbol, magic=magic)
        if positions is None:
            return self._fetch_failed("positions", symbol)
        if pending is None:
            return self._fetch_failed("orders", symbol)

        current_sides = {
            Signal.LONG if p.type == mt5.POSITION_TYPE_BUY else Signal.SHORT for p in positions
        }
        current_volume = sum(float(p.volume) for p in positions)
        tolerance = max(float(volume_step), 1e-9) / 2.0
        band = max(0.0, float(rebalance_band))

        if side == Signal.FLAT or volume <= 0:
            cancel_err = self._require_pending_cleared(
                symbol, magic=magic, state_store=state_store
            )
            if cancel_err is not None:
                return cancel_err
            closes = self.close_managed_positions(
                symbol, magic=magic, state_store=state_store
            )
            if closes is None:
                return self._fetch_failed("positions", symbol)
            if any(r.unknown for r in closes):
                return ReconcileResult(
                    ok=False,
                    action="intent_unknown",
                    message="flatten close result unknown; reorder blocked until reconciled",
                    dry_run=any(r.dry_run for r in closes),
                    orders=closes,
                )
            if any(not r.ok for r in closes):
                return ReconcileResult(
                    ok=False,
                    action="close_failed",
                    message="flatten close failed",
                    dry_run=any(r.dry_run for r in closes),
                    orders=closes,
                )

            positions_after = self.managed_positions(symbol, magic=magic)
            pending_after = self.managed_pending(symbol, magic=magic)
            if positions_after is None:
                return self._fetch_failed("positions", symbol)
            if pending_after is None:
                return self._fetch_failed("orders", symbol)

            dry = any(r.dry_run for r in closes)
            if pending_after or (positions_after and not dry):
                return ReconcileResult(
                    ok=False,
                    action="flatten_incomplete",
                    message=(
                        f"flat requires positions=0 and pending=0; "
                        f"got positions={len(positions_after)} pending={len(pending_after)}"
                    ),
                    dry_run=dry,
                    orders=closes,
                )

            had_exposure = bool(positions or pending)
            return ReconcileResult(
                ok=True,
                action="flatten" if had_exposure else "hold_flat",
                message=(
                    "dry-run flat (pending=0; position closes simulated)"
                    if dry
                    else "target flat (positions=0, pending=0)"
                ),
                dry_run=dry,
                orders=closes,
            )

        # Fully filled to target — cancel any leftover working orders.
        fully_filled = (
            len(current_sides) == 1
            and side in current_sides
            and abs(current_volume - float(volume)) <= tolerance
        )
        if fully_filled:
            cancel_err = self._require_pending_cleared(
                symbol, magic=magic, state_store=state_store
            )
            if cancel_err is not None:
                return cancel_err
            return ReconcileResult(ok=True, action="hold", message="target already satisfied")

        # RETURN partial fill: position + same-side pending remainder == target.
        if self._working_exposure_matches_target(
            side=side,
            target_volume=float(volume),
            positions=positions,
            pending=pending,
            tolerance=tolerance,
        ):
            logger.info(
                "Awaiting fill symbol=%s side=%s target=%s filled=%s pending=%s",
                symbol,
                side.name,
                volume,
                current_volume if side in current_sides else 0.0,
                sum(
                    self._pending_volume(o)
                    for o in pending
                    if self._pending_side(o) == side
                ),
            )
            return ReconcileResult(
                ok=True,
                action="await_fill",
                message="working exposure matches target (filled + same-side pending)",
            )

        same_side = self._same_side_exposure_clean(
            side=side, positions=positions, pending=pending
        )
        if same_side:
            filled, pending_vol, working = same_side
            basis = max(working, float(volume), 1e-12)
            rel = abs(working - float(volume)) / basis
            if rel <= band:
                # Entry-sized hold: ignore equity-driven micro lot changes.
                if pending_vol > 0 and working > float(volume) + tolerance:
                    cancel_err = self._require_pending_cleared(
                        symbol, magic=magic, state_store=state_store
                    )
                    if cancel_err is not None:
                        return cancel_err
                return ReconcileResult(
                    ok=True,
                    action="hold",
                    message=(
                        f"within rebalance band ({rel:.1%} <= {band:.0%}); "
                        f"keeping working={working:.4f} vs target={float(volume):.4f}"
                    ),
                )
            return self._delta_rebalance_same_side(
                symbol=symbol,
                side=side,
                target_volume=float(volume),
                volume_step=volume_step,
                tolerance=tolerance,
                magic=magic,
                sl=sl,
                filled=filled,
                pending_vol=pending_vol,
                working=working,
                state_store=state_store,
            )

        # Side change or mixed exposure — conservative close-then-open.
        cancel_err = self._require_pending_cleared(
            symbol, magic=magic, state_store=state_store
        )
        if cancel_err is not None:
            return cancel_err
        closes = self.close_managed_positions(
            symbol, magic=magic, state_store=state_store
        )
        if closes is None:
            return self._fetch_failed("positions", symbol)

        dry = any(r.dry_run for r in closes)
        # Position flatness is authoritative; DONE_PARTIAL is incomplete until re-query is empty.
        flat_err = self._require_managed_flat(
            symbol, magic=magic, dry=dry, orders=closes
        )
        if flat_err is not None:
            hard_fail = any(not r.ok and not r.partial and not r.unknown for r in closes)
            if hard_fail:
                flat_err.action = "close_failed"
                flat_err.message = (
                    "existing position close failed; new entry blocked; "
                    + flat_err.message
                )
            return flat_err

        pending_before_entry = self.managed_pending(symbol, magic=magic)
        if pending_before_entry is None:
            return self._fetch_failed("orders", symbol)
        if pending_before_entry:
            return ReconcileResult(
                ok=False,
                action="cancel_failed",
                message=(
                    "managed pending remain before new entry: "
                    f"{[int(o.ticket) for o in pending_before_entry]}"
                ),
                dry_run=dry,
                orders=closes,
            )

        entry = self.place_limit(
            OrderRequest(
                symbol=symbol,
                side=side,
                volume=float(volume),
                price=0.0,
                magic=magic,
                sl=sl,
            ),
            state_store=state_store,
        )
        if entry.unknown:
            return ReconcileResult(
                ok=False,
                action="intent_unknown",
                message=entry.message,
                dry_run=entry.dry_run or dry,
                orders=closes + [entry],
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
            dry_run=entry.dry_run or dry,
            orders=closes + [entry],
        )

    def _same_side_exposure_clean(
        self,
        *,
        side: Signal,
        positions: list[Any],
        pending: list[Any],
    ) -> Optional[tuple[float, float, float]]:
        """Return (filled, pending_vol, working) when all exposure is ``side``."""
        position_sides = {
            Signal.LONG if p.type == mt5.POSITION_TYPE_BUY else Signal.SHORT for p in positions
        }
        if len(position_sides) > 1:
            return None
        if position_sides and side not in position_sides:
            return None
        pending_sides = {self._pending_side(o) for o in pending}
        pending_sides.discard(None)
        if any(s != side for s in pending_sides):
            return None
        filled = sum(float(p.volume) for p in positions) if side in position_sides else 0.0
        pending_vol = sum(
            self._pending_volume(o) for o in pending if self._pending_side(o) == side
        )
        working = filled + pending_vol
        if working <= 0:
            return None
        return filled, pending_vol, working

    def _delta_rebalance_same_side(
        self,
        *,
        symbol: str,
        side: Signal,
        target_volume: float,
        volume_step: float,
        tolerance: float,
        magic: int,
        sl: Optional[float],
        filled: float,
        pending_vol: float,
        working: float,
        state_store: Optional[IntentStateStore] = None,
    ) -> ReconcileResult:
        """Top up or trim only the delta versus target on the same side."""
        delta = target_volume - working
        if delta > tolerance:
            add_vol = self._round_volume_up(delta, volume_step)
            if add_vol <= 0:
                return ReconcileResult(
                    ok=True,
                    action="hold",
                    message=f"top-up delta {delta} below volume step",
                )
            entry = self.place_limit(
                OrderRequest(
                    symbol=symbol,
                    side=side,
                    volume=float(add_vol),
                    price=0.0,
                    magic=magic,
                    sl=sl,
                ),
                state_store=state_store,
            )
            if entry.unknown:
                return ReconcileResult(
                    ok=False,
                    action="intent_unknown",
                    message=entry.message,
                    dry_run=entry.dry_run,
                    orders=[entry],
                )
            if entry.ok and not entry.dry_run:
                entry.message = f"top-up {add_vol} awaiting fill ({entry.message})"
            return ReconcileResult(
                ok=entry.ok,
                action="top_up",
                message=entry.message,
                dry_run=entry.dry_run,
                orders=[entry],
            )

        # Trim: clear residual pendings first, then partial-close filled excess.
        cancel_err = self._require_pending_cleared(
            symbol, magic=magic, state_store=state_store
        )
        if cancel_err is not None:
            return cancel_err
        positions = self.managed_positions(symbol, magic=magic)
        if positions is None:
            return self._fetch_failed("positions", symbol)
        filled_now = sum(float(p.volume) for p in positions)
        excess = filled_now - target_volume
        if excess <= tolerance:
            return ReconcileResult(
                ok=True,
                action="hold",
                message="trimmed pending; filled already near target",
            )
        closes = self._partial_close_volume(
            positions, excess, magic=magic, state_store=state_store
        )
        if any(r.unknown for r in closes):
            return ReconcileResult(
                ok=False,
                action="intent_unknown",
                message="partial trim result unknown; reorder blocked until reconciled",
                dry_run=any(r.dry_run for r in closes),
                orders=closes,
            )
        if any(not r.ok for r in closes):
            return ReconcileResult(
                ok=False,
                action="close_failed",
                message="partial trim failed",
                dry_run=any(r.dry_run for r in closes),
                orders=closes,
            )
        return ReconcileResult(
            ok=True,
            action="trim",
            message=f"trimmed excess={excess:.4f} toward target={target_volume:.4f}",
            dry_run=any(r.dry_run for r in closes),
            orders=closes,
        )

    def _partial_close_volume(
        self,
        positions: list[Any],
        excess: float,
        magic: int,
        *,
        state_store: Optional[IntentStateStore] = None,
    ) -> list[OrderResult]:
        remaining = float(excess)
        results: list[OrderResult] = []
        for position in positions:
            if remaining <= 0:
                break
            slice_vol = min(float(position.volume), remaining)
            result = self.close_position_market(
                position, magic=magic, volume=slice_vol, state_store=state_store
            )
            results.append(result)
            if result.unknown:
                break
            if result.ok:
                remaining -= slice_vol
        return results

    @staticmethod
    def _round_volume_up(volume: float, step: float) -> float:
        if step <= 0:
            return float(volume)
        return float(math.floor(volume / step + 1e-12) * step)

    def _working_exposure_matches_target(
        self,
        *,
        side: Signal,
        target_volume: float,
        positions: list[Any],
        pending: list[Any],
        tolerance: float,
    ) -> bool:
        """True when filled + same-side pending remainder equals the target."""
        if target_volume <= 0:
            return False

        position_sides = {
            Signal.LONG if p.type == mt5.POSITION_TYPE_BUY else Signal.SHORT for p in positions
        }
        if len(position_sides) > 1:
            return False
        if position_sides and side not in position_sides:
            return False

        pending_sides = {self._pending_side(o) for o in pending}
        pending_sides.discard(None)
        if not pending_sides and not positions:
            return False
        if any(s != side for s in pending_sides):
            return False

        filled = sum(float(p.volume) for p in positions) if side in position_sides else 0.0
        pending_vol = sum(
            self._pending_volume(o) for o in pending if self._pending_side(o) == side
        )
        working = filled + pending_vol
        return working > 0 and abs(working - target_volume) <= tolerance

    def close_position_limit(self, symbol: str, magic: int = 260717) -> Optional[OrderResult]:
        """Backward-compatible wrapper; now closes an exact position at market."""
        positions = self.managed_positions(symbol, magic=magic)
        if positions is None:
            return OrderResult(ok=False, message="positions_get failed; close aborted")
        if not positions:
            return OrderResult(ok=True, message="no position")
        return self.close_position_market(positions[0], magic=magic)

    def close_all(self, symbol: str, magic: int = 260717) -> OrderResult:
        """Kill-switch flatten for all positions on a symbol."""
        cancel = self.cancel_pending(symbol, magic=0)
        if cancel.fetch_failed:
            return OrderResult(ok=False, message="orders_get failed; flatten aborted")
        if cancel.unknown:
            return OrderResult(
                ok=False,
                unknown=True,
                intent_id=cancel.intent_id,
                message=f"cancel result unknown; flatten aborted ({cancel.message})",
            )
        if not cancel.ok:
            return OrderResult(
                ok=False,
                message=f"cancel_failed; flatten aborted ({cancel.message})",
            )

        pending = self.managed_pending(symbol, magic=0)
        if pending is None:
            return OrderResult(ok=False, message="orders_get failed; flatten aborted")
        if pending:
            tickets = [int(o.ticket) for o in pending]
            return OrderResult(
                ok=False,
                message=f"pending remain after cancel; flatten aborted tickets={tickets}",
            )

        positions = self.managed_positions(symbol, magic=0)
        if positions is None:
            return OrderResult(ok=False, message="positions_get failed; flatten aborted")
        if not positions:
            return OrderResult(ok=True, message="no position")
        results = [self.close_position_market(p, magic=magic) for p in positions]
        unknowns = [r for r in results if r.unknown]
        if unknowns:
            return unknowns[-1]
        failures = [r for r in results if not r.ok]
        return failures[-1] if failures else results[-1]

    @staticmethod
    def _success_retcodes() -> set[int]:
        """MT5 retcodes that mean the request fully completed or was accepted as pending.

        ``TRADE_RETCODE_DONE_PARTIAL`` is intentionally excluded — partial fills are
        incomplete and must be retried / verified flat before opposite entry.
        """
        codes = {
            int(mt5.TRADE_RETCODE_DONE),
            int(getattr(mt5, "TRADE_RETCODE_PLACED", 10008)),
        }
        return codes

    @classmethod
    def _is_partial_retcode(cls, retcode: Any) -> bool:
        try:
            return int(retcode) == int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _check_retcode_ok(cls, retcode: Any) -> bool:
        """``order_check`` success: broker may return 0 or TRADE_RETCODE_DONE."""
        try:
            code = int(retcode)
        except (TypeError, ValueError):
            return False
        return code == 0 or code == int(mt5.TRADE_RETCODE_DONE)

    @classmethod
    def _retcode_ok(cls, retcode: Any) -> bool:
        try:
            return int(retcode) in cls._success_retcodes()
        except (TypeError, ValueError):
            return False

    def _preflight_check(
        self,
        request: dict[str, Any],
        operation: str,
        *,
        intent_id: Optional[str] = None,
    ) -> Optional[OrderResult]:
        """Run MT5 ``order_check``; return a failed OrderResult or None if OK.

        Covers volume / price / stop level / margin / trade permissions /
        filling mode / market state as enforced by the terminal & broker.
        """
        try:
            check = self.connection.order_check(request)
        except MT5InvokeTimeout as exc:
            logger.error(
                "%s order_check timed out (%s); refusing order_send",
                operation,
                exc.fn_name,
            )
            return OrderResult(
                ok=False,
                intent_id=intent_id,
                message=(
                    f"order_check timeout ({exc.fn_name}); order_send not attempted"
                ),
                request=request,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed before live send
            logger.exception("%s order_check raised; refusing order_send", operation)
            return OrderResult(
                ok=False,
                intent_id=intent_id,
                message=f"order_check error: {exc}",
                request=request,
            )

        if check is None:
            err = self.connection.last_error()
            logger.error("%s order_check returned None: %s", operation, err)
            return OrderResult(
                ok=False,
                intent_id=intent_id,
                message=f"order_check failed: {err}",
                request=request,
            )

        retcode = getattr(check, "retcode", None)
        comment = getattr(check, "comment", "") or ""
        if not self._check_retcode_ok(retcode):
            logger.warning(
                "%s order_check rejected retcode=%s comment=%s margin=%s "
                "margin_free=%s",
                operation,
                retcode,
                comment,
                getattr(check, "margin", None),
                getattr(check, "margin_free", None),
            )
            return OrderResult(
                ok=False,
                retcode=int(retcode) if retcode is not None else None,
                intent_id=intent_id,
                message=f"order_check rejected: {comment or retcode}",
                request=request,
            )
        logger.debug(
            "%s order_check ok retcode=%s margin=%s margin_free=%s",
            operation,
            retcode,
            getattr(check, "margin", None),
            getattr(check, "margin_free", None),
        )
        return None

    def _send(
        self,
        request: dict[str, Any],
        operation: str,
        *,
        intent_id: Optional[str] = None,
        kind: str = "order",
        state_store: Optional[IntentStateStore] = None,
        position_ticket: Optional[int] = None,
    ) -> OrderResult:
        intent_id = intent_id or make_intent_id()
        symbol = str(request.get("symbol") or "")
        magic = int(request.get("magic") or 0)
        comment = str(request.get("comment") or intent_comment(intent_id))[:31]
        request = dict(request)
        request["comment"] = comment

        check_fail = self._preflight_check(request, operation, intent_id=intent_id)
        if check_fail is not None:
            return check_fail

        try:
            result = self.connection.order_send(request)
        except MT5InvokeTimeout as exc:
            self._record_unknown_intent(
                intent_id=intent_id,
                symbol=symbol,
                kind=kind,
                comment=comment,
                magic=magic,
                state_store=state_store,
                position_ticket=position_ticket,
            )
            return OrderResult(
                ok=False,
                unknown=True,
                intent_id=intent_id,
                message=(
                    f"result unknown after invoke timeout ({exc.fn_name}); "
                    "do not auto-retry — reconcile via orders/positions/deals"
                ),
                request=request,
            )

        if result is None:
            err = self.connection.last_error()
            logger.error("%s order_send returned None: %s", operation, err)
            return OrderResult(
                ok=False, message=str(err), request=request, intent_id=intent_id
            )
        retcode = int(result.retcode)
        message = getattr(result, "comment", "") or str(retcode)
        if self._is_partial_retcode(retcode):
            logger.warning(
                "%s partial fill retcode=%s comment=%s — treating as incomplete",
                operation,
                retcode,
                message,
            )
            return OrderResult(
                ok=False,
                retcode=retcode,
                order=int(result.order) if getattr(result, "order", 0) else None,
                deal=int(result.deal) if getattr(result, "deal", 0) else None,
                message=f"partial fill; incomplete ({message})",
                request=request,
                partial=True,
                intent_id=intent_id,
            )
        ok = OrderExecutor._retcode_ok(retcode)
        if not ok:
            logger.warning("%s rejected retcode=%s comment=%s", operation, retcode, message)
        return OrderResult(
            ok=ok,
            retcode=retcode,
            order=int(result.order) if getattr(result, "order", 0) else None,
            deal=int(result.deal) if getattr(result, "deal", 0) else None,
            message=message,
            request=request,
            intent_id=intent_id,
        )

    @staticmethod
    def _pending_filling_mode() -> int:
        """Filling mode for pending orders (limit/stop). Always RETURN."""
        return mt5.ORDER_FILLING_RETURN

    @staticmethod
    def _deal_filling_mode(info: Any) -> int:
        """Filling mode for market deals (close / instant execution)."""
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
