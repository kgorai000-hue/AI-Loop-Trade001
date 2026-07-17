"""Half-Kelly position sizing from max loss at stop, plus cost model."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostModel:
    """Cost per round-trip as a fraction of notional. Floor = min_cost_bps."""

    min_cost_bps: float = 10.0  # 10 bps = 0.1%
    spread_points: Optional[float] = None
    point: Optional[float] = None
    commission_per_lot: Optional[float] = None
    contract_size: Optional[float] = None
    price: Optional[float] = None

    def one_way_fraction(self) -> float:
        """Estimated one-way cost as fraction of price."""
        floor = self.min_cost_bps / 10_000.0
        parts: list[float] = []

        if (
            self.spread_points is not None
            and self.point is not None
            and self.price
            and self.price > 0
        ):
            spread_frac = (self.spread_points * self.point) / self.price
            parts.append(spread_frac)

        if (
            self.commission_per_lot is not None
            and self.contract_size
            and self.price
            and self.price > 0
            and self.contract_size > 0
        ):
            notional = self.contract_size * self.price
            if notional > 0:
                parts.append(self.commission_per_lot / notional)

        if not parts:
            return floor
        return max(floor, sum(parts))

    def round_trip_fraction(self) -> float:
        return 2.0 * self.one_way_fraction()

    @classmethod
    def from_symbol_info(
        cls,
        symbol_info: Any,
        tick: Any = None,
        min_cost_bps: float = 10.0,
    ) -> "CostModel":
        if symbol_info is None:
            return cls(min_cost_bps=min_cost_bps)

        spread = float(getattr(symbol_info, "spread", 0) or 0)
        point = float(getattr(symbol_info, "point", 0) or 0)
        contract = float(getattr(symbol_info, "trade_contract_size", 0) or 0)
        price = None
        if tick is not None:
            bid = float(getattr(tick, "bid", 0) or 0)
            ask = float(getattr(tick, "ask", 0) or 0)
            if bid > 0 and ask > 0:
                price = 0.5 * (bid + ask)
        if price is None:
            price = float(getattr(symbol_info, "bid", 0) or getattr(symbol_info, "ask", 0) or 0)

        commission = None
        for attr in ("trade_commission", "commission"):
            val = getattr(symbol_info, attr, None)
            if val is not None:
                try:
                    commission = float(val)
                    break
                except (TypeError, ValueError):
                    pass

        model = cls(
            min_cost_bps=min_cost_bps,
            spread_points=spread if spread > 0 else None,
            point=point if point > 0 else None,
            commission_per_lot=commission,
            contract_size=contract if contract > 0 else None,
            price=price if price and price > 0 else None,
        )
        logger.debug(
            "CostModel one_way=%.5f round_trip=%.5f (floor=%.5f)",
            model.one_way_fraction(),
            model.round_trip_fraction(),
            min_cost_bps / 10_000.0,
        )
        return model


@dataclass
class LotDecision:
    """Sized position with explicit stop and risk accounting."""

    lots: float
    stop_loss: Optional[float]
    stop_distance: float
    risk_capital: float
    risk_per_lot: float
    open_risk_reserved: float = 0.0
    margin_capped: bool = False
    message: str = ""


@dataclass
class RiskManager:
    half_kelly: bool = True
    default_win_rate: float = 0.52
    default_reward_risk: float = 1.5
    max_fraction: float = 0.25
    max_lots: float = 5.0
    min_lots: float = 0.01
    lookback_trades: int = 50
    min_cost_bps: float = 10.0
    stop_pct: float = 0.005
    stop_points: Optional[float] = None
    gap_buffer_mult: float = 1.25
    max_open_risk_fraction: float = 0.25
    max_margin_fraction: float = 0.50
    recent_pnls: list[float] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "RiskManager":
        cfg = cfg or {}
        stop_points = cfg.get("stop_points")
        return cls(
            half_kelly=bool(cfg.get("half_kelly", True)),
            default_win_rate=float(cfg.get("default_win_rate", 0.52)),
            default_reward_risk=float(cfg.get("default_reward_risk", 1.5)),
            max_fraction=float(cfg.get("max_fraction", 0.25)),
            max_lots=float(cfg.get("max_lots", 5.0)),
            min_lots=float(cfg.get("min_lots", 0.01)),
            lookback_trades=int(cfg.get("lookback_trades", 50)),
            min_cost_bps=float(cfg.get("min_cost_bps", 10)),
            stop_pct=float(cfg.get("stop_pct", 0.005)),
            stop_points=float(stop_points) if stop_points is not None else None,
            gap_buffer_mult=float(cfg.get("gap_buffer_mult", 1.25)),
            max_open_risk_fraction=float(cfg.get("max_open_risk_fraction", 0.25)),
            max_margin_fraction=float(cfg.get("max_margin_fraction", 0.50)),
        )

    def record_trade(self, pnl: float) -> None:
        self.recent_pnls.append(float(pnl))
        if len(self.recent_pnls) > self.lookback_trades * 2:
            self.recent_pnls = self.recent_pnls[-self.lookback_trades :]

    def load_trade_history(self, pnls: list[float]) -> None:
        self.recent_pnls = [float(p) for p in pnls][-self.lookback_trades * 2 :]

    def estimate_wr_rr(self) -> tuple[float, float]:
        pnls = self.recent_pnls[-self.lookback_trades :]
        if len(pnls) < 5:
            return self.default_win_rate, self.default_reward_risk
        arr = list(pnls)
        wins = [p for p in arr if p > 0]
        losses = [p for p in arr if p < 0]
        wr = len(wins) / len(arr)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        rr = avg_win / avg_loss if avg_loss > 0 else self.default_reward_risk
        rr = max(0.1, rr)
        return float(wr), float(rr)

    def kelly_fraction(
        self, win_rate: Optional[float] = None, reward_risk: Optional[float] = None
    ) -> float:
        """f* = W - (1-W)/R ; half-Kelly = 0.5 * f*."""
        w = self.default_win_rate if win_rate is None else win_rate
        r = self.default_reward_risk if reward_risk is None else reward_risk
        if r <= 0:
            return 0.0
        full = w - (1.0 - w) / r
        frac = 0.5 * full if self.half_kelly else full
        frac = max(0.0, min(frac, self.max_fraction))
        return float(frac)

    def stop_distance_price(self, price: float, point: float = 0.01) -> float:
        """Price distance from entry to stop (before gap buffer)."""
        if self.stop_points is not None and self.stop_points > 0 and point > 0:
            return float(self.stop_points) * float(point)
        if price <= 0 or self.stop_pct <= 0:
            return 0.0
        return float(price) * float(self.stop_pct)

    @staticmethod
    def loss_per_lot(
        stop_distance: float,
        tick_size: float,
        tick_value: float,
    ) -> float:
        """Account-currency loss for one lot if price moves ``stop_distance`` against us.

        Uses MT5 ``trade_tick_value`` / ``trade_tick_size`` so FX conversion is
        already embedded when the terminal reports deposit-currency tick value.
        """
        if stop_distance <= 0 or tick_size <= 0 or tick_value <= 0:
            return 0.0
        return (stop_distance / tick_size) * tick_value

    @staticmethod
    def estimate_position_open_risk(
        *,
        volume: float,
        current_price: float,
        side_long: bool,
        stop_loss: Optional[float],
        tick_size: float,
        tick_value: float,
        fallback_stop_distance: float,
    ) -> float:
        """Reserved max-loss for an open position (SL distance or fallback stop)."""
        if volume <= 0:
            return 0.0
        if stop_loss is not None and stop_loss > 0 and current_price > 0:
            if side_long:
                dist = max(0.0, current_price - float(stop_loss))
            else:
                dist = max(0.0, float(stop_loss) - current_price)
        else:
            dist = fallback_stop_distance
        return volume * RiskManager.loss_per_lot(dist, tick_size, tick_value)

    def stop_loss_price(
        self,
        *,
        side_long: bool,
        entry_price: float,
        stop_distance: float,
        digits: int = 2,
    ) -> Optional[float]:
        if entry_price <= 0 or stop_distance <= 0:
            return None
        raw = entry_price - stop_distance if side_long else entry_price + stop_distance
        return float(round(raw, digits))

    def position_lots(
        self,
        equity: float,
        price: float,
        contract_size: float = 1.0,
        volume_step: float = 0.01,
        volume_min: Optional[float] = None,
        volume_max: Optional[float] = None,
        win_rate: Optional[float] = None,
        reward_risk: Optional[float] = None,
        *,
        tick_size: Optional[float] = None,
        tick_value: Optional[float] = None,
        point: Optional[float] = None,
        side_long: bool = True,
        digits: int = 2,
        free_margin: Optional[float] = None,
        margin_per_lot: Optional[float] = None,
        open_risk: float = 0.0,
    ) -> LotDecision:
        """Size lots from Kelly risk capital / max loss at (gap-buffered) stop."""
        empty = LotDecision(
            lots=0.0,
            stop_loss=None,
            stop_distance=0.0,
            risk_capital=0.0,
            risk_per_lot=0.0,
            open_risk_reserved=max(0.0, float(open_risk)),
            message="invalid inputs",
        )
        if equity <= 0 or price <= 0:
            return empty

        if win_rate is None or reward_risk is None:
            ew, er = self.estimate_wr_rr()
            win_rate = win_rate if win_rate is not None else ew
            reward_risk = reward_risk if reward_risk is not None else er

        frac = self.kelly_fraction(win_rate, reward_risk)
        if frac <= 0:
            return LotDecision(
                lots=0.0,
                stop_loss=None,
                stop_distance=0.0,
                risk_capital=0.0,
                risk_per_lot=0.0,
                open_risk_reserved=max(0.0, float(open_risk)),
                message="kelly fraction=0",
            )

        point_v = float(point) if point and point > 0 else 0.01
        base_stop = self.stop_distance_price(price, point_v)
        stop_distance = base_stop * max(1.0, float(self.gap_buffer_mult))
        sl = self.stop_loss_price(
            side_long=side_long,
            entry_price=price,
            stop_distance=stop_distance,
            digits=digits,
        )

        kelly_budget = equity * frac
        open_reserved = max(0.0, float(open_risk))
        total_cap = equity * float(self.max_open_risk_fraction)
        risk_capital = min(kelly_budget, max(0.0, total_cap - open_reserved))

        ts = float(tick_size) if tick_size and tick_size > 0 else 0.0
        tv = float(tick_value) if tick_value and tick_value > 0 else 0.0
        risk_per_lot = self.loss_per_lot(stop_distance, ts, tv)

        if risk_per_lot > 0 and risk_capital > 0:
            raw_lots = risk_capital / risk_per_lot
            mode = "max_loss"
        elif contract_size > 0:
            notional_per_lot = price * contract_size
            raw_lots = kelly_budget / notional_per_lot if notional_per_lot > 0 else 0.0
            mode = "notional_fallback"
            logger.warning(
                "position_lots using notional fallback (tick_size/value missing); "
                "stop-based risk unavailable"
            )
        else:
            return LotDecision(
                lots=0.0,
                stop_loss=sl,
                stop_distance=stop_distance,
                risk_capital=risk_capital,
                risk_per_lot=0.0,
                open_risk_reserved=open_reserved,
                message="cannot size: missing tick value/size and contract",
            )

        vmin = volume_min if volume_min is not None else self.min_lots
        vmax = volume_max if volume_max is not None else self.max_lots
        vmax = min(vmax, self.max_lots)
        vmin = max(vmin, self.min_lots)

        margin_capped = False
        if (
            free_margin is not None
            and margin_per_lot is not None
            and free_margin > 0
            and margin_per_lot > 0
        ):
            margin_lots = (free_margin * float(self.max_margin_fraction)) / margin_per_lot
            if margin_lots < raw_lots:
                raw_lots = margin_lots
                margin_capped = True

        lots = self._round_volume(raw_lots, volume_step)
        lots = max(0.0, min(lots, vmax))
        if 0 < lots < vmin:
            lots = 0.0

        msg = (
            f"mode={mode} frac={frac:.4f} risk_cap={risk_capital:.2f} "
            f"risk/lot={risk_per_lot:.2f} stop={stop_distance:.5f} "
            f"open_risk={open_reserved:.2f}"
        )
        if margin_capped:
            msg += " margin_capped"
        if lots <= 0:
            msg += " -> lots=0"

        return LotDecision(
            lots=float(lots),
            stop_loss=sl if lots > 0 else None,
            stop_distance=stop_distance,
            risk_capital=risk_capital,
            risk_per_lot=risk_per_lot,
            open_risk_reserved=open_reserved,
            margin_capped=margin_capped,
            message=msg,
        )

    @staticmethod
    def _round_volume(lots: float, step: float) -> float:
        if step <= 0:
            return lots
        return math.floor(lots / step + 1e-12) * step
