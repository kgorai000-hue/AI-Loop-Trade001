"""Half-Kelly position sizing and trading cost model."""

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
            # MT5 spread is typically in points; cost ≈ spread * point / price
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

        # Commission fields vary by broker; try common attributes
        commission = None
        for attr in ("trade_commission", "commission", "trade_tick_value"):
            # Prefer explicit commission if present; skip tick_value misuse
            if attr == "trade_tick_value":
                continue
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
class RiskManager:
    half_kelly: bool = True
    default_win_rate: float = 0.52
    default_reward_risk: float = 1.5
    max_fraction: float = 0.25
    max_lots: float = 5.0
    min_lots: float = 0.01
    lookback_trades: int = 50
    min_cost_bps: float = 10.0
    recent_pnls: list[float] = field(default_factory=list)

    def record_trade(self, pnl: float) -> None:
        self.recent_pnls.append(float(pnl))
        if len(self.recent_pnls) > self.lookback_trades * 2:
            self.recent_pnls = self.recent_pnls[-self.lookback_trades :]

    def estimate_wr_rr(self) -> tuple[float, float]:
        pnls = self.recent_pnls[-self.lookback_trades :]
        if len(pnls) < 5:
            return self.default_win_rate, self.default_reward_risk
        arr = [p for p in pnls]
        wins = [p for p in arr if p > 0]
        losses = [p for p in arr if p < 0]
        wr = len(wins) / len(arr)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        rr = avg_win / avg_loss if avg_loss > 0 else self.default_reward_risk
        rr = max(0.1, rr)
        return float(wr), float(rr)

    def kelly_fraction(self, win_rate: Optional[float] = None, reward_risk: Optional[float] = None) -> float:
        """f* = W - (1-W)/R ; half-Kelly = 0.5 * f*."""
        w = self.default_win_rate if win_rate is None else win_rate
        r = self.default_reward_risk if reward_risk is None else reward_risk
        if r <= 0:
            return 0.0
        full = w - (1.0 - w) / r
        frac = 0.5 * full if self.half_kelly else full
        frac = max(0.0, min(frac, self.max_fraction))
        return float(frac)

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
    ) -> float:
        """Convert Kelly fraction of equity into lot size."""
        if equity <= 0 or price <= 0 or contract_size <= 0:
            return 0.0
        if win_rate is None or reward_risk is None:
            ew, er = self.estimate_wr_rr()
            win_rate = win_rate if win_rate is not None else ew
            reward_risk = reward_risk if reward_risk is not None else er

        frac = self.kelly_fraction(win_rate, reward_risk)
        if frac <= 0:
            return 0.0

        risk_capital = equity * frac
        notional_per_lot = price * contract_size
        raw_lots = risk_capital / notional_per_lot

        vmin = volume_min if volume_min is not None else self.min_lots
        vmax = volume_max if volume_max is not None else self.max_lots
        vmax = min(vmax, self.max_lots)
        vmin = max(vmin, self.min_lots)

        lots = self._round_volume(raw_lots, volume_step)
        lots = max(0.0, min(lots, vmax))
        if 0 < lots < vmin:
            lots = 0.0  # below minimum → skip trade
        return float(lots)

    @staticmethod
    def _round_volume(lots: float, step: float) -> float:
        if step <= 0:
            return lots
        return math.floor(lots / step + 1e-12) * step
