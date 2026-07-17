"""Half-Kelly position sizing from max loss at stop, plus cost model."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostModel:
    """Trading cost as fractions of notional.

    Components (all fractions, not bps):
    - ``spread_fraction``: full bid-ask width (charged **once** per round-trip)
    - ``commission_one_way``: commission each side
    - ``slippage_one_way``: assumed slippage each side (separate from FillModel price slip)
    - ``round_trip_floor``: minimum round-trip cost when live components are tiny/missing

    ``round_trip = max(floor, spread + 2*(commission + slippage))``
    ``one_way = round_trip / 2`` so entry+exit bar charges always equal the RT total.
    """

    spread_fraction: float = 0.0
    commission_one_way: float = 0.0
    slippage_one_way: float = 0.0
    round_trip_floor: float = 0.001  # 10 bps default floor on round-trip

    def _variable_round_trip(self) -> float:
        spread = max(0.0, float(self.spread_fraction))
        commission = max(0.0, float(self.commission_one_way))
        slippage = max(0.0, float(self.slippage_one_way))
        return spread + 2.0 * (commission + slippage)

    def round_trip_fraction(self) -> float:
        """Total cost for entry+exit as a fraction of notional."""
        return max(float(self.round_trip_floor), self._variable_round_trip())

    def one_way_fraction(self) -> float:
        """Half of round-trip (entry or exit leg)."""
        return 0.5 * self.round_trip_fraction()

    @classmethod
    def from_risk_config(cls, cfg: Optional[dict] = None) -> "CostModel":
        """Build from ``config.yaml`` ``risk:`` section (offline / no MT5 tick)."""
        cfg = cfg or {}
        # Prefer explicit fraction keys; accept bps aliases.
        floor = cls._fraction_from_cfg(
            cfg,
            frac_keys=("round_trip_floor",),
            bps_keys=("round_trip_floor_bps", "min_cost_bps"),
            default_bps=10.0,
        )
        commission = cls._fraction_from_cfg(
            cfg,
            frac_keys=("commission_one_way",),
            bps_keys=("commission_one_way_bps",),
            default_bps=0.0,
        )
        slippage = cls._fraction_from_cfg(
            cfg,
            frac_keys=("slippage_one_way",),
            bps_keys=("slippage_one_way_bps",),
            default_bps=0.0,
        )
        spread = cls._fraction_from_cfg(
            cfg,
            frac_keys=("spread_fraction",),
            bps_keys=("spread_bps",),
            default_bps=0.0,
        )
        return cls(
            spread_fraction=spread,
            commission_one_way=commission,
            slippage_one_way=slippage,
            round_trip_floor=floor,
        )

    @staticmethod
    def _fraction_from_cfg(
        cfg: dict,
        *,
        frac_keys: tuple[str, ...],
        bps_keys: tuple[str, ...],
        default_bps: float,
    ) -> float:
        for key in frac_keys:
            if key in cfg and cfg[key] is not None:
                return max(0.0, float(cfg[key]))
        for key in bps_keys:
            if key in cfg and cfg[key] is not None:
                return max(0.0, float(cfg[key]) / 10_000.0)
        return max(0.0, float(default_bps) / 10_000.0)

    @classmethod
    def from_symbol_info(
        cls,
        symbol_info: Any,
        tick: Any = None,
        *,
        risk_cfg: Optional[dict] = None,
        min_cost_bps: Optional[float] = None,
    ) -> "CostModel":
        """Live costs from MT5 symbol + optional risk config overrides."""
        base = cls.from_risk_config(risk_cfg)
        if min_cost_bps is not None:
            base.round_trip_floor = max(0.0, float(min_cost_bps) / 10_000.0)
        if symbol_info is None:
            return base

        spread_pts = float(getattr(symbol_info, "spread", 0) or 0)
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

        if spread_pts > 0 and point > 0 and price and price > 0:
            # Full bid-ask as fraction; counted once in round-trip.
            base.spread_fraction = (spread_pts * point) / price

        commission = None
        for attr in ("trade_commission", "commission"):
            val = getattr(symbol_info, attr, None)
            if val is not None:
                try:
                    commission = float(val)
                    break
                except (TypeError, ValueError):
                    pass
        if (
            commission is not None
            and commission > 0
            and contract > 0
            and price
            and price > 0
        ):
            notional = contract * price
            if notional > 0:
                base.commission_one_way = commission / notional

        logger.debug(
            "CostModel spread=%.6f commission_1w=%.6f slip_1w=%.6f "
            "one_way=%.6f round_trip=%.6f (floor=%.6f)",
            base.spread_fraction,
            base.commission_one_way,
            base.slippage_one_way,
            base.one_way_fraction(),
            base.round_trip_fraction(),
            base.round_trip_floor,
        )
        return base


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
    half_kelly: bool = False
    default_win_rate: float = 0.52
    default_reward_risk: float = 1.5
    max_fraction: float = 0.005
    max_lots: float = 5.0
    min_lots: float = 0.01
    lookback_trades: int = 50
    # Kelly stays off until this many closed trades; use cold_start_fraction instead.
    kelly_min_trades: int = 30
    cold_start_fraction: float = 0.005
    stop_pct: float = 0.005
    stop_points: Optional[float] = None
    gap_buffer_mult: float = 1.25
    max_open_risk_fraction: float = 0.01
    max_margin_fraction: float = 0.50
    recent_pnls: list[float] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "RiskManager":
        cfg = cfg or {}
        stop_points = cfg.get("stop_points")
        return cls(
            half_kelly=bool(cfg.get("half_kelly", False)),
            default_win_rate=float(cfg.get("default_win_rate", 0.52)),
            default_reward_risk=float(cfg.get("default_reward_risk", 1.5)),
            max_fraction=float(cfg.get("max_fraction", 0.005)),
            max_lots=float(cfg.get("max_lots", 5.0)),
            min_lots=float(cfg.get("min_lots", 0.01)),
            lookback_trades=int(cfg.get("lookback_trades", 50)),
            kelly_min_trades=int(cfg.get("kelly_min_trades", 30)),
            cold_start_fraction=float(cfg.get("cold_start_fraction", 0.005)),
            stop_pct=float(cfg.get("stop_pct", 0.005)),
            stop_points=float(stop_points) if stop_points is not None else None,
            gap_buffer_mult=float(cfg.get("gap_buffer_mult", 1.25)),
            max_open_risk_fraction=float(cfg.get("max_open_risk_fraction", 0.01)),
            max_margin_fraction=float(cfg.get("max_margin_fraction", 0.50)),
        )

    def record_trade(self, pnl: float) -> None:
        self.recent_pnls.append(float(pnl))
        if len(self.recent_pnls) > self.lookback_trades * 2:
            self.recent_pnls = self.recent_pnls[-self.lookback_trades :]

    def load_trade_history(self, pnls: list[float]) -> None:
        self.recent_pnls = [float(p) for p in pnls][-self.lookback_trades * 2 :]

    def empirical_trade_count(self) -> int:
        return len(self.recent_pnls[-self.lookback_trades :])

    def kelly_ready(self) -> bool:
        """True once enough closed trades exist to estimate W/R for Kelly."""
        return self.empirical_trade_count() >= max(0, int(self.kelly_min_trades))

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
        """f* = W - (1-W)/R ; half-Kelly = 0.5 * f*; capped by ``max_fraction``."""
        w = self.default_win_rate if win_rate is None else win_rate
        r = self.default_reward_risk if reward_risk is None else reward_risk
        if r <= 0:
            return 0.0
        full = w - (1.0 - w) / r
        frac = 0.5 * full if self.half_kelly else full
        frac = max(0.0, min(frac, self.max_fraction))
        return float(frac)

    def sizing_fraction(
        self,
        win_rate: Optional[float] = None,
        reward_risk: Optional[float] = None,
    ) -> tuple[float, str]:
        """Equity fraction to risk on the next trade, plus a mode label.

        Until ``kelly_min_trades`` closed trades accumulate, Kelly is disabled and
        ``cold_start_fraction`` is used (prototype / demo safe default).
        """
        if not self.kelly_ready():
            frac = max(0.0, min(float(self.cold_start_fraction), float(self.max_fraction)))
            return float(frac), "cold_start"
        if win_rate is None or reward_risk is None:
            ew, er = self.estimate_wr_rr()
            win_rate = win_rate if win_rate is not None else ew
            reward_risk = reward_risk if reward_risk is not None else er
        frac = self.kelly_fraction(win_rate, reward_risk)
        label = "half_kelly" if self.half_kelly else "kelly"
        return float(frac), label

    def stop_distance_price(self, price: float, point: float = 0.01) -> float:
        """Price distance from entry to stop (before gap buffer)."""
        if self.stop_points is not None and self.stop_points > 0 and point > 0:
            return float(self.stop_points) * float(point)
        if price <= 0 or self.stop_pct <= 0:
            return 0.0
        return float(price) * float(self.stop_pct)

    def buffered_stop_distance(self, price: float, point: float = 0.01) -> float:
        """Stop distance after ``gap_buffer_mult`` (sizing only; not fill price)."""
        base = self.stop_distance_price(price, point)
        return base * max(1.0, float(self.gap_buffer_mult))

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
        allow_notional_fallback: bool = False,
    ) -> LotDecision:
        """Size lots from Kelly risk capital / max loss at (gap-buffered) stop.

        Live trading must pass real ``tick_size`` / ``tick_value``. Notional
        sizing is research-only (``allow_notional_fallback=True`` for backtests).
        """
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

        # Explicit W/R from caller still uses Kelly path; otherwise cold-start
        # until enough closed trades exist (never size on default W/R alone).
        if win_rate is not None and reward_risk is not None:
            frac = self.kelly_fraction(win_rate, reward_risk)
            frac_mode = "half_kelly" if self.half_kelly else "kelly"
        else:
            frac, frac_mode = self.sizing_fraction()

        if frac <= 0:
            return LotDecision(
                lots=0.0,
                stop_loss=None,
                stop_distance=0.0,
                risk_capital=0.0,
                risk_per_lot=0.0,
                open_risk_reserved=max(0.0, float(open_risk)),
                message=f"{frac_mode} fraction=0",
            )

        point_v = float(point) if point and point > 0 else 0.01
        stop_distance = self.buffered_stop_distance(price, point_v)
        sl = self.stop_loss_price(
            side_long=side_long,
            entry_price=price,
            stop_distance=stop_distance,
            digits=digits,
        )

        risk_budget = equity * frac
        open_reserved = max(0.0, float(open_risk))
        total_cap = equity * float(self.max_open_risk_fraction)
        risk_capital = min(risk_budget, max(0.0, total_cap - open_reserved))

        ts = float(tick_size) if tick_size and tick_size > 0 else 0.0
        tv = float(tick_value) if tick_value and tick_value > 0 else 0.0
        risk_per_lot = self.loss_per_lot(stop_distance, ts, tv)

        if risk_per_lot > 0 and risk_capital > 0:
            raw_lots = risk_capital / risk_per_lot
            mode = "max_loss"
        elif allow_notional_fallback and contract_size > 0:
            notional_per_lot = price * contract_size
            raw_lots = risk_budget / notional_per_lot if notional_per_lot > 0 else 0.0
            mode = "notional_fallback"
            logger.warning(
                "position_lots using notional fallback (research/backtest only); "
                "tick_size/value missing -- stop-based risk unavailable"
            )
        else:
            return LotDecision(
                lots=0.0,
                stop_loss=None,
                stop_distance=stop_distance,
                risk_capital=risk_capital,
                risk_per_lot=0.0,
                open_risk_reserved=open_reserved,
                message=(
                    "tick_size/tick_value unavailable; lots=0 (notional fallback disabled)"
                ),
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
            f"mode={mode} sizing={frac_mode} frac={frac:.4f} "
            f"risk_cap={risk_capital:.2f} "
            f"risk/lot={risk_per_lot:.2f} stop={stop_distance:.5f} "
            f"open_risk={open_reserved:.2f} trades={self.empirical_trade_count()}"
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
