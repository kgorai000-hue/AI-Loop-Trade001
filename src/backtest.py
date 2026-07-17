"""Event-driven backtester with limit fills and account-level lot sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .metrics import BARS_PER_YEAR_M30, PerformanceReport, build_report
from .risk import CostModel, RiskManager
from .strategy import RegressionStrategy, StrategyParams


@dataclass
class FillModel:
    """Live-aligned limit entry assumptions for backtests."""

    enabled: bool = True
    ttl_bars: int = 2
    slippage_bps: float = 1.0
    limit_offset_bps: float = 0.0

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "FillModel":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("limit_fills", True)),
            ttl_bars=max(1, int(cfg.get("limit_ttl_bars", 2))),
            slippage_bps=float(cfg.get("slippage_bps", 1.0)),
            limit_offset_bps=float(cfg.get("limit_offset_bps", 0.0)),
        )

    @property
    def slippage_fraction(self) -> float:
        return max(0.0, self.slippage_bps) / 10_000.0

    @property
    def limit_offset_fraction(self) -> float:
        return max(0.0, self.limit_offset_bps) / 10_000.0


@dataclass
class SymbolSpec:
    """Contract metadata for account-currency PnL and margin (MT5-aligned)."""

    tick_size: float = 1.0
    tick_value: float = 1.0
    contract_size: float = 1.0
    volume_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 5.0
    point: float = 1.0
    digits: int = 2
    margin_per_lot: Optional[float] = None
    leverage: float = 100.0

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "SymbolSpec":
        cfg = cfg or {}
        margin = cfg.get("margin_per_lot")
        return cls(
            tick_size=float(cfg.get("tick_size", 1.0)),
            tick_value=float(cfg.get("tick_value", 1.0)),
            contract_size=float(cfg.get("contract_size", 1.0)),
            volume_step=float(cfg.get("volume_step", 0.01)),
            volume_min=float(cfg.get("volume_min", 0.01)),
            volume_max=float(cfg.get("volume_max", 5.0)),
            point=float(cfg.get("point", 1.0)),
            digits=int(cfg.get("digits", 2)),
            margin_per_lot=float(margin) if margin is not None else None,
            leverage=float(cfg.get("leverage", 100.0)),
        )

    @classmethod
    def from_mt5_info(cls, info: Any, *, margin_per_lot: Optional[float] = None) -> "SymbolSpec":
        point = float(getattr(info, "point", 0.01) or 0.01)
        tick_size = float(getattr(info, "trade_tick_size", 0) or 0) or point
        return cls(
            tick_size=tick_size,
            tick_value=float(getattr(info, "trade_tick_value", 0) or 0) or 1.0,
            contract_size=float(getattr(info, "trade_contract_size", 1) or 1),
            volume_step=float(getattr(info, "volume_step", 0.01) or 0.01),
            volume_min=float(getattr(info, "volume_min", 0.01) or 0.01),
            volume_max=float(getattr(info, "volume_max", 5) or 5),
            point=point,
            digits=int(getattr(info, "digits", 2) or 2),
            margin_per_lot=margin_per_lot,
            leverage=100.0,
        )

    def margin_for_price(self, price: float) -> float:
        if self.margin_per_lot is not None and self.margin_per_lot > 0:
            return float(self.margin_per_lot)
        notional = max(0.0, float(price) * float(self.contract_size))
        if self.leverage <= 0:
            return notional
        return notional / float(self.leverage)

    def price_pnl(self, lots: float, side: int, start: float, end: float) -> float:
        if lots <= 0 or side == 0:
            return 0.0
        delta = float(end) - float(start)
        if self.tick_size > 0 and self.tick_value > 0:
            return float(side) * float(lots) * (delta / self.tick_size) * self.tick_value
        return float(side) * float(lots) * float(self.contract_size) * delta

    def cost_money(self, lots: float, price: float, one_way_frac: float) -> float:
        if lots <= 0 or price <= 0 or one_way_frac <= 0:
            return 0.0
        move = float(price) * float(one_way_frac)
        if self.tick_size > 0 and self.tick_value > 0:
            return float(lots) * (move / self.tick_size) * self.tick_value
        return float(lots) * float(self.contract_size) * float(price) * float(one_way_frac)


@dataclass
class AccountConfig:
    """Account-level simulation so Validator DD matches live equity DD."""

    enabled: bool = False
    initial_equity: float = 10_000.0
    stop_out_level: float = 0.50

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "AccountConfig":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("account_sizing", True)),
            initial_equity=float(cfg.get("initial_equity", 10_000.0)),
            stop_out_level=float(cfg.get("stop_out_level", 0.50)),
        )


@dataclass
class BacktestResult:
    report: PerformanceReport
    trades: list[dict]
    bar_returns: pd.Series
    signals: pd.Series
    params: StrategyParams
    unfilled_entries: int = 0
    liquidations: int = 0
    skipped_entries: int = 0
    fill_model: Optional[FillModel] = None
    account_config: Optional[AccountConfig] = None
    final_equity: Optional[float] = None


class Backtester:
    """
    Bar-close signal backtester with optional limit-entry and account sizing.

    Unit mode (account.enabled=False): position ±1, fractional price returns.
    Account mode: Kelly lots, account-currency PnL, margin, SL, stop-out.
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        initial_equity: float = 1.0,
        periods_per_year: float = BARS_PER_YEAR_M30,
        fill_model: Optional[FillModel] = None,
        account: Optional[AccountConfig] = None,
        symbol_spec: Optional[SymbolSpec] = None,
        risk: Optional[RiskManager] = None,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.fill_model = fill_model or FillModel()
        self.account = account or AccountConfig(enabled=False, initial_equity=initial_equity)
        if account is None and initial_equity != 1.0:
            self.account.initial_equity = float(initial_equity)
        self.initial_equity = (
            float(self.account.initial_equity) if self.account.enabled else float(initial_equity)
        )
        self.periods_per_year = periods_per_year
        self.symbol_spec = symbol_spec or SymbolSpec()
        self.risk = risk or RiskManager()

    @classmethod
    def from_app_config(
        cls,
        app_config: dict[str, Any],
        *,
        cost_model: Optional[CostModel] = None,
        risk: Optional[RiskManager] = None,
        symbol_spec: Optional[SymbolSpec] = None,
        initial_equity: Optional[float] = None,
    ) -> "Backtester":
        bcfg = app_config.get("backtest") or {}
        account = AccountConfig.from_config(bcfg)
        if initial_equity is not None and initial_equity > 0:
            account.initial_equity = float(initial_equity)
        return cls(
            cost_model=cost_model
            or CostModel(min_cost_bps=float(app_config.get("risk", {}).get("min_cost_bps", 10))),
            fill_model=FillModel.from_config(bcfg),
            account=account,
            symbol_spec=symbol_spec or SymbolSpec.from_config(bcfg),
            risk=risk or RiskManager.from_config(app_config.get("risk")),
        )

    def run(
        self,
        df: pd.DataFrame,
        params: Optional[StrategyParams] = None,
        strategy: Optional[RegressionStrategy] = None,
    ) -> BacktestResult:
        if not self.fill_model.enabled:
            return self._run_immediate(df, params=params, strategy=strategy)
        return self._run_limit(df, params=params, strategy=strategy)

    def _prepare(
        self,
        df: pd.DataFrame,
        params: Optional[StrategyParams],
        strategy: Optional[RegressionStrategy],
    ) -> tuple[StrategyParams, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = params or StrategyParams()
        strategy = strategy or RegressionStrategy(params)
        strategy.update_params(params)
        annotated = strategy.signal_series(df)
        closes = annotated["close"].to_numpy(dtype=float)
        highs, lows = self._high_low(annotated, closes)
        signals = annotated["signal"].to_numpy(dtype=int)
        n = len(annotated)
        fwd = np.zeros(n, dtype=float)
        fwd[:-1] = closes[1:] / closes[:-1] - 1.0
        return params, annotated, closes, highs, lows, signals, fwd

    @staticmethod
    def _high_low(annotated: pd.DataFrame, closes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if "high" in annotated.columns and "low" in annotated.columns:
            highs = annotated["high"].to_numpy(dtype=float)
            lows = annotated["low"].to_numpy(dtype=float)
            highs = np.maximum(highs, closes)
            lows = np.minimum(lows, closes)
            return highs, lows
        return closes.copy(), closes.copy()

    def _limit_price(self, side: int, close: float) -> float:
        off = self.fill_model.limit_offset_fraction
        return float(close * (1.0 - off)) if side > 0 else float(close * (1.0 + off))

    def _fill_price(self, side: int, limit: float) -> float:
        slip = self.fill_model.slippage_fraction
        return float(limit * (1.0 + slip)) if side > 0 else float(limit * (1.0 - slip))

    @staticmethod
    def _limit_touched(side: int, limit: float, high: float, low: float) -> bool:
        return low <= limit if side > 0 else high >= limit

    def _market_exit_price(self, side: int, close: float) -> float:
        slip = self.fill_model.slippage_fraction
        return float(close * (1.0 - slip)) if side > 0 else float(close * (1.0 + slip))

    def _sl_hit_price(
        self, side: int, stop_loss: Optional[float], high: float, low: float
    ) -> Optional[float]:
        if stop_loss is None or stop_loss <= 0:
            return None
        if side > 0 and low <= stop_loss:
            return float(stop_loss)
        if side < 0 and high >= stop_loss:
            return float(stop_loss)
        return None

    def _size_entry(self, *, equity: float, price: float, side: int) -> tuple[float, Optional[float], float]:
        if not self.account.enabled:
            return 1.0, None, 0.0
        if equity <= 0 or price <= 0:
            return 0.0, None, 0.0
        spec = self.symbol_spec
        margin_per = spec.margin_for_price(price)
        decision = self.risk.position_lots(
            equity=equity,
            price=price,
            contract_size=spec.contract_size,
            volume_step=spec.volume_step,
            volume_min=spec.volume_min,
            volume_max=min(spec.volume_max, self.risk.max_lots),
            tick_size=spec.tick_size,
            tick_value=spec.tick_value,
            point=spec.point,
            side_long=side > 0,
            digits=spec.digits,
            free_margin=equity if equity > 0 else None,
            margin_per_lot=margin_per if margin_per > 0 else None,
            open_risk=0.0,
        )
        if decision.lots <= 0:
            return 0.0, None, 0.0
        return float(decision.lots), decision.stop_loss, float(decision.lots) * margin_per

    def _apply_delta(self, equity: float, bar_rets: np.ndarray, i: int, delta: float) -> float:
        if equity > 0:
            bar_rets[i] += delta / equity
        return equity + delta

    def _record_close(
        self,
        trades: list[dict],
        trade_pnls: list[float],
        *,
        side: int,
        lots: float,
        entry_price: float,
        entry_i: int,
        exit_price: float,
        exit_i: int,
        fill: str,
        entry_cost: float,
        exit_cost: float,
    ) -> None:
        if self.account.enabled:
            gross = self.symbol_spec.price_pnl(lots, side, entry_price, exit_price)
            pnl = gross - entry_cost - exit_cost
        else:
            cost_rt = self.cost_model.round_trip_fraction()
            pnl = side * (exit_price / entry_price - 1.0) - cost_rt
        trade_pnls.append(pnl)
        if trades and trades[-1].get("exit_i") is None and trades[-1].get("side") == side:
            trades[-1].update(
                {
                    "exit_i": exit_i,
                    "exit": exit_price,
                    "pnl": pnl,
                    "lots": lots,
                    "exit_fill": fill,
                }
            )
        else:
            trades.append(
                {
                    "entry_i": entry_i,
                    "exit_i": exit_i,
                    "side": side,
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl": pnl,
                    "lots": lots,
                    "fill": fill,
                    "exit_fill": fill,
                }
            )

    def _run_limit(
        self,
        df: pd.DataFrame,
        params: Optional[StrategyParams] = None,
        strategy: Optional[RegressionStrategy] = None,
    ) -> BacktestResult:
        params, annotated, closes, highs, lows, signals, fwd = self._prepare(
            df, params, strategy
        )
        n = len(annotated)
        max_hold = params.max_hold_bars
        cost_one = self.cost_model.one_way_fraction()
        ttl = self.fill_model.ttl_bars
        acct = self.account.enabled
        equity = float(self.account.initial_equity) if acct else 1.0
        stop_out = float(self.account.stop_out_level)

        bar_rets = np.zeros(n, dtype=float)
        trades: list[dict] = []
        trade_pnls: list[float] = []
        unfilled = 0
        liquidations = 0
        skipped = 0

        position = 0
        lots = 0.0
        used_margin = 0.0
        stop_loss: Optional[float] = None
        mark_price = 0.0
        hold = 0
        entry_price = 0.0
        entry_i = -1
        entry_cost = 0.0

        pending_side = 0
        pending_limit = 0.0
        pending_age = 0

        def flat() -> None:
            nonlocal position, lots, used_margin, stop_loss, mark_price
            nonlocal hold, entry_price, entry_i, entry_cost
            position = 0
            lots = 0.0
            used_margin = 0.0
            stop_loss = None
            mark_price = 0.0
            hold = 0
            entry_price = 0.0
            entry_i = -1
            entry_cost = 0.0

        def exit_now(i: int, exit_price: float, fill: str, *, liq: bool = False) -> None:
            nonlocal equity, liquidations
            if position == 0:
                return
            if acct:
                exit_cost = self.symbol_spec.cost_money(lots, exit_price, cost_one)
                move = self.symbol_spec.price_pnl(lots, position, mark_price, exit_price)
                equity = self._apply_delta(equity, bar_rets, i, move - exit_cost)
                self._record_close(
                    trades,
                    trade_pnls,
                    side=position,
                    lots=lots,
                    entry_price=entry_price,
                    entry_i=entry_i,
                    exit_price=exit_price,
                    exit_i=i,
                    fill=fill,
                    entry_cost=entry_cost,
                    exit_cost=exit_cost,
                )
            else:
                bar_rets[i] -= cost_one
                self._record_close(
                    trades,
                    trade_pnls,
                    side=position,
                    lots=1.0,
                    entry_price=entry_price,
                    entry_i=entry_i,
                    exit_price=exit_price,
                    exit_i=i,
                    fill=fill,
                    entry_cost=0.0,
                    exit_cost=0.0,
                )
            if liq:
                liquidations += 1
            flat()

        for i in range(n - 1):
            if pending_side != 0:
                pending_age += 1
                if self._limit_touched(pending_side, pending_limit, highs[i], lows[i]):
                    fill_px = self._fill_price(pending_side, pending_limit)
                    sized, sl, margin = self._size_entry(
                        equity=equity, price=fill_px, side=pending_side
                    )
                    if sized <= 0:
                        skipped += 1
                    else:
                        position = pending_side
                        lots = sized
                        used_margin = margin
                        stop_loss = sl
                        entry_price = fill_px
                        mark_price = fill_px
                        entry_i = i
                        hold = 0
                        if acct:
                            entry_cost = self.symbol_spec.cost_money(lots, entry_price, cost_one)
                            equity = self._apply_delta(equity, bar_rets, i, -entry_cost)
                        else:
                            entry_cost = 0.0
                            bar_rets[i] -= cost_one
                        trades.append(
                            {
                                "entry_i": entry_i,
                                "exit_i": None,
                                "side": position,
                                "entry": entry_price,
                                "exit": None,
                                "pnl": None,
                                "lots": lots,
                                "limit": pending_limit,
                                "stop_loss": stop_loss,
                                "fill": "limit",
                            }
                        )
                    pending_side = 0
                    pending_limit = 0.0
                    pending_age = 0
                elif pending_age >= ttl:
                    unfilled += 1
                    pending_side = 0
                    pending_limit = 0.0
                    pending_age = 0

            if position != 0:
                sl_px = self._sl_hit_price(position, stop_loss, highs[i], lows[i])
                if sl_px is not None:
                    exit_now(i, sl_px, "stop_loss")

            if (
                acct
                and position != 0
                and used_margin > 0
                and equity / used_margin < stop_out
            ):
                exit_now(i, self._market_exit_price(position, closes[i]), "stop_out", liq=True)

            desired = int(signals[i])
            if position != 0:
                hold += 1
                if hold >= max_hold:
                    desired = 0

            if position != 0 and desired != position:
                exit_now(i, self._market_exit_price(position, closes[i]), "market_exit")
                if pending_side != 0 and pending_side != desired:
                    pending_side = 0
                    pending_limit = 0.0
                    pending_age = 0

            if desired == 0 and pending_side != 0:
                pending_side = 0
                pending_limit = 0.0
                pending_age = 0

            if position == 0 and pending_side == 0 and desired != 0:
                pending_side = desired
                pending_limit = self._limit_price(desired, closes[i])
                pending_age = 0

            if position != 0:
                if acct:
                    move = self.symbol_spec.price_pnl(
                        lots, position, mark_price, closes[i + 1]
                    )
                    equity = self._apply_delta(equity, bar_rets, i, move)
                    mark_price = float(closes[i + 1])
                    if used_margin > 0 and equity / used_margin < stop_out:
                        exit_now(
                            i + 1,
                            self._market_exit_price(position, closes[i + 1]),
                            "stop_out",
                            liq=True,
                        )
                else:
                    bar_rets[i] += position * fwd[i]

        if pending_side != 0:
            unfilled += 1
        if position != 0 and entry_i >= 0:
            exit_now(
                n - 1,
                self._market_exit_price(position, closes[-1]),
                "market_exit",
            )

        return self._finalize(
            params=params,
            annotated=annotated,
            signals=signals,
            bar_rets=bar_rets,
            trades=trades,
            trade_pnls=trade_pnls,
            unfilled=unfilled,
            liquidations=liquidations,
            skipped=skipped,
            equity=equity if acct else None,
        )

    def _run_immediate(
        self,
        df: pd.DataFrame,
        params: Optional[StrategyParams] = None,
        strategy: Optional[RegressionStrategy] = None,
    ) -> BacktestResult:
        params, annotated, closes, highs, lows, signals, fwd = self._prepare(
            df, params, strategy
        )
        n = len(annotated)
        max_hold = params.max_hold_bars
        cost_one = self.cost_model.one_way_fraction()
        acct = self.account.enabled
        equity = float(self.account.initial_equity) if acct else 1.0
        stop_out = float(self.account.stop_out_level)

        bar_rets = np.zeros(n, dtype=float)
        trades: list[dict] = []
        trade_pnls: list[float] = []
        liquidations = 0
        skipped = 0

        position = 0
        lots = 0.0
        used_margin = 0.0
        stop_loss: Optional[float] = None
        mark_price = 0.0
        hold = 0
        entry_price = 0.0
        entry_i = -1
        entry_cost = 0.0

        def flat() -> None:
            nonlocal position, lots, used_margin, stop_loss, mark_price
            nonlocal hold, entry_price, entry_i, entry_cost
            position = 0
            lots = 0.0
            used_margin = 0.0
            stop_loss = None
            mark_price = 0.0
            hold = 0
            entry_price = 0.0
            entry_i = -1
            entry_cost = 0.0

        def exit_now(i: int, exit_price: float, fill: str, *, liq: bool = False) -> None:
            nonlocal equity, liquidations
            if position == 0:
                return
            if acct:
                exit_cost = self.symbol_spec.cost_money(lots, exit_price, cost_one)
                move = self.symbol_spec.price_pnl(lots, position, mark_price, exit_price)
                equity = self._apply_delta(equity, bar_rets, i, move - exit_cost)
                self._record_close(
                    trades,
                    trade_pnls,
                    side=position,
                    lots=lots,
                    entry_price=entry_price,
                    entry_i=entry_i,
                    exit_price=exit_price,
                    exit_i=i,
                    fill=fill,
                    entry_cost=entry_cost,
                    exit_cost=exit_cost,
                )
            else:
                bar_rets[i] -= cost_one
                self._record_close(
                    trades,
                    trade_pnls,
                    side=position,
                    lots=1.0,
                    entry_price=entry_price,
                    entry_i=entry_i,
                    exit_price=exit_price,
                    exit_i=i,
                    fill=fill,
                    entry_cost=0.0,
                    exit_cost=0.0,
                )
            if liq:
                liquidations += 1
            flat()

        for i in range(n - 1):
            if position != 0:
                sl_px = self._sl_hit_price(position, stop_loss, highs[i], lows[i])
                if sl_px is not None:
                    exit_now(i, sl_px, "stop_loss")

            if (
                acct
                and position != 0
                and used_margin > 0
                and equity / used_margin < stop_out
            ):
                exit_now(i, closes[i], "stop_out", liq=True)

            desired = int(signals[i])
            if position != 0:
                hold += 1
                if hold >= max_hold:
                    desired = 0

            if desired != position:
                if position != 0 and entry_i >= 0:
                    exit_now(i, closes[i], "immediate")

                if desired != 0:
                    sized, sl, margin = self._size_entry(
                        equity=equity, price=closes[i], side=desired
                    )
                    if sized <= 0:
                        skipped += 1
                        flat()
                    else:
                        position = desired
                        lots = sized
                        used_margin = margin
                        stop_loss = sl
                        entry_price = closes[i]
                        mark_price = entry_price
                        entry_i = i
                        hold = 0
                        if acct:
                            entry_cost = self.symbol_spec.cost_money(
                                lots, entry_price, cost_one
                            )
                            equity = self._apply_delta(equity, bar_rets, i, -entry_cost)
                        else:
                            entry_cost = 0.0
                            bar_rets[i] -= cost_one

            if position != 0:
                if acct:
                    move = self.symbol_spec.price_pnl(
                        lots, position, mark_price, closes[i + 1]
                    )
                    equity = self._apply_delta(equity, bar_rets, i, move)
                    mark_price = float(closes[i + 1])
                    if used_margin > 0 and equity / used_margin < stop_out:
                        exit_now(i + 1, closes[i + 1], "stop_out", liq=True)
                else:
                    bar_rets[i] += position * fwd[i]

        if position != 0 and entry_i >= 0:
            exit_now(n - 1, closes[-1], "immediate")

        return self._finalize(
            params=params,
            annotated=annotated,
            signals=signals,
            bar_rets=bar_rets,
            trades=trades,
            trade_pnls=trade_pnls,
            unfilled=0,
            liquidations=liquidations,
            skipped=skipped,
            equity=equity if acct else None,
        )

    def _finalize(
        self,
        *,
        params: StrategyParams,
        annotated: pd.DataFrame,
        signals: np.ndarray,
        bar_rets: np.ndarray,
        trades: list[dict],
        trade_pnls: list[float],
        unfilled: int,
        liquidations: int,
        skipped: int,
        equity: Optional[float],
    ) -> BacktestResult:
        closed = [t for t in trades if t.get("exit_i") is not None and t.get("pnl") is not None]
        warmup = params.long_window
        bar_series = pd.Series(bar_rets, index=annotated.index)
        bar_series.iloc[:warmup] = 0.0
        sig_series = pd.Series(signals, index=annotated.index)
        report_initial = float(self.account.initial_equity) if self.account.enabled else 1.0
        report = build_report(
            bar_series,
            signals=sig_series,
            trade_pnls=trade_pnls,
            periods_per_year=self.periods_per_year,
            initial_equity=report_initial,
        )
        return BacktestResult(
            report=report,
            trades=closed,
            bar_returns=bar_series,
            signals=sig_series,
            params=params,
            unfilled_entries=unfilled,
            liquidations=liquidations,
            skipped_entries=skipped,
            fill_model=self.fill_model,
            account_config=self.account,
            final_equity=equity,
        )

    def run_is_oos(
        self,
        df: pd.DataFrame,
        params: StrategyParams,
        is_fraction: float = 0.70,
    ) -> tuple[BacktestResult, BacktestResult, float]:
        from .metrics import oos_degradation

        n = len(df)
        split = max(params.long_window + 10, int(n * is_fraction))
        split = min(split, n - max(20, params.short_window))
        is_df = df.iloc[:split].reset_index(drop=True)
        oos_df = df.iloc[split:].reset_index(drop=True)

        warmup = params.long_window
        report_initial = float(self.account.initial_equity) if self.account.enabled else 1.0
        if len(is_df) >= warmup and len(oos_df) > 0:
            oos_with_warm = pd.concat([is_df.iloc[-warmup:], oos_df], ignore_index=True)
            oos_full = self.run(oos_with_warm, params=params)
            oos_rets = oos_full.bar_returns.iloc[warmup:].reset_index(drop=True)
            oos_sigs = oos_full.signals.iloc[warmup:].reset_index(drop=True)
            oos_trades = [t for t in oos_full.trades if t["entry_i"] >= warmup]
            oos_report = build_report(
                oos_rets,
                signals=oos_sigs,
                trade_pnls=[t["pnl"] for t in oos_trades],
                periods_per_year=self.periods_per_year,
                initial_equity=report_initial,
            )
            oos_result = BacktestResult(
                report=oos_report,
                trades=oos_trades,
                bar_returns=oos_rets,
                signals=oos_sigs,
                params=params,
                unfilled_entries=oos_full.unfilled_entries,
                liquidations=oos_full.liquidations,
                skipped_entries=oos_full.skipped_entries,
                fill_model=self.fill_model,
                account_config=self.account,
                final_equity=oos_full.final_equity,
            )
        else:
            oos_result = self.run(oos_df, params=params)

        is_result = self.run(is_df, params=params)
        deg = oos_degradation(is_result.report.sharpe, oos_result.report.sharpe)
        return is_result, oos_result, deg
