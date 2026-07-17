from __future__ import annotations

from types import SimpleNamespace

import MetaTrader5 as mt5
import pandas as pd

from src.strategy import Regime, Signal, StrategyDecision, StrategyParams
from src.symbol_trader import SymbolConfig, SymbolTrader


class Store:
    def __init__(self):
        self.state = {"locked": False, "equity_peak": 1000.0}

    def get_params(self):
        return StrategyParams(long_window=3, short_window=2, max_hold_bars=2)

    def update_state(self, **kwargs):
        self.state.update(kwargs)

    def read_state(self):
        return dict(self.state)

    def is_locked(self):
        return bool(self.state.get("locked"))


class Connection:
    def symbol_info(self, symbol):
        return SimpleNamespace(
            volume_step=0.01,
            volume_min=0.01,
            volume_max=5.0,
            trade_contract_size=1.0,
            point=0.1,
            trade_tick_size=0.1,
            trade_tick_value=1.0,
            digits=1,
            trade_stops_level=0,
            bid=100.0,
        )

    def account_info(self):
        return SimpleNamespace(equity=1000.0, margin=0.0, margin_free=1000.0)

    def order_calc_margin(self, order_type, symbol, volume, price):
        return 100.0

    def ensure(self):
        return True


class Risk:
    def position_lots(self, **kwargs):
        from src.risk import LotDecision

        return LotDecision(
            lots=1.0,
            stop_loss=95.0,
            stop_distance=5.0,
            risk_capital=10.0,
            risk_per_lot=10.0,
            message="test",
        )

    def load_trade_history(self, pnls):
        pass

    def record_trade(self, pnl):
        pass

    def stop_distance_price(self, price, point=0.01):
        return price * 0.005

    gap_buffer_mult = 1.25

    def estimate_position_open_risk(self, **kwargs):
        return 0.0


class Feed:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def copy_closed_rates(self, count):
        self.calls.append(count)
        return self.frame.tail(count).reset_index(drop=True)

    def tick(self):
        return SimpleNamespace(bid=100.0, ask=101.0)


class Executor:
    def __init__(self, positions=None):
        self.positions = positions or []
        self.targets = []

    def managed_positions(self, symbol, magic=260717):
        return list(self.positions)

    def reconcile_target(self, **kwargs):
        self.targets.append(kwargs)
        return SimpleNamespace(as_dict=lambda: {"ok": True, "action": "test"}, orders=[])


def _trader(frame, positions=None):
    executor = Executor(positions=positions)
    trader = SymbolTrader(
        SymbolConfig(
            "#US30",
            "US30",
            timeframe="M30",
            long_window=3,
            short_window=2,
            max_hold_bars=2,
        ),
        Connection(),
        executor,
        Risk(),
        Store(),
        {},
    )
    trader.feed = Feed(frame)
    return trader, executor


def test_on_new_bar_uses_same_completed_bar_for_detection_and_decision():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=8, freq="30min", tz="UTC"),
            "close": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    trader, executor = _trader(frame)

    output = trader.on_new_bar()

    assert trader.feed.calls == [1, 8]
    assert output["bar_time"] == str(frame["time"].iloc[-1])
    assert executor.targets[-1]["side"] == Signal.LONG
    assert executor.targets[-1]["sl"] == 95.0
    assert trader.store.state["last_processed_bar"] == frame["time"].iloc[-1].isoformat()


def test_on_new_bar_skips_already_processed_bar_after_restart():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=8, freq="30min", tz="UTC"),
            "close": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    store = Store()
    store.state["last_processed_bar"] = frame["time"].iloc[-1].isoformat()
    trader, _ = _trader(frame)
    trader.store = store
    trader._last_bar_time = trader._load_last_processed_bar()

    assert trader.on_new_bar() is None
    assert trader.feed.calls == [1]


def test_on_new_bar_does_not_advance_when_trade_fails():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=8, freq="30min", tz="UTC"),
            "close": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    trader, _ = _trader(frame)
    trader.feed.tick = lambda: None

    out = trader.on_new_bar()
    assert out is not None
    assert out["order"]["ok"] is False
    assert trader.store.state.get("last_processed_bar") is None
    assert trader._last_bar_time is None

    # Retry succeeds after market data returns.
    trader.feed.tick = lambda: SimpleNamespace(bid=100.0, ask=101.0)
    out2 = trader.on_new_bar()
    assert out2 is not None
    assert out2["order"]["ok"] is True
    assert trader.store.state["last_processed_bar"] == frame["time"].iloc[-1].isoformat()


def test_max_hold_forces_flat_target():
    bar_time = pd.Timestamp("2026-01-01T02:00:00Z")
    position = SimpleNamespace(
        type=mt5.POSITION_TYPE_BUY,
        volume=1.0,
        ticket=42,
        magic=260717,
        time=int(pd.Timestamp("2026-01-01T01:00:00Z").timestamp()),
    )
    frame = pd.DataFrame({"time": [bar_time], "close": [100.0]})
    trader, executor = _trader(frame, positions=[position])
    decision = StrategyDecision(
        signal=Signal.LONG,
        regime=Regime.TREND,
        b_long=1.0,
        b_short=1.0,
        bar_time=bar_time,
    )

    result = trader.maybe_trade(decision)

    assert result["forced_flat"] == "max_hold_bars"
    assert executor.targets[-1]["side"] == Signal.FLAT
    assert executor.targets[-1]["volume"] == 0.0
    assert trader.store.state["equity"] == 1000.0
