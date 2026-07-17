from __future__ import annotations

from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from src.data import DataFeed


class Connection:
    def ensure(self):
        return True

    def symbol_info(self, symbol):
        return object()

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        return mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)

    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        return mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    def symbol_info_tick(self, symbol):
        return mt5.symbol_info_tick(symbol)

    def last_error(self):
        return mt5.last_error()


def test_copy_closed_rates_excludes_forming_bar(monkeypatch):
    seen = {}

    def fake_copy(symbol, timeframe, start_pos, count):
        seen.update(start_pos=start_pos, count=count)
        return [
            {
                "time": 1_700_000_000,
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "tick_volume": 1,
                "spread": 1,
                "real_volume": 1,
            }
        ]

    monkeypatch.setattr(mt5, "copy_rates_from_pos", fake_copy)
    feed = DataFeed(Connection(), "#US30", "M30")
    frame = feed.copy_closed_rates(7)

    assert seen == {"start_pos": 1, "count": 7}
    assert len(frame) == 1


def test_drop_forming_bars_keeps_only_completed():
    feed = DataFeed(Connection(), "#US30", "M30")
    # as_of = 12:15 UTC → last closed M30 open is 11:30
    as_of = datetime(2024, 1, 1, 12, 15, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-01-01T11:00:00Z",
                    "2024-01-01T11:30:00Z",
                    "2024-01-01T12:00:00Z",  # still forming at 12:15
                ],
                utc=True,
            ),
            "close": [1.0, 2.0, 3.0],
        }
    )
    out = feed.drop_forming_bars(df, as_of=as_of)
    assert list(out["close"]) == [1.0, 2.0]
    assert out["time"].iloc[-1] == pd.Timestamp("2024-01-01T11:30:00Z")


def test_copy_rates_range_filters_forming_bar(monkeypatch):
    as_of = datetime(2024, 1, 1, 12, 15, tzinfo=timezone.utc)

    def fake_range(symbol, timeframe, date_from, date_to):
        return [
            {
                "time": 1_704_106_800,  # 2024-01-01 11:00:00 UTC
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.0,
                "tick_volume": 1,
                "spread": 1,
                "real_volume": 1,
            },
            {
                "time": 1_704_108_600,  # 11:30
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 2.0,
                "tick_volume": 1,
                "spread": 1,
                "real_volume": 1,
            },
            {
                "time": 1_704_110_400,  # 12:00 forming
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 3.0,
                "tick_volume": 1,
                "spread": 1,
                "real_volume": 1,
            },
        ]

    monkeypatch.setattr(mt5, "copy_rates_range", fake_range)
    feed = DataFeed(Connection(), "#US30", "M30")
    frame = feed.copy_rates_range(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        as_of,
        closed_only=True,
    )
    assert frame is not None
    assert list(frame["close"]) == [1.0, 2.0]


def test_last_n_months_uses_closed_fallback(monkeypatch):
    seen = {}

    def fake_range(*_a, **_k):
        return None

    def fake_from_pos(symbol, timeframe, start_pos, count):
        seen.update(start_pos=start_pos, count=count)
        return [
            {
                "time": 1_700_000_000,
                "open": 1,
                "high": 2,
                "low": 0.5,
                "close": 1.5,
                "tick_volume": 1,
                "spread": 1,
                "real_volume": 1,
            }
        ]

    monkeypatch.setattr(mt5, "copy_rates_range", fake_range)
    monkeypatch.setattr(mt5, "copy_rates_from_pos", fake_from_pos)
    feed = DataFeed(Connection(), "#US30", "M30")
    frame = feed.last_n_months(months=1, pad_bars=10)
    assert seen["start_pos"] == 1
    assert frame is not None
    assert len(frame) == 1
