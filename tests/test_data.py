from __future__ import annotations

import MetaTrader5 as mt5

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
