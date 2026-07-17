from __future__ import annotations

import MetaTrader5 as mt5

from src.data import DataFeed


class Connection:
    def ensure(self):
        return True

    def symbol_info(self, symbol):
        return object()


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
