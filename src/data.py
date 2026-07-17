"""Historical / live bar feed over MetaTrader5."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import MetaTrader5 as mt5
import pandas as pd

from .connection import MT5Connection

logger = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class DataFeed:
    """Fetch OHLCV bars for a symbol via an MT5Connection."""

    def __init__(self, connection: MT5Connection, symbol: str, timeframe: str = "M30") -> None:
        self.connection = connection
        self.symbol = symbol
        self.timeframe_name = timeframe.upper()
        if self.timeframe_name not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self.timeframe = TIMEFRAME_MAP[self.timeframe_name]

    def ensure_symbol(self) -> bool:
        info = self.connection.symbol_info(self.symbol)
        return info is not None

    def copy_rates(self, count: int, start_pos: int = 0) -> Optional[pd.DataFrame]:
        """Return `count` bars as a DataFrame, oldest first.

        MT5 position 0 is the currently forming bar. Callers that make trading
        decisions should use :meth:`copy_closed_rates` instead.
        """
        if not self.connection.ensure() or not self.ensure_symbol():
            return None
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, start_pos, count)
        if rates is None or len(rates) == 0:
            logger.warning(
                "copy_rates_from_pos empty for %s %s: %s",
                self.symbol,
                self.timeframe_name,
                mt5.last_error(),
            )
            return None
        return self._to_df(rates)

    def copy_closed_rates(self, count: int) -> Optional[pd.DataFrame]:
        """Return only completed bars, excluding MT5's forming bar at position 0."""
        return self.copy_rates(count=count, start_pos=1)

    def copy_rates_range(
        self,
        date_from: datetime,
        date_to: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        if not self.connection.ensure() or not self.ensure_symbol():
            return None
        if date_to is None:
            date_to = datetime.now(timezone.utc)
        if date_from.tzinfo is None:
            date_from = date_from.replace(tzinfo=timezone.utc)
        if date_to.tzinfo is None:
            date_to = date_to.replace(tzinfo=timezone.utc)

        rates = mt5.copy_rates_range(self.symbol, self.timeframe, date_from, date_to)
        if rates is None or len(rates) == 0:
            logger.warning(
                "copy_rates_range empty for %s: %s",
                self.symbol,
                mt5.last_error(),
            )
            return None
        return self._to_df(rates)

    def last_n_months(self, months: int = 6, pad_bars: int = 300) -> Optional[pd.DataFrame]:
        """Fetch ~`months` of history plus warmup bars for indicators."""
        now = datetime.now(timezone.utc)
        date_from = now - timedelta(days=int(months * 30.5) + 5)
        df = self.copy_rates_range(date_from, now)
        if df is None:
            bars_per_day = {
                "M1": 1440,
                "M5": 288,
                "M15": 96,
                "M30": 48,
                "H1": 24,
                "H4": 6,
                "D1": 1,
            }
            per_day = bars_per_day.get(self.timeframe_name, 48)
            count = int(months * 30.5 * per_day) + pad_bars
            df = self.copy_rates(count)
        return df

    def tick(self) -> Optional[Any]:
        if not self.connection.ensure() or not self.ensure_symbol():
            return None
        return mt5.symbol_info_tick(self.symbol)

    @staticmethod
    def _to_df(rates) -> pd.DataFrame:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "tick_volume", "spread", "real_volume"):
            if col in df.columns:
                df[col] = df[col].astype(float)
        return df
