"""Official MetaTrader5==5.0.5735 constants shared by Linux stub and Windows CI.

Keep this module under ``tests/`` -- production code imports the real package
(or the conftest stub that mirrors these values).
"""

from __future__ import annotations

# Values verified against the Windows MetaTrader5 5.0.5735 wheel.
OFFICIAL_CONSTANTS: dict[str, int] = {
    "TIMEFRAME_M1": 1,
    "TIMEFRAME_M5": 5,
    "TIMEFRAME_M15": 15,
    "TIMEFRAME_M30": 30,
    "TIMEFRAME_H1": 16385,
    "TIMEFRAME_H4": 16388,
    "TIMEFRAME_D1": 16408,
    "ORDER_TYPE_BUY": 0,
    "ORDER_TYPE_SELL": 1,
    "ORDER_TYPE_BUY_LIMIT": 2,
    "ORDER_TYPE_SELL_LIMIT": 3,
    "POSITION_TYPE_BUY": 0,
    "POSITION_TYPE_SELL": 1,
    "TRADE_ACTION_DEAL": 1,
    "TRADE_ACTION_PENDING": 5,
    "TRADE_ACTION_REMOVE": 8,
    "ORDER_TIME_GTC": 0,
    "ORDER_FILLING_FOK": 0,
    "ORDER_FILLING_IOC": 1,
    "ORDER_FILLING_RETURN": 2,
    "ORDER_FILLING_BOC": 3,
    "TRADE_RETCODE_PLACED": 10008,
    "TRADE_RETCODE_DONE": 10009,
    "TRADE_RETCODE_DONE_PARTIAL": 10010,
    "ACCOUNT_TRADE_MODE_DEMO": 0,
    "ACCOUNT_TRADE_MODE_CONTEST": 1,
    "ACCOUNT_TRADE_MODE_REAL": 2,
    "DEAL_ENTRY_IN": 0,
    "DEAL_ENTRY_OUT": 1,
    "DEAL_ENTRY_INOUT": 2,
    "DEAL_ENTRY_OUT_BY": 3,
}

STUB_API_NAMES: tuple[str, ...] = (
    "order_calc_margin",
    "order_calc_profit",
    "copy_rates_from_pos",
    "copy_rates_range",
    "history_deals_get",
    "symbol_info_tick",
    "positions_get",
    "orders_get",
    "order_check",
    "order_send",
    "last_error",
    "initialize",
    "shutdown",
    "terminal_info",
    "account_info",
    "symbol_info",
    "symbol_select",
)
