from __future__ import annotations

import sys
import types


def _default(*args, **kwargs):
    return None


mt5 = types.ModuleType("MetaTrader5")
for name, value in {
    "TIMEFRAME_M1": 1,
    "TIMEFRAME_M5": 5,
    "TIMEFRAME_M15": 15,
    "TIMEFRAME_M30": 30,
    "TIMEFRAME_H1": 60,
    "TIMEFRAME_H4": 240,
    "TIMEFRAME_D1": 1440,
    "ORDER_TYPE_BUY": 0,
    "ORDER_TYPE_SELL": 1,
    "ORDER_TYPE_BUY_LIMIT": 2,
    "ORDER_TYPE_SELL_LIMIT": 3,
    "POSITION_TYPE_BUY": 0,
    "POSITION_TYPE_SELL": 1,
    "TRADE_ACTION_DEAL": 10,
    "TRADE_ACTION_PENDING": 11,
    "TRADE_ACTION_REMOVE": 12,
    "ORDER_TIME_GTC": 20,
    "ORDER_FILLING_FOK": 30,
    "ORDER_FILLING_IOC": 31,
    "ORDER_FILLING_RETURN": 32,
    "TRADE_RETCODE_DONE": 10009,
}.items():
    setattr(mt5, name, value)

for name in (
    "order_calc_margin",
    "copy_rates_from_pos",
    "copy_rates_range",
    "symbol_info_tick",
    "positions_get",
    "orders_get",
    "order_send",
    "last_error",
    "initialize",
    "shutdown",
    "terminal_info",
    "account_info",
    "symbol_info",
    "symbol_select",
):
    setattr(mt5, name, _default)

sys.modules.setdefault("MetaTrader5", mt5)
