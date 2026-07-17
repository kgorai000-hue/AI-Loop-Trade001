"""Windows / real MetaTrader5 wheel compatibility checks.

Ubuntu CI uses a stub (``tests/conftest.py``) when the Windows-only wheel is
absent. This module asserts the official constant values and, on Windows with
the real package installed, namedtuple field layouts used by live trading code.

It does **not** require a running MT5 terminal or FxPro login -- ``initialize`` /
broker filling discovery remain out of scope for CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import MetaTrader5 as mt5
import pytest

from mt5_constants import OFFICIAL_CONSTANTS
from src.execution import OrderExecutor
from src.persistence import StateStore, _atomic_write_text

# Fields our live code reads from order_send / history deals / positions.
ORDER_SEND_RESULT_FIELDS = (
    "retcode",
    "deal",
    "order",
    "volume",
    "price",
    "bid",
    "ask",
    "comment",
    "request_id",
    "retcode_external",
    "request",
)
TRADE_DEAL_FIELDS = (
    "ticket",
    "order",
    "time",
    "time_msc",
    "type",
    "entry",
    "magic",
    "position_id",
    "reason",
    "volume",
    "price",
    "commission",
    "swap",
    "profit",
    "fee",
    "symbol",
    "comment",
    "external_id",
)
TRADE_POSITION_FIELDS = (
    "ticket",
    "time",
    "type",
    "magic",
    "volume",
    "price_open",
    "sl",
    "tp",
    "price_current",
    "swap",
    "profit",
    "symbol",
    "comment",
)
SYMBOL_INFO_REQUIRED = (
    "name",
    "digits",
    "point",
    "filling_mode",
    "trade_stops_level",
    "trade_tick_size",
    "trade_tick_value",
    "volume_min",
    "volume_max",
    "volume_step",
)

_REAL_WHEEL = hasattr(mt5, "OrderSendResult")
_WINDOWS = sys.platform.startswith("win")


def test_mt5_module_importable():
    assert mt5 is not None
    assert hasattr(mt5, "order_send")
    assert hasattr(mt5, "order_check")
    assert hasattr(mt5, "initialize")
    assert hasattr(mt5, "history_deals_get")


@pytest.mark.parametrize("name,expected", sorted(OFFICIAL_CONSTANTS.items()))
def test_mt5_constants_match_official_wheel(name: str, expected: int):
    assert getattr(mt5, name) == expected


def test_order_check_without_terminal_returns_none_or_result():
    """No initialize() -- must not crash; None or a CheckResult are both OK."""
    result = mt5.order_check(
        {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": "#US30",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": 1.0,
            "deviation": 10,
            "magic": 260717,
            "comment": "ci",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
    )
    assert result is None or hasattr(result, "retcode")


def test_order_send_without_terminal_returns_none():
    """No initialize() -- both stub and real wheel return None (not a crash)."""
    result = mt5.order_send(
        {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": "#US30",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": 1.0,
            "deviation": 10,
            "magic": 260717,
            "comment": "ci",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
    )
    assert result is None


def test_fxpro_style_filling_mode_selection():
    """Bitmask on SymbolInfo.filling_mode -> ORDER_FILLING_* (FxPro IOC common)."""
    assert OrderExecutor._deal_filling_mode(SimpleNamespace(filling_mode=2)) == mt5.ORDER_FILLING_IOC
    assert OrderExecutor._deal_filling_mode(SimpleNamespace(filling_mode=1)) == mt5.ORDER_FILLING_FOK
    assert OrderExecutor._deal_filling_mode(SimpleNamespace(filling_mode=3)) == mt5.ORDER_FILLING_IOC
    assert OrderExecutor._deal_filling_mode(SimpleNamespace(filling_mode=0)) == mt5.ORDER_FILLING_RETURN
    assert OrderExecutor._pending_filling_mode() == mt5.ORDER_FILLING_RETURN


@pytest.mark.skipif(not _REAL_WHEEL, reason="real MetaTrader5 wheel not installed")
def test_real_wheel_order_send_result_namedtuple_fields():
    fields = getattr(mt5.OrderSendResult, "__match_args__", None) or getattr(
        mt5.OrderSendResult, "_fields", ()
    )
    for name in ORDER_SEND_RESULT_FIELDS:
        assert name in fields, f"missing OrderSendResult.{name}"


@pytest.mark.skipif(not _REAL_WHEEL, reason="real MetaTrader5 wheel not installed")
def test_real_wheel_trade_deal_namedtuple_fields():
    fields = getattr(mt5.TradeDeal, "__match_args__", None) or getattr(mt5.TradeDeal, "_fields", ())
    for name in TRADE_DEAL_FIELDS:
        assert name in fields, f"missing TradeDeal.{name}"


@pytest.mark.skipif(not _REAL_WHEEL, reason="real MetaTrader5 wheel not installed")
def test_real_wheel_trade_position_namedtuple_fields():
    fields = getattr(mt5.TradePosition, "__match_args__", None) or getattr(
        mt5.TradePosition, "_fields", ()
    )
    for name in TRADE_POSITION_FIELDS:
        assert name in fields, f"missing TradePosition.{name}"


@pytest.mark.skipif(not _REAL_WHEEL, reason="real MetaTrader5 wheel not installed")
def test_real_wheel_symbol_info_has_filling_mode():
    fields = getattr(mt5.SymbolInfo, "__match_args__", None) or getattr(mt5.SymbolInfo, "_fields", ())
    for name in SYMBOL_INFO_REQUIRED:
        assert name in fields, f"missing SymbolInfo.{name}"


@pytest.mark.skipif(not _WINDOWS, reason="Windows-only os.replace semantics")
def test_windows_atomic_state_replace(tmp_path: Path):
    """STATE.md updates must survive Windows replace (no lingering .tmp)."""
    store = StateStore(tmp_path, "US30")
    store.update_state(equity=1000.0, last_review_date="2026-07-17")
    store.update_state(equity=2000.0)
    assert store.read_state()["equity"] == 2000.0
    assert not store.state_path.with_name(store.state_path.name + ".tmp").exists()

    path = tmp_path / "direct.txt"
    _atomic_write_text(path, "alpha")
    _atomic_write_text(path, "beta")
    assert path.read_text(encoding="utf-8") == "beta"
    assert not Path(str(path) + ".tmp").exists()


@pytest.mark.skipif(not _WINDOWS or not _REAL_WHEEL, reason="Windows + real MT5 wheel")
def test_windows_real_wheel_file_path():
    """Confirm CI installed the Windows MetaTrader5 distribution, not a stub."""
    path = getattr(mt5, "__file__", "") or ""
    assert path, "MetaTrader5.__file__ missing"
    assert "site-packages" in path.replace("\\", "/").lower() or os.path.isfile(path)
