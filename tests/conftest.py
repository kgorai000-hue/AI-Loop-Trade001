from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from mt5_constants import OFFICIAL_CONSTANTS, STUB_API_NAMES


def _install_mt5_stub() -> None:
    """Linux CI stub with constants matching MetaTrader5==5.0.5735."""

    def _default(*args, **kwargs):
        return None

    def _order_check(request):
        # Pass preflight so unit tests exercise order_send paths by default.
        return SimpleNamespace(
            retcode=OFFICIAL_CONSTANTS["TRADE_RETCODE_DONE"],
            balance=0.0,
            equity=0.0,
            profit=0.0,
            margin=0.0,
            margin_free=0.0,
            margin_level=0.0,
            comment="Done",
            request=request,
        )

    mt5 = types.ModuleType("MetaTrader5")
    for name, value in OFFICIAL_CONSTANTS.items():
        setattr(mt5, name, value)
    for name in STUB_API_NAMES:
        if name == "order_check":
            setattr(mt5, name, _order_check)
        else:
            setattr(mt5, name, _default)
    sys.modules["MetaTrader5"] = mt5


try:
    import MetaTrader5  # noqa: F401
except ImportError:
    _install_mt5_stub()
