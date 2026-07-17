from __future__ import annotations

import sys
import types

from mt5_constants import OFFICIAL_CONSTANTS, STUB_API_NAMES


def _install_mt5_stub() -> None:
    """Linux CI stub with constants matching MetaTrader5==5.0.5735."""

    def _default(*args, **kwargs):
        return None

    mt5 = types.ModuleType("MetaTrader5")
    for name, value in OFFICIAL_CONSTANTS.items():
        setattr(mt5, name, value)
    for name in STUB_API_NAMES:
        setattr(mt5, name, _default)
    sys.modules["MetaTrader5"] = mt5


try:
    import MetaTrader5  # noqa: F401
except ImportError:
    _install_mt5_stub()
