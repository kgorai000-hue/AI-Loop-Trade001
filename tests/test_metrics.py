from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import build_report, information_coefficient, market_forward_returns


def test_market_forward_returns_are_price_pct_change():
    close = pd.Series([100.0, 110.0, 99.0, 99.0])
    fwd = market_forward_returns(close)
    assert abs(fwd.iloc[0] - 0.10) < 1e-12
    assert abs(fwd.iloc[1] - (-0.10)) < 1e-12
    assert abs(fwd.iloc[2] - 0.0) < 1e-12
    assert np.isnan(fwd.iloc[3])


def test_ic_uses_market_forward_not_strategy_returns():
    close = pd.Series([100.0, 110.0, 100.0, 120.0, 110.0, 130.0, 120.0, 140.0])
    mkt = market_forward_returns(close)
    # Sign of next price move.
    signals = pd.Series(np.sign(mkt.fillna(0.0)).astype(int))
    signals.iloc[-1] = 0
    # Strategy equity returns: unrelated constant path.
    strat = pd.Series([0.01] * 7 + [0.0])

    ic_market = information_coefficient(signals, mkt)
    assert ic_market > 0.8

    report = build_report(
        strat,
        signals=signals,
        market_forward_returns=mkt,
    )
    assert report.ic == ic_market

    # Old (incorrect) definition would correlate signal with strategy PnL.
    ic_legacy = information_coefficient(signals, strat.shift(-1))
    assert abs(report.ic - ic_legacy) > 0.3

    # Omitting market forwards must not silently fall back to strategy returns.
    report_no_mkt = build_report(strat, signals=signals)
    assert report_no_mkt.ic == 0.0
