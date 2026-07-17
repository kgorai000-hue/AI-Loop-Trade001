from __future__ import annotations

import pandas as pd

from src.search import (
    _resync_best_after_gate,
    apply_dsr_family_gate,
    apply_fdr_bh_gate,
    validation_ranking_row,
    validation_result_from_ranking,
)
from src.strategy import StrategyParams
from src.validator import StrategyValidator, ValidationResult, ValidatorConfig


def _params(long: int, short: int = 24) -> StrategyParams:
    return StrategyParams(long_window=long, short_window=short, max_hold_bars=8)


def _val(**kwargs) -> ValidationResult:
    base = dict(
        accepted=True,
        sharpe=2.0,
        max_drawdown=0.05,
        p_value=0.01,
        ic=0.1,
        oos_degradation=0.1,
        n_trades=50,
        oos_n_trades=20,
        dsr=0.99,
        hac_p_value=0.01,
    )
    base.update(kwargs)
    return ValidationResult(**base)


def test_resync_promotes_matching_validation_from_ranking():
    weak = _params(180)
    strong = _params(240)
    rankings = [
        validation_ranking_row(_val(accepted=False, sharpe=2.5, p_value=0.04), weak),
        validation_ranking_row(_val(accepted=True, sharpe=2.1, p_value=0.01), strong),
    ]
    # Simulate FDR having rejected the prior winner (weak).
    rankings[0]["accepted"] = False
    rankings[0]["reasons"] = ["p_value 0.0400 fails FDR-BH"]
    rankings[0]["flags"] = ["fdr_bh_gate"]

    params, val, n_acc = _resync_best_after_gate(
        best_params=weak,
        best_val=_val(sharpe=2.5, p_value=0.04),
        rankings=rankings,
    )
    assert n_acc == 1
    assert params is not None and params.as_dict() == strong.as_dict()
    assert val is not None
    assert val.accepted is True
    assert val.sharpe == 2.1
    assert val.p_value == 0.01


def test_fdr_promotion_does_not_keep_stale_best_val():
    cfg = ValidatorConfig(multiple_testing="fdr_bh", p_value_max=0.05, min_dsr=0.0)
    v = StrategyValidator(cfg)
    a = _params(200)
    b = _params(240)
    rankings = [
        validation_ranking_row(_val(accepted=True, sharpe=2.8, p_value=0.90), a),
        validation_ranking_row(_val(accepted=True, sharpe=2.0, p_value=0.01), b),
    ]
    # BH keeps only p=0.01; p=0.90 fails -> promote B with B's metrics.
    params, val, n_acc = apply_fdr_bh_gate(
        validator=v,
        best_params=a,
        best_val=_val(sharpe=2.8, p_value=0.90),
        accepted_count=2,
        rankings=rankings,
    )
    assert n_acc == 1
    assert params is not None and params.as_dict() == b.as_dict()
    assert val is not None
    assert val.sharpe == 2.0
    assert val.p_value == 0.01
    assert val.accepted is True


def test_dsr_promotion_rebuilds_best_validation():
    cfg = ValidatorConfig(min_dsr=0.99, pbo_enabled=False, multiple_testing="none")
    v = StrategyValidator(cfg)
    weak = _params(180)
    strong = _params(240)
    # Near-zero mean -> fails DSR; mild positive drift should clear the gate.
    import numpy as np

    rng = np.random.default_rng(0)
    flat = pd.Series(rng.normal(0.0, 0.01, size=200))
    edged = pd.Series(rng.normal(0.003, 0.01, size=200))
    rankings = [
        validation_ranking_row(
            _val(accepted=True, sharpe=2.5, dsr=0.99),
            weak,
            flags=["dsr_gate_deferred_family"],
        ),
        validation_ranking_row(
            _val(accepted=True, sharpe=2.0, dsr=0.99),
            strong,
            flags=["dsr_gate_deferred_family"],
        ),
    ]
    params, val, n_acc = apply_dsr_family_gate(
        validator=v,
        best_params=weak,
        best_val=_val(sharpe=2.5, dsr=0.99),
        accepted_count=2,
        rankings=rankings,
        return_series=[flat, edged],
    )
    assert rankings[0]["accepted"] is False
    assert "dsr_gate" in rankings[0]["flags"]
    if n_acc >= 1:
        assert params is not None and val is not None
        assert val.accepted is True
        survivor = next(r for r in rankings if r.get("accepted"))
        assert params.as_dict() == survivor["params"]
        assert val.sharpe == survivor["sharpe"]
        assert abs(val.dsr - float(survivor["dsr"])) < 1e-12
    else:
        assert params is None
        assert val is None or val.accepted is False


def test_validation_roundtrip_ranking():
    params = _params(220)
    original = _val(sharpe=1.7, is_sharpe=1.8, oos_sharpe=1.5, pbo=0.2)
    row = validation_ranking_row(original, params)
    rebuilt = validation_result_from_ranking(row)
    assert rebuilt.as_dict() == original.as_dict()


def test_validation_result_preserves_zero_p_values():
    params = _params(200)
    original = _val(
        sharpe=0.0,
        max_drawdown=0.0,
        p_value=0.0,
        ic=0.0,
        dsr=0.0,
        hac_p_value=0.0,
        bootstrap_p_value=0.0,
        n_trades=0,
        oos_n_trades=0,
    )
    row = validation_ranking_row(original, params)
    rebuilt = validation_result_from_ranking(row)
    assert rebuilt.p_value == 0.0
    assert rebuilt.hac_p_value == 0.0
    assert rebuilt.bootstrap_p_value == 0.0
    assert rebuilt.sharpe == 0.0
    assert rebuilt.dsr == 0.0
    assert rebuilt.n_trades == 0
