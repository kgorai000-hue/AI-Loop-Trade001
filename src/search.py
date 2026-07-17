"""Shared helpers for parameter search rankings and PBO gating."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from .inference import probability_of_backtest_overfitting
from .persistence import StateStore
from .strategy import StrategyParams
from .validator import StrategyValidator, ValidationResult

logger = logging.getLogger(__name__)


def validation_ranking_row(
    val: ValidationResult,
    params: StrategyParams,
    **extra: Any,
) -> dict[str, Any]:
    """Build a rankings dict entry from a validation result."""
    row: dict[str, Any] = {
        "params": params.as_dict(),
        "accepted": val.accepted,
        "sharpe": val.sharpe,
        "max_drawdown": val.max_drawdown,
        "p_value": val.p_value,
        "ic": val.ic,
        "oos_degradation": val.oos_degradation,
        "overfitting": val.overfitting,
        "reasons": val.reasons,
        "n_trades": val.n_trades,
        "oos_n_trades": val.oos_n_trades,
        "regime_trades": val.regime_trades,
        "p_value_threshold": val.p_value_threshold,
        "dsr": val.dsr,
        "sr_star": val.sr_star,
        "hac_p_value": val.hac_p_value,
        "bootstrap_p_value": val.bootstrap_p_value,
    }
    row.update(extra)
    return row


def compute_search_pbo(
    return_series: list[pd.Series],
    *,
    n_slices: int = 8,
) -> Optional[float]:
    """Build a (T, N) matrix from aligned candidate bar returns and run CSCV PBO."""
    if len(return_series) < 2:
        return None
    length = min(len(s) for s in return_series)
    if length < 20:
        return None
    cols = [s.fillna(0.0).to_numpy(dtype=float)[:length] for s in return_series]
    mat = np.column_stack(cols)
    return probability_of_backtest_overfitting(mat, n_slices=n_slices)


def apply_pbo_gate(
    *,
    validator: StrategyValidator,
    state_store: Optional[StateStore],
    best_params: Optional[StrategyParams],
    best_val: Optional[ValidationResult],
    accepted_count: int,
    rankings: list[dict[str, Any]],
    return_series: list[pd.Series],
) -> tuple[Optional[StrategyParams], Optional[ValidationResult], int, Optional[float]]:
    """Attach CSCV PBO to rankings; reject the whole search if PBO exceeds the ceiling."""
    cfg = validator.config
    pbo = compute_search_pbo(return_series, n_slices=cfg.pbo_slices)
    if pbo is None:
        return best_params, best_val, accepted_count, None

    for row in rankings:
        row["pbo"] = pbo

    if not cfg.pbo_enabled:
        if best_val is not None:
            best_val.pbo = pbo
        return best_params, best_val, accepted_count, pbo

    if pbo > cfg.pbo_max:
        logger.warning("PBO gate failed: pbo=%.4f > %.4f", pbo, cfg.pbo_max)
        if state_store:
            state_store.append_lesson(
                f"PBO rejected search: pbo={pbo:.4f} > {cfg.pbo_max}"
            )
        if best_val is not None:
            best_val.pbo = pbo
            best_val.accepted = False
            best_val.overfitting = True
            reason = f"pbo {pbo:.4f} > {cfg.pbo_max}"
            if reason not in best_val.reasons:
                best_val.reasons.append(reason)
            if "pbo_gate" not in best_val.flags:
                best_val.flags.append("pbo_gate")
        for row in rankings:
            if row.get("accepted"):
                row["accepted"] = False
                reasons = list(row.get("reasons") or [])
                reasons.append(f"pbo {pbo:.4f} > {cfg.pbo_max}")
                row["reasons"] = reasons
        return None, best_val, 0, pbo

    if best_val is not None:
        best_val.pbo = pbo
    return best_params, best_val, accepted_count, pbo
