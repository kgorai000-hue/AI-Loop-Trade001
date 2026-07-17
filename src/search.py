"""Shared helpers for parameter search rankings and PBO / FDR / DSR gating."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from .inference import (
    benjamini_hochberg_accept,
    deflated_sharpe_ratio,
    nonannual_sharpe,
    probability_of_backtest_overfitting,
    trial_sharpe_dispersion,
)
from .persistence import StateStore
from .strategy import StrategyParams
from .validator import StrategyValidator, ValidationResult

logger = logging.getLogger(__name__)

SearchBest = tuple[Optional[StrategyParams], Optional[ValidationResult], int]


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
        "is_sharpe": val.is_sharpe,
        "oos_sharpe": val.oos_sharpe,
        "overfitting": val.overfitting,
        "reasons": list(val.reasons),
        "flags": list(val.flags),
        "n_trades": val.n_trades,
        "oos_n_trades": val.oos_n_trades,
        "regime_trades": dict(val.regime_trades or {}),
        "p_value_threshold": val.p_value_threshold,
        "dsr": val.dsr,
        "sr_star": val.sr_star,
        "hac_p_value": val.hac_p_value,
        "bootstrap_p_value": val.bootstrap_p_value,
        "pbo": val.pbo,
    }
    row.update(extra)
    return row


def _num(row: dict[str, Any], key: str, default: float) -> float:
    """Coerce ``row[key]`` to float; only ``None`` / missing uses ``default``.

    Important: ``0.0`` is a valid p-value / metric and must not fall through
    ``x or default`` (which would incorrectly become ``default``).
    """
    if key not in row or row[key] is None:
        return float(default)
    return float(row[key])


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    if key not in row or row[key] is None:
        return int(default)
    return int(row[key])


def validation_result_from_ranking(row: dict[str, Any]) -> ValidationResult:
    """Rebuild a ValidationResult from a rankings row (post-gate promotion)."""
    pbo = row.get("pbo")
    return ValidationResult(
        accepted=bool(row.get("accepted")),
        reasons=list(row.get("reasons") or []),
        flags=list(row.get("flags") or []),
        sharpe=_num(row, "sharpe", 0.0),
        max_drawdown=_num(row, "max_drawdown", 0.0),
        p_value=_num(row, "p_value", 1.0),
        ic=_num(row, "ic", 0.0),
        oos_degradation=_num(row, "oos_degradation", 0.0),
        is_sharpe=_num(row, "is_sharpe", 0.0),
        oos_sharpe=_num(row, "oos_sharpe", 0.0),
        overfitting=bool(row.get("overfitting")),
        n_trades=_int(row, "n_trades", 0),
        oos_n_trades=_int(row, "oos_n_trades", 0),
        regime_trades=dict(row.get("regime_trades") or {}),
        p_value_threshold=_num(row, "p_value_threshold", 0.05),
        dsr=_num(row, "dsr", 0.0),
        sr_star=_num(row, "sr_star", 0.0),
        hac_p_value=_num(row, "hac_p_value", 1.0),
        bootstrap_p_value=_num(row, "bootstrap_p_value", 1.0),
        pbo=float(pbo) if pbo is not None else None,
    )


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


def _accepted_count(rankings: list[dict[str, Any]]) -> int:
    return sum(1 for r in rankings if r.get("accepted"))


def _survivors_by_sharpe(rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (r for r in rankings if r.get("accepted")),
        key=lambda r: _num(r, "sharpe", 0.0),
        reverse=True,
    )


def _ranking_for_params(
    rankings: list[dict[str, Any]],
    params: Optional[StrategyParams],
) -> Optional[dict[str, Any]]:
    if params is None:
        return None
    bp = params.as_dict()
    return next((r for r in rankings if r.get("params") == bp), None)


def _best_still_accepted(
    best_params: Optional[StrategyParams],
    rankings: list[dict[str, Any]],
) -> bool:
    row = _ranking_for_params(rankings, best_params)
    return bool(row is not None and row.get("accepted"))


def _resync_best_after_gate(
    *,
    best_params: Optional[StrategyParams],
    best_val: Optional[ValidationResult],
    rankings: list[dict[str, Any]],
) -> SearchBest:
    """Keep or replace the search winner after a family-wide gate mutates rankings.

    On promotion, rebuild ``best_val`` from the survivor ranking row so
    ``best_params`` and ``best_validation`` always describe the same candidate.
    """
    new_accepted = _accepted_count(rankings)
    if _best_still_accepted(best_params, rankings):
        row = _ranking_for_params(rankings, best_params)
        assert row is not None
        return best_params, validation_result_from_ranking(row), new_accepted

    # Previous winner rejected -- keep its (updated) row for diagnostics only.
    rejected_row = _ranking_for_params(rankings, best_params)
    rejected_val: Optional[ValidationResult]
    if rejected_row is not None:
        rejected_val = validation_result_from_ranking(rejected_row)
    elif best_val is not None:
        best_val.accepted = False
        rejected_val = best_val
    else:
        rejected_val = None

    survivors = _survivors_by_sharpe(rankings)
    if not survivors:
        return None, rejected_val, 0

    top = survivors[0]
    return StrategyParams(**top["params"]), validation_result_from_ranking(top), new_accepted


def _append_unique(items: list[str], msg: str) -> None:
    if msg not in items:
        items.append(msg)


def apply_fdr_bh_gate(
    *,
    validator: StrategyValidator,
    best_params: Optional[StrategyParams],
    best_val: Optional[ValidationResult],
    accepted_count: int,
    rankings: list[dict[str, Any]],
) -> SearchBest:
    """Apply Benjamini-Hochberg FDR across the search family's p-values.

    When ``multiple_testing=fdr_bh``, per-candidate validation defers the p-gate;
    this post-pass rejects candidates whose p fails BH at ``p_value_max``.
    """
    cfg = validator.config
    mt = (cfg.multiple_testing or "none").lower()
    if mt not in ("fdr", "fdr_bh", "bh") or not rankings:
        return best_params, best_val, accepted_count

    alpha = float(cfg.p_value_max)
    p_values = [_num(row, "p_value", 1.0) for row in rankings]
    mask = benjamini_hochberg_accept(p_values, alpha=alpha)
    reason_tail = f"fails FDR-BH (alpha={alpha}, m={len(p_values)})"

    for i, row in enumerate(rankings):
        if not row.get("accepted") or mask[i]:
            continue
        row["accepted"] = False
        reasons = list(row.get("reasons") or [])
        _append_unique(reasons, f"p_value {p_values[i]:.4f} {reason_tail}")
        row["reasons"] = reasons
        flags = list(row.get("flags") or [])
        if "fdr_bh_gate" not in flags:
            flags.append("fdr_bh_gate")
        row["flags"] = flags

    return _resync_best_after_gate(
        best_params=best_params,
        best_val=best_val,
        rankings=rankings,
    )


def apply_dsr_family_gate(
    *,
    validator: StrategyValidator,
    best_params: Optional[StrategyParams],
    best_val: Optional[ValidationResult],
    accepted_count: int,
    rankings: list[dict[str, Any]],
    return_series: list[pd.Series],
) -> SearchBest:
    """Recompute DSR with cross-trial Sharpe dispersion and enforce ``min_dsr``.

    Per-candidate validation defers the DSR gate when ``n_tests>1`` because SR*
    requires the family-wide ``sigma_{SR_n}``, not a single strategy's estimation SE.
    """
    cfg = validator.config
    if cfg.min_dsr <= 0 or not rankings or not return_series:
        return best_params, best_val, accepted_count

    n = min(len(rankings), len(return_series))
    if n < 1:
        return best_params, best_val, accepted_count

    trial_srs = [
        nonannual_sharpe(return_series[i].fillna(0.0).to_numpy(dtype=float))
        for i in range(n)
    ]
    _, trials_std = trial_sharpe_dispersion(trial_srs)
    n_trials = max(n, 1)

    for i in range(n):
        row = rankings[i]
        dsr, sr_star, _sr = deflated_sharpe_ratio(
            return_series[i],
            n_trials=n_trials,
            trial_sharpes=trial_srs,
            sr_trials_std=trials_std,
        )
        row["dsr"] = dsr
        row["sr_star"] = sr_star
        row["sr_trials_std"] = trials_std
        flags = [f for f in (row.get("flags") or []) if f != "dsr_gate_deferred_family"]
        if "dsr_family_applied" not in flags:
            flags.append("dsr_family_applied")
        reasons = [
            r
            for r in (row.get("reasons") or [])
            if not str(r).startswith("deflated_sharpe ")
        ]
        if dsr < cfg.min_dsr:
            reasons.append(
                f"deflated_sharpe {dsr:.4f} < {cfg.min_dsr} "
                f"(sr_star={sr_star:.6f}, n_trials={n_trials}, "
                f"sr_trials_std={trials_std:.6f})"
            )
            if "dsr_gate" not in flags:
                flags.append("dsr_gate")
            row["accepted"] = False
        row["flags"] = flags
        row["reasons"] = reasons

    return _resync_best_after_gate(
        best_params=best_params,
        best_val=best_val,
        rankings=rankings,
    )


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
            _append_unique(best_val.reasons, f"pbo {pbo:.4f} > {cfg.pbo_max}")
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


def apply_search_family_gates(
    *,
    validator: StrategyValidator,
    state_store: Optional[StateStore],
    best_params: Optional[StrategyParams],
    best_val: Optional[ValidationResult],
    accepted_count: int,
    rankings: list[dict[str, Any]],
    return_series: list[pd.Series],
) -> tuple[Optional[StrategyParams], Optional[ValidationResult], int, Optional[float]]:
    """Run FDR -> DSR(family) -> PBO post-passes shared by grid and LLM search."""
    best_params, best_val, accepted_count = apply_fdr_bh_gate(
        validator=validator,
        best_params=best_params,
        best_val=best_val,
        accepted_count=accepted_count,
        rankings=rankings,
    )
    best_params, best_val, accepted_count = apply_dsr_family_gate(
        validator=validator,
        best_params=best_params,
        best_val=best_val,
        accepted_count=accepted_count,
        rankings=rankings,
        return_series=return_series,
    )
    return apply_pbo_gate(
        validator=validator,
        state_store=state_store,
        best_params=best_params,
        best_val=best_val,
        accepted_count=accepted_count,
        rankings=rankings,
        return_series=return_series,
    )
