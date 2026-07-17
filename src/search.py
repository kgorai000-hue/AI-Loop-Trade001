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
        "overfitting": val.overfitting,
        "reasons": list(val.reasons),
        "flags": list(val.flags),
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


def _accepted_count(rankings: list[dict[str, Any]]) -> int:
    return sum(1 for r in rankings if r.get("accepted"))


def _survivors_by_sharpe(rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (r for r in rankings if r.get("accepted")),
        key=lambda r: float(r.get("sharpe") or 0.0),
        reverse=True,
    )


def _best_still_accepted(
    best_params: Optional[StrategyParams],
    rankings: list[dict[str, Any]],
) -> bool:
    if best_params is None:
        return False
    bp = best_params.as_dict()
    return any(r.get("params") == bp and r.get("accepted") for r in rankings)


def _resync_best_after_gate(
    *,
    best_params: Optional[StrategyParams],
    best_val: Optional[ValidationResult],
    rankings: list[dict[str, Any]],
    reject_best_val: bool = True,
    clear_best_val_on_promote: bool = False,
) -> SearchBest:
    """Keep or replace the search winner after a family-wide gate mutates rankings."""
    new_accepted = _accepted_count(rankings)
    if _best_still_accepted(best_params, rankings):
        return best_params, best_val, new_accepted

    if reject_best_val and best_val is not None:
        best_val.accepted = False

    survivors = _survivors_by_sharpe(rankings)
    if not survivors:
        return None, best_val, 0
    top = survivors[0]
    promoted = StrategyParams(**top["params"])
    if clear_best_val_on_promote:
        return promoted, None, new_accepted
    return promoted, best_val, new_accepted


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
    """Apply Benjamini–Hochberg FDR across the search family's p-values.

    When ``multiple_testing=fdr_bh``, per-candidate validation defers the p-gate;
    this post-pass rejects candidates whose p fails BH at ``p_value_max``.
    """
    cfg = validator.config
    mt = (cfg.multiple_testing or "none").lower()
    if mt not in ("fdr", "fdr_bh", "bh") or not rankings:
        return best_params, best_val, accepted_count

    alpha = float(cfg.p_value_max)
    p_values = [float(row.get("p_value", 1.0) or 1.0) for row in rankings]
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

    if best_val is not None and best_params is not None and not _best_still_accepted(
        best_params, rankings
    ):
        msg = f"p_value {best_val.p_value:.4f} {reason_tail}"
        _append_unique(best_val.reasons, msg)
        if "fdr_bh_gate" not in best_val.flags:
            best_val.flags.append("fdr_bh_gate")

    return _resync_best_after_gate(
        best_params=best_params,
        best_val=best_val,
        rankings=rankings,
        reject_best_val=True,
        clear_best_val_on_promote=False,
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
    requires the family-wide ``σ_{SR_n}``, not a single strategy's estimation SE.
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

    if best_val is not None and best_params is not None:
        bp = best_params.as_dict()
        match_i = next(
            (i for i in range(n) if rankings[i].get("params") == bp),
            None,
        )
        if match_i is not None:
            best_val.dsr = float(rankings[match_i].get("dsr") or 0.0)
            best_val.sr_star = float(rankings[match_i].get("sr_star") or 0.0)
            best_val.flags = [
                f for f in best_val.flags if f != "dsr_gate_deferred_family"
            ]
            if "dsr_family_applied" not in best_val.flags:
                best_val.flags.append("dsr_family_applied")
            best_val.reasons = [
                r
                for r in best_val.reasons
                if not str(r).startswith("deflated_sharpe ")
            ]
            if best_val.dsr < cfg.min_dsr:
                best_val.reasons.append(
                    f"deflated_sharpe {best_val.dsr:.4f} < {cfg.min_dsr} "
                    f"(sr_star={best_val.sr_star:.6f}, n_trials={n_trials}, "
                    f"sr_trials_std={trials_std:.6f})"
                )
                if "dsr_gate" not in best_val.flags:
                    best_val.flags.append("dsr_gate")
                best_val.accepted = False

    return _resync_best_after_gate(
        best_params=best_params,
        best_val=best_val,
        rankings=rankings,
        reject_best_val=False,
        clear_best_val_on_promote=True,
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
    """Run FDR → DSR(family) → PBO post-passes shared by grid and LLM search."""
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
