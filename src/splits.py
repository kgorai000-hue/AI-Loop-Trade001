"""Chronological rolling-OOS and walk-forward fold helpers for nested search.

Periodic re-optimization uses a *rolling OOS* gate — not a permanent sealed
holdout. A true never-observed holdout would require a frozen end date that is
never re-entered; weekly ops cannot claim that.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def max_evaluated_oos_end(
    windows: Optional[list[dict[str, Any]]],
) -> Optional[pd.Timestamp]:
    """Latest ``end`` among previously evaluated rolling-OOS windows."""
    if not windows:
        return None
    ends: list[pd.Timestamp] = []
    for win in windows:
        raw = win.get("end") if isinstance(win, dict) else None
        if raw is None:
            continue
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        ends.append(ts)
    return max(ends) if ends else None


def chronological_rolling_oos(
    df: pd.DataFrame,
    oos_fraction: float = 0.15,
    *,
    min_oos_bars: int = 40,
    min_train_bars: int = 100,
    exclude_oos_before: Optional[pd.Timestamp] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[dict[str, str]]]:
    """Split ``df`` into search window + rolling OOS tail (time order).

    Bars with ``time <= exclude_oos_before`` have already been used as a gate
    and must not re-enter rolling OOS. Previously scored OOS may move into the
    search set on later runs (that is expected for rolling evaluation).

    Returns ``(search_df, rolling_oos, window)`` where ``window`` is
    ``{"start": iso, "end": iso}`` or ``None`` when OOS is empty.
    """
    n = len(df)
    empty = df.iloc[0:0].copy()
    if n <= 0:
        return empty, empty, None

    frac = min(max(float(oos_fraction), 0.0), 0.45)
    hold_n = int(round(n * frac)) if frac > 0 else 0
    hold_n = max(hold_n, min_oos_bars if frac > 0 else 0)
    hold_n = min(hold_n, max(0, n - min_train_bars))
    if hold_n <= 0 or frac <= 0:
        return df.reset_index(drop=True), empty, None

    split = n - hold_n
    candidate = df.iloc[split:].reset_index(drop=True)

    if exclude_oos_before is not None:
        excl = pd.Timestamp(exclude_oos_before)
        times = pd.to_datetime(candidate["time"])
        tz = getattr(times.dtype, "tz", None)
        if tz is not None:
            if excl.tzinfo is None:
                excl = excl.tz_localize(tz)
            else:
                excl = excl.tz_convert(tz)
        elif excl.tzinfo is not None:
            excl = excl.tz_convert("UTC").tz_localize(None)
        # Only *new* bars after the last evaluated OOS end may gate.
        rolling = candidate.loc[times > excl].reset_index(drop=True)
        if len(rolling) < min_oos_bars:
            # Insufficient unseen bars — fail-closed at the caller.
            return df.reset_index(drop=True), empty, None
        oos_start = rolling["time"].iloc[0]
        search = df.loc[df["time"] < oos_start].reset_index(drop=True)
        if len(search) < min_train_bars:
            return df.reset_index(drop=True), empty, None
    else:
        search = df.iloc[:split].reset_index(drop=True)
        rolling = candidate

    window = {
        "start": pd.Timestamp(rolling["time"].iloc[0]).isoformat(),
        "end": pd.Timestamp(rolling["time"].iloc[-1]).isoformat(),
    }
    return search, rolling, window


def chronological_holdout(
    df: pd.DataFrame,
    holdout_fraction: float = 0.15,
    *,
    min_holdout_bars: int = 40,
    min_train_bars: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deprecated alias for ``chronological_rolling_oos`` without exclusion.

    Prefer ``chronological_rolling_oos`` for periodic ops.
    """
    search, oos, _ = chronological_rolling_oos(
        df,
        holdout_fraction,
        min_oos_bars=min_holdout_bars,
        min_train_bars=min_train_bars,
        exclude_oos_before=None,
    )
    return search, oos


def walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 3,
    *,
    min_train_fraction: float = 0.40,
    min_train_bars: int = 80,
    min_test_bars: int = 30,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window walk-forward folds on pre-rolling-OOS data.

    Fold ``i`` uses ``df[:test_start]`` as train and the next contiguous
    block as outer OOS. Train never includes that fold's OOS.
    """
    n = len(df)
    folds_n = max(1, int(n_folds))
    if n < min_train_bars + min_test_bars:
        return []

    min_train = max(min_train_bars, int(n * float(min_train_fraction)))
    remaining = n - min_train
    if remaining < min_test_bars:
        return []

    fold_oos = max(min_test_bars, remaining // folds_n)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(folds_n):
        test_end = n - (folds_n - 1 - i) * fold_oos
        test_start = test_end - fold_oos
        if test_start < min_train_bars:
            continue
        if test_end > n:
            test_end = n
        if test_start >= test_end:
            continue
        train = df.iloc[:test_start].reset_index(drop=True)
        test = df.iloc[test_start:test_end].reset_index(drop=True)
        if len(train) < min_train_bars or len(test) < min_test_bars:
            continue
        folds.append((train, test))
    return folds


def with_warmup(
    train: pd.DataFrame,
    test: pd.DataFrame,
    warmup_bars: int,
) -> tuple[pd.DataFrame, int]:
    """Prepend train tail for signal warmup; return frame and bars to strip."""
    warm = max(0, int(warmup_bars))
    if warm <= 0 or train.empty or test.empty:
        return test.reset_index(drop=True), 0
    warm = min(warm, len(train))
    combined = pd.concat([train.iloc[-warm:], test], ignore_index=True)
    return combined, warm
