"""Chronological holdout and walk-forward fold helpers for nested search."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def chronological_holdout(
    df: pd.DataFrame,
    holdout_fraction: float = 0.15,
    *,
    min_holdout_bars: int = 40,
    min_train_bars: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into search window + final untouched holdout (time order)."""
    n = len(df)
    if n <= 0:
        empty = df.iloc[0:0].copy()
        return empty, empty

    frac = min(max(float(holdout_fraction), 0.0), 0.45)
    hold_n = int(round(n * frac)) if frac > 0 else 0
    hold_n = max(hold_n, min_holdout_bars if frac > 0 else 0)
    hold_n = min(hold_n, max(0, n - min_train_bars))
    if hold_n <= 0 or frac <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()

    split = n - hold_n
    train = df.iloc[:split].reset_index(drop=True)
    holdout = df.iloc[split:].reset_index(drop=True)
    return train, holdout


def walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 3,
    *,
    min_train_fraction: float = 0.40,
    min_train_bars: int = 80,
    min_test_bars: int = 30,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window walk-forward folds on pre-holdout data.

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
