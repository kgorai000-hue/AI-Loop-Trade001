from __future__ import annotations

import pandas as pd

from src.splits import (
    chronological_rolling_oos,
    max_evaluated_oos_end,
    walk_forward_folds,
    with_warmup,
)


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC"),
            "close": range(n),
        }
    )


def test_chronological_rolling_oos_is_tail_only():
    df = _frame(200)
    train, hold, win = chronological_rolling_oos(
        df, 0.15, min_oos_bars=20, min_train_bars=50
    )
    assert win is not None
    assert len(train) + len(hold) == 200
    assert hold["time"].iloc[0] > train["time"].iloc[-1]
    assert len(hold) >= 20


def test_rolling_oos_excludes_previously_evaluated_end():
    df = _frame(400)
    _, first, win = chronological_rolling_oos(
        df, 0.15, min_oos_bars=20, min_train_bars=50
    )
    assert win is not None
    assert len(first) >= 20

    extra = pd.DataFrame(
        {
            "time": pd.date_range(
                df["time"].iloc[-1] + pd.Timedelta(minutes=30),
                periods=80,
                freq="30min",
                tz="UTC",
            ),
            "close": range(400, 480),
        }
    )
    grown = pd.concat([df, extra], ignore_index=True)
    excl = max_evaluated_oos_end([win])
    search, rolling, win2 = chronological_rolling_oos(
        grown,
        0.15,
        min_oos_bars=20,
        min_train_bars=50,
        exclude_oos_before=excl,
    )
    assert win2 is not None
    assert rolling["time"].iloc[0] > excl
    assert search["time"].iloc[-1] < rolling["time"].iloc[0]


def test_rolling_oos_fail_closed_when_no_new_bars():
    df = _frame(200)
    _, _, win = chronological_rolling_oos(df, 0.15, min_oos_bars=20, min_train_bars=50)
    search, rolling, win2 = chronological_rolling_oos(
        df,
        0.15,
        min_oos_bars=20,
        min_train_bars=50,
        exclude_oos_before=pd.Timestamp(win["end"]),
    )
    assert rolling.empty
    assert win2 is None
    assert len(search) == 200


def test_walk_forward_folds_never_leak_future_into_train():
    df = _frame(300)
    folds = walk_forward_folds(
        df, n_folds=3, min_train_fraction=0.4, min_train_bars=50, min_test_bars=20
    )
    assert len(folds) >= 2
    for train, test in folds:
        assert train["time"].iloc[-1] < test["time"].iloc[0]
        assert train["time"].iloc[0] == df["time"].iloc[0]


def test_with_warmup_prepends_train_tail():
    df = _frame(50)
    train, test = df.iloc[:30], df.iloc[30:]
    combined, warm = with_warmup(train, test, warmup_bars=5)
    assert warm == 5
    assert len(combined) == len(test) + 5
    assert combined["close"].iloc[0] == train["close"].iloc[-5]
