from __future__ import annotations

import threading
from pathlib import Path

from src.persistence import StateStore, _atomic_write_text


def test_corrupt_state_fail_closes_locked(tmp_path: Path):
    store = StateStore(tmp_path, "US30")
    store.set_locked(True, reason="dd")
    # Corrupt the on-disk file after a valid write.
    store.state_path.write_text("# STATE\n```yaml\n{{not: valid yaml\n```\n", encoding="utf-8")

    state = store.read_state()
    assert state["locked"] is True
    assert store.is_locked() is True
    # Persisted fail-closed rewrite.
    reloaded = store.read_state()
    assert reloaded["locked"] is True


def test_missing_locked_key_fail_closes(tmp_path: Path):
    store = StateStore(tmp_path, "US30")
    body = (
        "# STATE -- US30\n\n```yaml\n"
        "symbol: '#US30'\n"
        "equity: 1000\n"
        "```\n"
    )
    store.state_path.write_text(body, encoding="utf-8")
    state = store.read_state()
    assert state["locked"] is True


def test_update_state_preserves_lock_under_concurrent_writers(tmp_path: Path):
    store = StateStore(tmp_path, "US30")
    store.set_locked(True, reason="kill")

    errors: list[BaseException] = []

    def writer(i: int) -> None:
        try:
            for _ in range(40):
                store.update_state(equity=1000.0 + i, margin=0.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert store.read_state()["locked"] is True
    assert store.is_locked() is True


def test_atomic_write_creates_backup(tmp_path: Path):
    store = StateStore(tmp_path, "US30")
    store.update_state(equity=1.0)
    store.update_state(equity=2.0)
    backups = list((tmp_path / "US30").glob("STATE.md.*.bak"))
    assert len(backups) >= 1


def test_atomic_write_helper_replaces(tmp_path: Path):
    path = tmp_path / "f.txt"
    _atomic_write_text(path, "one")
    _atomic_write_text(path, "two")
    assert path.read_text(encoding="utf-8") == "two"
    assert not path.with_name("f.txt.tmp").exists()


def test_last_metrics_replaces_not_merges(tmp_path: Path):
    store = StateStore(tmp_path, "US30")
    store.update_state(
        last_metrics={
            "sharpe": 2.0,
            "pbo": 0.4,
            "dsr": 0.9,
            "outer_mean_sharpe": 1.5,
        }
    )
    store.update_state(
        last_metrics={
            "sharpe": 1.7,
            "max_drawdown": 0.05,
            "p_value": 0.02,
            "ic": 0.1,
            "oos_degradation": 0.1,
        }
    )
    metrics = store.read_state()["last_metrics"]
    assert metrics["sharpe"] == 1.7
    assert "pbo" not in metrics
    assert "dsr" not in metrics
    assert "outer_mean_sharpe" not in metrics
    assert metrics["max_drawdown"] == 0.05


def test_other_dicts_still_shallow_merge(tmp_path: Path):
    store = StateStore(tmp_path, "US30")
    store.update_state(params={"long_window": 200, "short_window": 40, "max_hold_bars": 12})
    store.update_state(params={"long_window": 220})
    params = store.read_state()["params"]
    assert params["long_window"] == 220
    assert params["short_window"] == 40
    assert params["max_hold_bars"] == 12
