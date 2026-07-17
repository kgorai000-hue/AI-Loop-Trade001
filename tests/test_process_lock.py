from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.process_lock import ProcessLock, ProcessLockError


def test_process_lock_exclusive(tmp_path: Path):
    a = ProcessLock(mutex_name="Local\\AILoopTrade-test-a", lock_path=tmp_path / "a.lock")
    b = ProcessLock(mutex_name="Local\\AILoopTrade-test-a", lock_path=tmp_path / "a.lock")

    assert a.acquire() is True
    assert a.held is True
    assert b.acquire() is False
    a.release()
    assert b.acquire() is True
    b.release()


def test_process_lock_context_manager(tmp_path: Path):
    path = tmp_path / "ctx.lock"
    name = "Local\\AILoopTrade-test-ctx"
    with ProcessLock(mutex_name=name, lock_path=path) as held:
        assert held.held is True
        second = ProcessLock(mutex_name=name, lock_path=path)
        with pytest.raises(ProcessLockError):
            second.__enter__()


def test_acquire_or_exit_raises_systemexit(tmp_path: Path):
    path = tmp_path / "exit.lock"
    name = "Local\\AILoopTrade-test-exit"
    first = ProcessLock(mutex_name=name, lock_path=path)
    first.acquire()
    second = ProcessLock(mutex_name=name, lock_path=path)
    with pytest.raises(SystemExit) as exc:
        second.acquire_or_exit(exit_code=7)
    assert exc.value.code == 7
    first.release()


def test_for_project_uses_state_dir(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    state = root / "state"
    lock = ProcessLock.for_project(root, "state")
    assert lock.lock_path == state / ".ai_loop_trade.lock"
    assert lock.acquire() is True
    assert lock.lock_path.exists()
    assert lock.lock_path.read_text(encoding="ascii").strip().isdigit()
    lock.release()


@pytest.mark.skipif(sys.platform == "win32", reason="posix flock path")
def test_posix_second_process_blocked_via_subprocess(tmp_path: Path):
    """Ensure flock is visible across processes (not only threads)."""
    import subprocess
    import textwrap

    lock_path = tmp_path / "sub.lock"
    script = textwrap.dedent(
        f"""
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from src.process_lock import ProcessLock
        lock = ProcessLock(mutex_name="x", lock_path=Path({str(lock_path)!r}))
        ok = lock.acquire()
        print("OK" if ok else "BUSY")
        if ok:
            time.sleep(2)
            lock.release()
        """
    )
    # Hold lock in this process
    holder = ProcessLock(mutex_name="x", lock_path=lock_path)
    assert holder.acquire() is True
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "BUSY" in proc.stdout
    holder.release()
