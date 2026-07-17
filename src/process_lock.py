"""OS-level single-instance lock (cross-process; not threading.RLock)."""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_WIN_ERROR_ALREADY_EXISTS = 183


class ProcessLockError(RuntimeError):
    """Raised when another process already holds the instance lock."""


class ProcessLock:
    """Exclusive process lock so only one trading runtime may run.

    - Windows: named mutex ``Local\\AILoopTrade-<hash>`` (released on process exit).
    - POSIX: non-blocking ``fcntl.flock`` on a lock file under ``state/``.
    - Lock file always records the holder PID for operators.
    """

    def __init__(self, *, mutex_name: str, lock_path: Path) -> None:
        self.mutex_name = mutex_name
        self.lock_path = Path(lock_path)
        self._mutex_handle: Optional[int] = None
        self._lock_fd: Optional[int] = None
        self._held = False
        self._atexit_registered = False

    @classmethod
    def for_project(cls, project_root: Path, state_dir: str | Path) -> "ProcessLock":
        root = Path(project_root).resolve()
        digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:12]
        mutex_name = f"Local\\AILoopTrade-{digest}"
        state_root = Path(state_dir)
        if not state_root.is_absolute():
            state_root = root / state_root
        state_root.mkdir(parents=True, exist_ok=True)
        lock_path = state_root / ".ai_loop_trade.lock"
        return cls(mutex_name=mutex_name, lock_path=lock_path)

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns False if another instance holds it."""
        if self._held:
            return True

        if sys.platform == "win32":
            if not self._acquire_windows_mutex():
                return False
            # Mutex is authoritative on Windows; PID file is diagnostic only.
            self._write_pid_file()
        else:
            if not self._acquire_posix_file_lock():
                return False

        self._held = True
        if not self._atexit_registered:
            atexit.register(self.release)
            self._atexit_registered = True
        logger.info(
            "Process lock acquired pid=%s mutex=%s file=%s",
            os.getpid(),
            self.mutex_name if sys.platform == "win32" else "(posix-file)",
            self.lock_path,
        )
        return True

    def acquire_or_exit(self, *, exit_code: int = 1) -> None:
        """Acquire the lock or terminate the process immediately."""
        if self.acquire():
            return
        msg = (
            f"Another AI-Loop-Trade instance is already running "
            f"(lock={self.lock_path}"
            f"{f', mutex={self.mutex_name}' if sys.platform == 'win32' else ''}"
            f"). Exiting."
        )
        logger.error(msg)
        print(msg, file=sys.stderr)
        raise SystemExit(exit_code)

    def release(self) -> None:
        if not self._held and self._mutex_handle is None and self._lock_fd is None:
            return
        self._held = False
        self._release_posix_file_lock()
        self._release_windows_mutex()

    def __enter__(self) -> "ProcessLock":
        if not self.acquire():
            raise ProcessLockError(
                f"lock busy: {self.lock_path} / {self.mutex_name}"
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _write_pid_file(self) -> None:
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_path.write_text(f"{os.getpid()}\n", encoding="ascii")
        except OSError:
            logger.warning("Could not write PID lock file %s", self.lock_path)

    def _acquire_windows_mutex(self) -> bool:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        # Clear last error so a successful create is not confused with stale 183.
        ctypes.set_last_error(0)
        handle = kernel32.CreateMutexW(None, wintypes.BOOL(False), self.mutex_name)
        if not handle:
            err = ctypes.get_last_error()
            logger.error("CreateMutexW failed error=%s name=%s", err, self.mutex_name)
            return False
        if ctypes.get_last_error() == _WIN_ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            logger.error(
                "Windows mutex already held name=%s (another process is running)",
                self.mutex_name,
            )
            return False
        self._mutex_handle = int(handle)
        return True

    def _release_windows_mutex(self) -> None:
        if self._mutex_handle is None or sys.platform != "win32":
            self._mutex_handle = None
            return
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(wintypes.HANDLE(self._mutex_handle))
        self._mutex_handle = None

    def _acquire_posix_file_lock(self) -> bool:
        import fcntl

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            logger.error("Lock file busy: %s", self.lock_path)
            return False
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        except OSError:
            logger.warning("Could not write PID into %s", self.lock_path)
        self._lock_fd = fd
        return True

    def _release_posix_file_lock(self) -> None:
        fd = self._lock_fd
        self._lock_fd = None
        if fd is None:
            return
        try:
            if sys.platform != "win32":
                import fcntl

                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
