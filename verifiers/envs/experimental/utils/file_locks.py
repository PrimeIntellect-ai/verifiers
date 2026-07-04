from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl

    _HAVE_FCNTL = True
except ImportError:
    # fcntl is Unix-only. On Windows, fall back to msvcrt byte-range locking.
    # msvcrt has no shared locks, so shared locks degrade to exclusive locks
    # there, which is strictly more conservative.
    import msvcrt

    _HAVE_FCNTL = False


def sibling_lock_path(path: Path, suffix: str = ".lock") -> Path:
    resolved = path.expanduser().resolve()
    return resolved.parent / f".{resolved.name}{suffix}"


@contextmanager
def _msvcrt_file_lock(lock_path: Path, *, nonblocking: bool = False) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        lock_file.seek(0)
        if nonblocking:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                # Match the fcntl path, which raises BlockingIOError when a
                # nonblocking lock is contended; msvcrt raises a plain OSError.
                raise BlockingIOError(
                    exc.errno,
                    f"lock is held by another process: {lock_path}",
                ) from exc
        else:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            try:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass


@contextmanager
def shared_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path = lock_path.expanduser().resolve()
    if not _HAVE_FCNTL:
        with _msvcrt_file_lock(lock_path):
            yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        yield


@contextmanager
def exclusive_file_lock(
    lock_path: Path,
    *,
    nonblocking: bool = False,
) -> Iterator[None]:
    lock_path = lock_path.expanduser().resolve()
    if not _HAVE_FCNTL:
        with _msvcrt_file_lock(lock_path, nonblocking=nonblocking):
            yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), flags)
        yield


@contextmanager
def shared_path_lock(path: Path, suffix: str = ".lock") -> Iterator[None]:
    with shared_file_lock(sibling_lock_path(path, suffix)):
        yield


@contextmanager
def exclusive_path_lock(
    path: Path,
    *,
    suffix: str = ".lock",
    nonblocking: bool = False,
) -> Iterator[None]:
    with exclusive_file_lock(
        sibling_lock_path(path, suffix),
        nonblocking=nonblocking,
    ):
        yield
