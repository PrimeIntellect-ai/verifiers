from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
from pathlib import Path


def sibling_lock_path(path: Path, suffix: str = ".lock") -> Path:
    resolved = path.expanduser().resolve()
    return resolved.parent / f".{resolved.name}{suffix}"


@contextmanager
def shared_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path = lock_path.expanduser().resolve()
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
