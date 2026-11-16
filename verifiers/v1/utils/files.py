"""Bounded reads of regular files, including paths relative to directory handles."""

import os
import stat
from pathlib import Path


def read_regular(path: str | Path, limit: int, *, dir_fd: int | None = None) -> bytes:
    """Read at most `limit` bytes, rejecting final symlinks and non-regular files."""
    if limit < 0:
        raise ValueError("file byte budget must be non-negative")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd
    )
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"not a regular file: {path}")
        if info.st_size > limit or len(data := stream.read(limit + 1)) > limit:
            raise ValueError(f"file exceeds {limit} bytes: {path}")
        return data
