# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Archive through directory descriptors without traversing dataset mounts."""

import fnmatch
import json
import os
import stat
import sys
import tarfile
from pathlib import Path


def archive_node(
    archive, name: str, parent: int | None, blocked: set[int], excludes: list[str]
) -> None:
    parts = name.split("/")
    if any(
        fnmatch.fnmatchcase("/".join(parts[index:]), pattern)
        for pattern in excludes
        for index in range(len(parts))
    ):
        return
    descriptor = os.open(
        "/" + name if parent is None else parts[-1],
        os.O_PATH | os.O_NOFOLLOW,
        dir_fd=parent,
    )
    try:
        # Mount IDs distinguish bind mounts even when they share a device number.
        metadata = Path(f"/proc/self/fdinfo/{descriptor}").read_text()
        mount = next(
            int(line.split()[1])
            for line in metadata.splitlines()
            if line.startswith("mnt_id:")
        )
        if mount in blocked:
            raise ValueError(f"artifact traverses a dataset mount: {name}")
        status = os.fstat(descriptor)
        info = tarfile.TarInfo(name)
        info.mode = stat.S_IMODE(status.st_mode)
        info.mtime = status.st_mtime
        if stat.S_ISREG(status.st_mode):
            info.size = status.st_size
            # Reopen the pinned inode, not its path, which the agent can replace.
            with open(f"/proc/self/fd/{descriptor}", "rb") as source:
                archive.addfile(info, source)
        elif stat.S_ISDIR(status.st_mode):
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
            for child in os.listdir(f"/proc/self/fd/{descriptor}"):
                archive_node(archive, f"{name}/{child}", descriptor, blocked, excludes)
        elif stat.S_ISLNK(status.st_mode):
            info.type = tarfile.SYMTYPE
            info.linkname = os.readlink("", dir_fd=descriptor)
            archive.addfile(info)
        elif stat.S_ISFIFO(status.st_mode):
            info.type = tarfile.FIFOTYPE
            archive.addfile(info)
        elif stat.S_ISCHR(status.st_mode) or stat.S_ISBLK(status.st_mode):
            info.type = (
                tarfile.CHRTYPE if stat.S_ISCHR(status.st_mode) else tarfile.BLKTYPE
            )
            info.devmajor = os.major(status.st_rdev)
            info.devminor = os.minor(status.st_rdev)
            archive.addfile(info)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    source, destination, blocked, excludes = json.loads(sys.argv[1])
    with (
        open(destination, "xb") as output,
        tarfile.open(fileobj=output, mode="w") as archive,
    ):
        archive_node(archive, source.lstrip("/"), None, set(blocked), excludes)
