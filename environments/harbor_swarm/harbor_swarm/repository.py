"""UTF-8 repository snapshots transferred through native runtime artifacts."""

import io
import tarfile
from pathlib import PurePosixPath


def archive(workspace: str, files: dict[str, str]) -> dict[str, bytes | None]:
    root = PurePosixPath(workspace)
    if not root.is_absolute() or root == PurePosixPath("/") or ".." in root.parts:
        raise ValueError("Workspace must be an absolute directory below /")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as tar:
        directory = tarfile.TarInfo(str(root).lstrip("/"))
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        tar.addfile(directory)
        for path, content in sorted(files.items()):
            relative = PurePosixPath(path)
            if relative.is_absolute() or any(
                p in {"", ".", "..", ".git"} for p in path.split("/")
            ):
                raise ValueError("Unsafe repository path")
            data = content.encode()
            entry = tarfile.TarInfo(str(root / relative).lstrip("/"))
            entry.size, entry.mode = len(data), 0o644
            tar.addfile(entry, io.BytesIO(data))
    return {workspace: output.getvalue()}


# Runs only in a fresh seed runtime, before any agent has access.
READ_WORKSPACE = """import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = {}
size = 0
for path in sorted(root.rglob('*')):
    if path.is_symlink():
        raise ValueError('Seed workspace contains a symlink')
    if not path.is_file():
        continue
    if len(files) >= 1000 or path.stat().st_size + size > 2 * 1024 * 1024:
        raise ValueError('Seed workspace exceeds 1000 files or 2 MiB')
    data = path.read_bytes()
    size += len(data)
    files[str(path.relative_to(root))] = data.decode('utf-8')
print(json.dumps(files))
"""
