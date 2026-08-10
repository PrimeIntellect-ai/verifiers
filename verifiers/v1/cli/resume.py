"""Resume primitives shared by eval-like CLIs."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def task_key(data: Mapping) -> str:
    """Content identity for task wire data, independent of field order."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def distribute(
    selected_keys: list[str], owed: dict[str, int], num_results: int
) -> list[int]:
    """Spread each key's owed results over its selected instances, in order."""
    remaining = dict(owed)
    counts: list[int] = []
    for key in selected_keys:
        take = min(num_results, remaining.get(key, 0))
        if take:
            remaining[key] -= take
        counts.append(take)
    return counts


def split_resume(argv: list[str], command: str) -> tuple[Path | None, list[str]]:
    """Pull ``--resume <dir>`` from argv, returning the dir and other arguments."""
    for i, arg in enumerate(argv):
        if arg == "--resume":
            if i + 1 >= len(argv):
                raise SystemExit(
                    f"--resume needs an output dir: uv run {command} --resume <dir>"
                )
            return Path(argv[i + 1]), argv[:i] + argv[i + 2 :]
        if arg.startswith("--resume="):
            return Path(arg.split("=", 1)[1]), argv[:i] + argv[i + 1 :]
    return None, argv
