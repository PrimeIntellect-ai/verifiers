"""Edit skill implementation."""

import os
from pathlib import Path


async def run(
    path: str,
    old_str: str,
    new_str: str,
    *,
    cwd: str | None = None,
) -> str:
    """Replace exactly one occurrence of old_str in path."""
    base_dir = Path(cwd or os.getcwd())
    filepath = base_dir / path
    if not filepath.exists():
        return f"Error: {path} not found"
    try:
        content = filepath.read_text()
    except Exception as exc:
        return f"Error reading {path}: {exc}"

    count = content.count(old_str)
    if count == 0:
        return f"Error: string not found in {path}"
    if count > 1:
        return f"Error: found {count} occurrences, need exactly 1"

    filepath.write_text(content.replace(old_str, new_str, 1))
    return f"Edited {path}"
