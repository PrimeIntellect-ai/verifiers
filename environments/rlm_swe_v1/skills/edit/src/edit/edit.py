"""Edit skill implementation."""

from __future__ import annotations

import os
from pathlib import Path


async def run(
    path: str,
    old_str: str,
    new_str: str,
    *,
    cwd: str | None = None,
) -> str:
    """Safe single-occurrence string replacement.

    Args:
        path: File path to edit.
        old_str: The exact string to find (must be unique).
        new_str: The replacement string.
        cwd: Working directory used to resolve the file path.

    Returns:
        Status message indicating success or a specific error.
    """
    base_dir = Path(cwd or os.getcwd())
    filepath = base_dir / path
    if not filepath.exists():
        return f"Error: {path} not found"
    try:
        content = filepath.read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"

    count = content.count(old_str)
    if count == 0:
        return f"Error: string not found in {path}"
    if count > 1:
        return f"Error: found {count} occurrences, need exactly 1"

    filepath.write_text(content.replace(old_str, new_str, 1))
    return f"Edited {path}"
