"""Fuzzy tool-name matching for the interception handlers (`verifiers.v1.intercepts`).

Tool names vary by provider and harness — Claude's `web_search_20250305`, Codex's
`web_search_preview`, `Bash` vs `shell` vs `run_command` — so a pattern matches a name
fuzzily: normalized equality, either-side containment, or a shared `TOOL_SYNONYMS` group.
"""

from __future__ import annotations

import re

# Canonical tool group -> aliases. A pattern naming the canonical or any alias matches
# every member of the group.
TOOL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "web_search": (
        "web_search_preview",
        "web_search_call",
        "google_search",
        "bing_search",
        "brave_search",
        "duckduckgo_search",
        "tavily_search",
    ),
    "bash": (
        "shell",
        "shell_command",
        "run_command",
        "terminal",
        "console",
        "exec",
        "local_shell",
    ),
    "edit": (
        "apply_patch",
        "str_replace_editor",
        "text_editor",
        "edit_file",
        "write_file",
    ),
    "read": (
        "read_file",
        "open_file",
        "view_file",
        "get_file_contents",
    ),
}


def normalize_tool_name(name: str) -> str:
    """Lowercase, alnum only: `web_search_20250305` -> `websearch20250305`."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


_GROUPS = [
    frozenset(normalize_tool_name(alias) for alias in (canonical, *aliases))
    for canonical, aliases in TOOL_SYNONYMS.items()
]


def _in_group(group: frozenset[str], normalized: str) -> bool:
    """Group membership: an exact alias, or one carrying a date suffix
    (`websearch20250305` belongs to the `websearch` group)."""
    return any(
        normalized == member
        or (normalized.startswith(member) and normalized[len(member) :].isdigit())
        for member in group
    )


def match_tool(name: str, *patterns: str) -> bool:
    """True when `name` matches any pattern: normalized equality, either-side containment
    (`web_search` matches `web_search_call` and `web_search_20250305`), or a shared
    `TOOL_SYNONYMS` group (`bash` matches `shell`, `edit` matches `apply_patch`)."""
    normalized = normalize_tool_name(name)
    if not normalized:
        return False
    for pattern in patterns:
        pat = normalize_tool_name(pattern)
        if not pat:
            continue
        if normalized == pat or normalized in pat or pat in normalized:
            return True
        if any(
            _in_group(group, normalized) and _in_group(group, pat) for group in _GROUPS
        ):
            return True
    return False


__all__ = ["TOOL_SYNONYMS", "match_tool", "normalize_tool_name"]
