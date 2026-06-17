"""Helpers for identifying test files touched by a unified diff."""


def _strip_git_prefix(path: str, prefix: str) -> str:
    path = path.strip()
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        path = path[1:-1]
    if path.startswith(prefix):
        return path[len(prefix) :]
    return path


def _iter_diff_headers(test_patch: str):
    lines = test_patch.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            minus = line[4:]
            j = i + 1
            plus = None
            while j < len(lines):
                candidate = lines[j]
                if candidate.startswith("+++ "):
                    plus = candidate[4:]
                    break
                if (
                    candidate.startswith("@@")
                    or candidate.startswith("diff --git")
                    or candidate.startswith("--- ")
                ):
                    break
                j += 1
            if plus is not None:
                yield minus, plus
                i = j + 1
                continue
        i += 1


def get_modified_files(test_patch: str) -> list[str]:
    """Return paths for files the diff modifies."""
    out: list[str] = []
    seen: set[str] = set()
    for minus, _ in _iter_diff_headers(test_patch):
        if minus.strip() == "/dev/null":
            continue
        path = _strip_git_prefix(minus, "a/").split("\t", 1)[0].rstrip()
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def get_new_files(test_patch: str) -> list[str]:
    """Return paths for files the diff creates."""
    out: list[str] = []
    seen: set[str] = set()
    for minus, plus in _iter_diff_headers(test_patch):
        if minus.strip() != "/dev/null":
            continue
        path = _strip_git_prefix(plus, "b/").split("\t", 1)[0].rstrip()
        if path and path not in seen:
            seen.add(path)
            out.append(path)
    return out
