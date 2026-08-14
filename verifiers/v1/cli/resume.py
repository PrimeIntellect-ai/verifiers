"""Resume primitives shared by eval-like CLIs."""

from pathlib import Path


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


def split_resume(
    argv: list[str], command: str, *, allow_bare: bool = False
) -> tuple[Path | bool | None, list[str]]:
    """Pull ``--resume [<dir>]`` from argv, returning the dir and other arguments.

    With ``allow_bare``, a ``--resume`` without a value returns ``True`` — the caller
    resolves the run dir from the remaining arguments (e.g. ``--run.name``)."""
    for i, arg in enumerate(argv):
        if arg == "--resume":
            if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                if allow_bare:
                    return True, argv[:i] + argv[i + 1 :]
                raise SystemExit(
                    f"--resume needs an output dir: uv run {command} --resume <dir>"
                )
            return Path(argv[i + 1]), argv[:i] + argv[i + 2 :]
        if arg.startswith("--resume="):
            return Path(arg.split("=", 1)[1]), argv[:i] + argv[i + 1 :]
    return None, argv
