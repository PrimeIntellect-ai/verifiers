"""Prune errored rollouts from a finished eval's `results.jsonl`, in place.

A run leaves permanent reward-0 rows when rollouts error (SandboxError/ProviderError/...).
`--prune <dir>` drops every row that ended with an error; `--prune-include` / `--prune-exclude`
narrow that by the error's exception type, with the same include/exclude semantics as
`--retries.rollout.include` / `exclude` and matched against the same most-recent `errors[].type`
(via the shared `error_type_selected`), so prune and retry stay parallel. It rewrites the file
atomically (reusing `resume.rewrite_results`) and prints a summary - no model, no config.
"""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pydantic_core import from_json

from verifiers.v1.cli.eval.resume import rewrite_results
from verifiers.v1.retries import error_type_selected


def _take_value(arg: str, argv: list[str], i: int, flag: str) -> tuple[bool, str, int]:
    """If `arg` is `flag` (value in the next token) or `flag=<value>`, return
    (True, value, next index); otherwise (False, "", i) unchanged."""
    if arg == flag:
        if i + 1 >= len(argv):
            raise SystemExit(
                f"{flag} needs a comma-separated list, e.g. {flag} SandboxError,ProviderError"
            )
        return True, argv[i + 1], i + 2
    if arg.startswith(flag + "="):
        return True, arg.split("=", 1)[1], i + 1
    return False, "", i


def split_prune(
    argv: list[str],
) -> tuple[bool, Path | None, list[str], list[str], list[str]]:
    """Pull `--prune [<dir>]`, `--prune-include <csv>`, and `--prune-exclude <csv>` out of argv,
    returning (prune requested, dir or None, include, exclude, the other args). `--prune` takes an
    output dir when standalone (`uv run eval --prune <dir>`), or is a bare modifier when combined
    with `--resume`, which then supplies the dir (`uv run eval --resume <dir> --prune`). `include`/
    `exclude` are comma-separated exception type names mirroring `--retries.rollout.include`/
    `exclude` (empty `include` = all error types not excluded). Modeled on `split_resume`."""
    prune = False
    prune_dir: Path | None = None
    include: list[str] = []
    exclude: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--prune":
            prune = True
            # A following non-flag token is the dir; otherwise `--prune` is a bare modifier
            # (combined with `--resume`, which supplies the dir).
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                prune_dir = Path(argv[i + 1])
                i += 2
            else:
                i += 1
            continue
        if arg.startswith("--prune="):
            prune = True
            value = arg.split("=", 1)[1]
            if not value:  # `--prune=` would otherwise become Path("") == cwd
                raise SystemExit("--prune needs an output dir, e.g. --prune=<dir>")
            prune_dir = Path(value)
            i += 1
            continue
        consumed, value, i = _take_value(arg, argv, i, "--prune-include")
        if consumed:
            include = _parse_csv(value)
            if (
                not include
            ):  # empty (e.g. `--prune-include=`) must fail, not vanish silently
                raise SystemExit(
                    "--prune-include needs a comma-separated list, e.g. --prune-include SandboxError"
                )
            continue
        consumed, value, i = _take_value(arg, argv, i, "--prune-exclude")
        if consumed:
            exclude = _parse_csv(value)
            if not exclude:
                raise SystemExit(
                    "--prune-exclude needs a comma-separated list, e.g. --prune-exclude ProviderError"
                )
            continue
        rest.append(arg)
        i += 1
    return prune, prune_dir, include, exclude, rest


def _parse_csv(csv: str) -> list[str]:
    return [t.strip() for t in csv.split(",") if t.strip()]


def _read_rows(results_path: Path) -> Iterator[tuple[int, str | None]]:
    """Stream `(byte offset, most-recent error type or None)` per row without retaining decoded
    traces. The most-recent error (`errors[-1]`) is the one retry matches on (`trace.error`), so
    prune selects the same way. Our own reader so the `resume` rewrite of `_read_results` can land
    independently."""
    with results_path.open("rb") as results:
        while True:
            offset = results.tell()
            line = results.readline()
            if not line:
                break
            if line.strip():
                try:
                    row = from_json(line)
                except ValueError:
                    row = json.loads(line)
                errors = row.get("errors") or []
                yield offset, (errors[-1]["type"] if errors else None)


def prune_results(prune_dir: Path, include: list[str], exclude: list[str]) -> None:
    """Drop errored rows from `<prune_dir>/results.jsonl`, in place. A row is pruned when it ended
    with an error whose type is selected by `include`/`exclude` (`error_type_selected` - the same
    rule as retry matching); clean rows and unselected errors are kept. Reuses
    `resume.rewrite_results` for the atomic write, then prints a summary."""
    if not prune_dir.is_dir():
        raise SystemExit(f"--prune: {prune_dir} is not a directory")
    results_path = prune_dir / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(
            f"--prune: no results.jsonl in {prune_dir} - not an eval output dir"
        )

    keep: list[int] = []
    pruned: Counter[str] = Counter()
    total = 0
    for offset, error_type in _read_rows(results_path):
        total += 1
        if error_type is not None and error_type_selected(error_type, include, exclude):
            pruned[error_type] += 1
            continue
        keep.append(offset)

    rewrite_results(prune_dir, keep)

    pruned_rows = total - len(keep)
    if pruned_rows:
        breakdown = ", ".join(f"{t} {n}" for t, n in pruned.most_common())
        print(
            f"pruned {pruned_rows} errored rows ({breakdown}); kept {len(keep)} of {total}"
        )
    else:
        print(f"nothing to prune in {prune_dir}: all {total} rows kept")
