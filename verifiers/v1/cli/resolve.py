"""Shared CLI resolution: select the taskset + harness by their `id`.

Both entrypoints (`cli/eval.py`, `cli/serve.py`) select their plugins the same way — a `.id`
per plugin (`--taskset.id`, `--harness.id`), the discriminator. We read those ids from argv
*before* the typed parse, resolve each to its config type (a package imported by id), and
narrow the `taskset` / `harness` fields of the base config. So the single `cli()` parse keeps
both typed and `-h` shows the resolved config types.

The taskset id has a positional shorthand: a leading bare token is the taskset id
(`eval gsm8k` == `eval --taskset.id gsm8k`). Narrowing reads ids from the CLI only — a config
file (`@ file.toml`) needn't repeat them: `EnvConfig._resolve_plugins` resolves each field to
its specific type from the parsed data at validation time (the path prime-rl uses), so a
fully-specified config runs with just `@ file.toml`. We only avoid pre-narrowing a field a
config file may set, and leave that to the validator.
"""

from pathlib import Path

import verifiers.v1 as vf


def with_positional_taskset(argv: list[str]) -> list[str]:
    """A leading bare token is the taskset id — `eval gsm8k` == `eval --taskset.id gsm8k`."""
    if argv and not argv[0].startswith(("-", "@")):
        return ["--taskset.id", argv[0], *argv[1:]]
    return argv


def references_config_file(argv: list[str]) -> bool:
    """Whether argv points at a `@ file.toml` (any token starting with `@`). Its ids are
    resolved by `EnvConfig` from the parsed data, so the CLI needn't repeat them."""
    return any(arg.startswith("@") for arg in argv)


def extract_id(argv: list[str], field: str, default: str = "") -> str:
    """The chosen `<field>.id` from `--<field>.id <x>` (or `=<x>`) on the CLI, before the typed
    parse (the positional taskset shorthand is applied upstream). Absent here, the id can still
    come from a `@ file.toml` — `EnvConfig` resolves it from the parsed data."""
    flag = f"--{field}.id"
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def narrow_config(base: type, argv: list[str]) -> type:
    """`base` (an `EnvConfig` subclass) with `taskset`/`harness` narrowed to the config types
    of the ids given on the CLI — so the single `cli()` parse stays typed and `-h` renders
    them. A field whose id isn't on the CLI is left as the base type for `EnvConfig` to resolve
    from a `@ file.toml` (never pre-narrowed to a type the config could then contradict).
    Absent a config file, the harness falls back to `default` so bare `-h` still renders it."""
    taskset_id = extract_id(argv, "taskset")
    harness_id = extract_id(
        argv, "harness", "" if references_config_file(argv) else "default"
    )
    annotations: dict[str, type] = {}
    fields: dict[str, object] = {}
    for field, resolve, ident in (
        ("taskset", vf.taskset_config_type, taskset_id),
        ("harness", vf.harness_config_type, harness_id),
    ):
        if ident:
            ftype = resolve(ident)
            annotations[field] = ftype
            fields[field] = ftype(id=ident)
    return type(base.__name__, (base,), {"__annotations__": annotations, **fields})


def local_examples(rel: str) -> list[str]:
    """Best-effort: plugin ids found under a local examples dir, for the help hint only.
    A plugin is resolved by id (or path) at run time — we just import it and use its load
    hook — so this is not an authoritative list, and it's empty outside this repo."""
    root = Path(rel)
    if not root.is_dir():
        return []
    return sorted(
        path.name.replace("_", "-")
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith((".", "__"))
    )
