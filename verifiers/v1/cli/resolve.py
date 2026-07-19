"""Shared CLI resolution: select the environment (and its taskset) by id.

The ids are read from argv *before* the typed parse and the base config's `env`
field is narrowed to the selected env's config class, so the single `cli()` parse
stays typed and `-h` renders the real flags. Ids can also come from a `@ file.toml`:
validation narrows from the parsed data (`resolve_env_field`), so only what the CLI
states explicitly is pre-narrowed — never to a type a config file could contradict.
"""

import verifiers.v1 as vf


def with_positional_taskset(
    argv: list[str], flag: str = "--env.taskset.id"
) -> list[str]:
    """A leading bare token is the taskset id — `eval gsm8k` == `eval --env.taskset.id
    gsm8k` (the taskset-first CLIs `validate`/`debug` pass their own root flag)."""
    if argv and not argv[0].startswith(("-", "@")):
        return [flag, argv[0], *argv[1:]]
    return argv


def references_config_file(argv: list[str]) -> bool:
    """Whether argv points at a `@ file.toml` (any token starting with `@`). Its ids
    are resolved from the parsed data at validation time, so the CLI needn't repeat
    them."""
    return any(arg.startswith("@") for arg in argv)


def extract_id(argv: list[str], field: str, default: str = "") -> str:
    """The chosen `<field>.id` from `--<field>.id <x>` (or `=<x>`) on the CLI, before
    the typed parse (the positional taskset shorthand is applied upstream). Absent
    here, the id can still come from a `@ file.toml` — validation resolves it from
    the parsed data."""
    flag = f"--{field}.id"
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def narrow_config(base: type, argv: list[str]) -> type:
    """`base` with its `env` field narrowed to the config class of the env the CLI
    names (`--env.id`, else the taskset's own env, else the single-agent env),
    including its `taskset` sub-field. Ids a config file may set are left to the
    validator; an explicit CLI `--env.id` always narrows (the id is
    authoritative)."""
    taskset_id = extract_id(argv, "env.taskset")
    env_id = extract_id(argv, "env")
    if not env_id and (not taskset_id or references_config_file(argv)):
        return base
    env_type = vf.env_config_type(taskset_id, env_id)
    if taskset_id:
        # Nested narrowing: `--env.taskset.<knob>` parses typed and renders in -h.
        taskset_type = vf.taskset_config_type(taskset_id)
        env_type = type(
            env_type.__name__,
            (env_type,),
            {
                "__annotations__": {"taskset": taskset_type},
                "taskset": taskset_type(id=taskset_id),
            },
        )
    default = env_type(id=env_id) if env_id else env_type()
    return type(
        base.__name__,
        (base,),
        {"__annotations__": {"env": env_type}, "env": default},
    )
