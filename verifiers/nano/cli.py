"""The eval CLI: `python -m verifiers.nano <env-id> [options]`.

Mirrors the `~/prime-rl` entrypoint pattern (`config = cli(Config)`). The env id
is the first positional argument; it selects the env's `EnvConfig` subclass so
the single `prime-pydantic-config` parse keeps env-specific fields typed and
overridable via dotted flags (e.g. `--env.runtime.kind docker`, `--env.agent.kind rlm`) / `@ eval.toml`.

`-h`/`--help` (or no env id) prints the available envs plus the full, rich
pydantic-config option help; `<env-id> --help` prints that env's typed help.
"""

import asyncio
import sys
from pathlib import Path

from pydantic_config import cli

import verifiers.nano as vf
from verifiers.nano import examples

USAGE = "usage: python -m verifiers.nano <env-id> [options] [@ file.toml]"


def available_envs() -> list[str]:
    """The built-in example env ids (hyphenated module names under examples/)."""
    return sorted(
        path.stem.replace("_", "-")
        for path in Path(examples.__file__).parent.glob("*.py")
        if path.stem != "__init__"
    )


def eval_config_type(env_id: str) -> type[vf.EvalConfig]:
    """The eval config type for `env_id`, narrowing `env` to the env's EnvConfig."""
    env_config_type = getattr(vf.import_env(env_id), "EnvConfig", vf.EnvConfig)
    if env_config_type is vf.EvalConfig:
        return vf.EvalConfig
    return type(
        "EvalConfig",
        (vf.EvalConfig,),
        {"__annotations__": {"env": env_config_type}, "env": env_config_type()},
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)

    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        print("\nexample envs:", ", ".join(available_envs()))
        sys.argv = [sys.argv[0], "--help"]
        cli(vf.EvalConfig)  # renders the full option help (incl. runtimes), then exits
        return
    if argv[0].startswith(("-", "@")):
        raise SystemExit(USAGE)  # options given but no env id

    env_id, rest = argv[0], argv[1:]
    config_type = eval_config_type(env_id)
    sys.argv = [sys.argv[0], *rest]  # let prime-pydantic-config render help/errors
    config = cli(config_type)
    config.id = env_id

    env = vf.load_environment(env_id, config.env)
    transcripts, metadata = asyncio.run(vf.run_eval(env, config))
    for transcript in transcripts:
        print(transcript.model_dump_json(indent=2, exclude_none=True))
    print(
        f"avg_reward={metadata.avg_reward:.3f}  "
        f"rewards={metadata.avg_rewards}  metrics={metadata.avg_metrics}"
    )
