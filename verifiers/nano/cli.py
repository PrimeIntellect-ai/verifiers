"""The eval CLI: `python -m verifiers.nano <env-id> [options]`.

Mirrors the `~/prime-rl` entrypoint pattern (`config = cli(Config)`). The env id
is the first positional argument; it selects the env's `EnvConfig` subclass so
the single `prime-pydantic-config` parse keeps env-specific fields typed and
overridable via `--env.taskset.<field>` / `@ eval.toml`.
"""

import asyncio
import sys

from pydantic_config import cli

import verifiers.nano as vf

USAGE = (
    "usage: python -m verifiers.nano <env-id> [--model ... --num-tasks ... @ eval.toml]"
)


def peel_env_id(argv: list[str]) -> tuple[str, list[str]]:
    if argv and not argv[0].startswith(("-", "@")):
        return argv[0], argv[1:]
    raise SystemExit(USAGE)


def eval_config_type(env_id: str) -> type[vf.EvalConfig]:
    """The eval config type for `env_id`, narrowing `env` to the env's EnvConfig."""
    env_config_type = getattr(vf.import_env(env_id), "EnvConfig", vf.EnvConfig)
    if env_config_type is vf.EnvConfig:
        return vf.EvalConfig
    return type(
        "EvalConfig",
        (vf.EvalConfig,),
        {"__annotations__": {"env": env_config_type}, "env": env_config_type()},
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    env_id, rest = peel_env_id(argv)
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
