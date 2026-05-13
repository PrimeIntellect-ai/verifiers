"""v1 eval entrypoint backed by pydantic-config (tyro + TOML).

Composes a `vf.Env` on the fly from the env package's `load_taskset(config)`
and a swappable harness package's `load_harness(config)`. Does not route
through `vf.load_environment`; legacy v0 `vf.Environment`-style envs are
not supported here — use `vf-eval` for those.

Examples:
    vf-eval-v1 reverse-text --help
    vf-eval-v1 reverse-text --harness-id opencode --help
    vf-eval-v1 reverse-text --taskset.dataset-split train -n 1 -r 1
    vf-eval-v1 reverse-text @ configs/eval/my-run.toml
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated, Any, cast

import tomllib

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import tyro
from pydantic import Field, create_model
from pydantic_config import BaseConfig, cli

import verifiers.v1 as vf
from verifiers.types import ClientConfig, GenerateOutputs

# --------------------------------------------------------------------------- #
# Top-level config. Pydantic-config owns validation; nested env- and
# harness-specific configs are typed dynamically per package.
# --------------------------------------------------------------------------- #


class AbstractEvalConfig(BaseConfig):
    """vf-eval-v1: evaluate a v1 environment via load_taskset + load_harness."""

    taskset_id: Annotated[str, tyro.conf.Positional] = Field(
        description="Taskset package name (resolves load_taskset).",
    )
    harness_id: str | None = Field(
        default=None,
        description="Harness package name (resolves load_harness).",
    )
    model: Annotated[str, tyro.conf.arg(aliases=["-m"])] = Field(
        default="openai/gpt-4.1-mini", description="Model id."
    )
    num_examples: Annotated[int, tyro.conf.arg(aliases=["-n"])] = Field(
        default=5, description="Examples to evaluate."
    )
    rollouts_per_example: Annotated[int, tyro.conf.arg(aliases=["-r"])] = Field(
        default=3, description="Rollouts per example."
    )


def build_eval_config_cls(
    taskset_cls: type[vf.TasksetConfig],
    harness_cls: type[vf.HarnessConfig],
) -> type[BaseConfig]:
    return create_model(
        "EvalConfig",
        __base__=AbstractEvalConfig,
        taskset=(taskset_cls, Field(default_factory=taskset_cls)),
        harness=(harness_cls, Field(default_factory=harness_cls)),
    )


# --------------------------------------------------------------------------- #
# Argv pre-scan: peek the taskset positional and --harness-id flag so we can
# discover the concrete config types before tyro builds the schema. Tyro
# still owns parsing/validation.
# --------------------------------------------------------------------------- #


def _load_toml(path: str) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, ValueError):
        return {}


def _peek_positional(argv: list[str]) -> str | None:
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "@":
            i += 2
            continue
        if not a.startswith("-"):
            return a
        i += 1
    return None


def _peek_flag(argv: list[str], flag: str) -> str | None:
    long = f"--{flag}"
    long_eq = f"--{flag}="
    for i, a in enumerate(argv):
        if a == long and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            return argv[i + 1]
        if a.startswith(long_eq):
            return a.split("=", 1)[1]
    for i, a in enumerate(argv):
        if a == "@" and i + 1 < len(argv):
            data = _load_toml(argv[i + 1])
            if isinstance(data.get(flag), str):
                return cast(str, data[flag])
    return None


# --------------------------------------------------------------------------- #
# Eval execution.
# --------------------------------------------------------------------------- #


def _resolve_harness_id(cfg: Any) -> str:
    """Pick the harness package: explicit --harness-id, else the env's own
    load_harness, else base."""
    if cfg.harness_id is not None:
        return cast(str, cfg.harness_id)
    env_module = vf.import_taskset_module(cfg.taskset_id)
    if hasattr(env_module, "load_harness"):
        return cast(str, cfg.taskset_id)
    return "base"


def _summarize(outputs: GenerateOutputs) -> None:
    rewards = [o["reward"] for o in outputs["outputs"] if o["reward"] is not None]
    if not rewards:
        print("no rewards recorded")
        return
    mean = sum(rewards) / len(rewards)
    print(f"\nrollouts: {len(rewards)}    mean reward: {mean:.4f}")


async def run(config: Any) -> None:
    taskset = vf.load_taskset(config.taskset_id, config.taskset)
    harness = vf.load_harness(_resolve_harness_id(config), config.harness)
    env = vf.Env(taskset=taskset, harness=harness)

    client_config = ClientConfig(
        client_type="openai_chat_completions",
        api_base_url="https://api.pinference.ai/api/v1",
        api_key_var="PRIME_API_KEY",
    )
    outputs = await env.evaluate(
        client=client_config,
        model=config.model,
        num_examples=config.num_examples,
        rollouts_per_example=config.rollouts_per_example,
    )
    _summarize(outputs)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Peek the selectors so we know which config types to use when building
    # the dynamic EvalConfig. Tyro still owns parsing/validation.
    taskset_id = _peek_positional(argv)
    harness_id = _peek_flag(argv, "harness-id")

    if taskset_id is not None:
        taskset_cls = vf.get_taskset_config_cls(vf.import_taskset_module(taskset_id))
    else:
        taskset_cls = vf.TasksetConfig

    fallback_harness = taskset_id if harness_id is None else harness_id
    if fallback_harness is not None:
        harness_cls = vf.get_harness_config_cls(
            vf.resolve_harness_module(fallback_harness)
        )
    else:
        harness_cls = vf.HarnessConfig

    EvalConfig = build_eval_config_cls(taskset_cls, harness_cls)
    config = cast(Any, cli(EvalConfig, args=argv))

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
