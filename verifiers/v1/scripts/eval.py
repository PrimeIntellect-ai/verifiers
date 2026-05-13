"""v1 eval entrypoint backed by pydantic-config (tyro + TOML).

Composes a `vf.Env` on the fly from the env package's `load_taskset(config)`
and a swappable harness package's `load_harness(config)`. Does not route
through `vf.load_environment`; legacy v0 `vf.Environment`-style envs are
not supported here — use `vf-eval` for those.

Examples:
    vf-eval-v1 --taskset-id reverse-text --help
    vf-eval-v1 --taskset-id reverse-text --harness-id opencode --help
    vf-eval-v1 --taskset-id reverse-text --taskset.dataset-split train -n 1 -r 1
    vf-eval-v1 --taskset-id reverse-text @ configs/eval/my-run.toml
"""

from __future__ import annotations

import argparse
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


class EvalConfig(BaseConfig):
    """vf-eval-v1: evaluate a v1 environment via load_taskset + load_harness."""

    taskset_id: str = Field(
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
        "ResolvedEvalConfig",
        __base__=EvalConfig,
        taskset=(taskset_cls, Field(default_factory=taskset_cls)),
        harness=(harness_cls, Field(default_factory=harness_cls)),
    )


# --------------------------------------------------------------------------- #
# Selector pre-resolution: argparse for the lightweight first pass (ignores
# unknown args via parse_known_args), plus a peek into any `@ file.toml`.
# The full pydantic-config parse runs afterwards once we know which concrete
# config types to plug into the dynamic EvalConfig.
# --------------------------------------------------------------------------- #


def _resolve_selectors(argv: list[str]) -> tuple[str | None, str | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--taskset-id")
    parser.add_argument("--harness-id")
    known, _ = parser.parse_known_args(argv)
    taskset_id, harness_id = known.taskset_id, known.harness_id

    for i, a in enumerate(argv):
        if a == "@" and i + 1 < len(argv):
            try:
                with open(argv[i + 1], "rb") as f:
                    data = tomllib.load(f)
            except (OSError, ValueError):
                continue
            if taskset_id is None and isinstance(data.get("taskset_id"), str):
                taskset_id = data["taskset_id"]
            if harness_id is None and isinstance(data.get("harness_id"), str):
                harness_id = data["harness_id"]
    return taskset_id, harness_id


# --------------------------------------------------------------------------- #
# Eval execution.
# --------------------------------------------------------------------------- #


async def run(config: Any) -> None:
    taskset = vf.load_taskset(config.taskset_id, config.taskset)
    harness = vf.load_harness(config.harness_id, config.harness)
    env = vf.Env(taskset=taskset, harness=harness)

    outputs = await env.evaluate(
        client=ClientConfig(),
        model=config.model,
        num_examples=config.num_examples,
        rollouts_per_example=config.rollouts_per_example,
    )

    def summarize_outputs(outputs: GenerateOutputs) -> None:
        rewards = [o["reward"] for o in outputs["outputs"] if o["reward"] is not None]
        if not rewards:
            print("no rewards recorded")
            return
        mean = sum(rewards) / len(rewards)
        print(f"\nrollouts: {len(rewards)}    mean reward: {mean:.4f}")

    summarize_outputs(outputs)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Resolve --taskset-id / --harness-id first so we know which concrete
    # config types to plug into the dynamic EvalConfig.
    taskset_id, harness_id = _resolve_selectors(argv)

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

    ResolvedEvalConfig = build_eval_config_cls(taskset_cls, harness_cls)
    config = cast(Any, cli(ResolvedEvalConfig, args=argv))

    # Fall back to the env's own load_harness, else base, when --harness-id
    # was not explicitly set.
    if config.harness_id is None:
        config.harness_id = fallback_harness or "base"

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
