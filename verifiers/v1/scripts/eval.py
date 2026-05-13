"""v1 eval entrypoint backed by pydantic-config (tyro + TOML).

Composes a `vf.Env` on the fly from the env package's `load_taskset(config)`
and a swappable harness package's `load_harness(config)`. Does not route
through `vf.load_environment`; legacy v0 `vf.Environment`-style envs are
not supported here — use `vf-eval` for those.

Examples:
    vf-eval-v1 --task reverse-text --help
    vf-eval-v1 --task reverse-text --harness opencode --help
    vf-eval-v1 --task reverse-text --taskset.dataset-split train --num-examples 1
    vf-eval-v1 @ configs/eval/my-run.toml
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import sys
import tomllib
from typing import Annotated, Any, cast, get_args, get_origin

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import tyro
from pydantic import Field, create_model
from pydantic_config import BaseConfig, cli

import verifiers.v1 as vf1
from verifiers.types import ClientConfig, GenerateOutputs

# --------------------------------------------------------------------------- #
# Env package discovery.
# --------------------------------------------------------------------------- #


def _import_env_module(env_id: str) -> Any:
    """Import an env package, preferring the `*_v1` module when present."""
    base = env_id.replace("-", "_").split("/")[-1]
    last_err: Exception | None = None
    for module_name in (base, f"{base}_v1"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            last_err = e
            continue
        if hasattr(module, "load_taskset"):
            return module
        last_err = AttributeError(f"{module_name!r} has no `load_taskset`")
    raise SystemExit(f"could not load env {env_id!r}: {last_err}")


def _config_type_from_annotation(annotation: Any, fallback: type) -> type:
    if annotation is inspect.Parameter.empty or annotation is None:
        return fallback
    if get_origin(annotation) is None:
        return annotation if isinstance(annotation, type) else fallback
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        if isinstance(arg, type):
            return arg
    return fallback


def _config_annotation(fn: Any, fallback: type) -> type:
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except Exception:
        hints = {}
    return _config_type_from_annotation(
        hints.get("config", inspect.Parameter.empty), fallback
    )


def _check_load_signature(fn: Any, fn_name: str) -> None:
    params = list(inspect.signature(fn).parameters.values())
    if len(params) != 1 or params[0].name != "config":
        raise SystemExit(
            f"{fn.__module__}.{fn_name} must take exactly one positional "
            f"`config` argument (got {[p.name for p in params]})."
        )


def _discover_taskset_config(module: Any) -> type[vf1.TasksetConfig]:
    _check_load_signature(module.load_taskset, "load_taskset")
    return cast(
        type[vf1.TasksetConfig],
        _config_annotation(module.load_taskset, vf1.TasksetConfig),
    )


def _discover_harness_config(module: Any) -> type[vf1.HarnessConfig]:
    if not hasattr(module, "load_harness"):
        return vf1.HarnessConfig
    _check_load_signature(module.load_harness, "load_harness")
    return cast(
        type[vf1.HarnessConfig],
        _config_annotation(module.load_harness, vf1.HarnessConfig),
    )


# --------------------------------------------------------------------------- #
# Named harness registry. `--harness <name>` swaps the env's default harness.
# Module paths (containing a dot) are imported directly.
# --------------------------------------------------------------------------- #


HARNESS_REGISTRY: dict[str, str] = {
    "base": "verifiers.v1.packages.harnesses.base",
    "opencode": "verifiers.v1.packages.harnesses.opencode",
    "rlm": "verifiers.v1.packages.harnesses.rlm",
    "pi": "verifiers.v1.packages.harnesses.pi",
    "mini-swe": "verifiers.v1.packages.harnesses.mini_swe_agent",
}


def _resolve_harness_module(name: str) -> Any:
    target = HARNESS_REGISTRY.get(name, name)
    try:
        module = importlib.import_module(target)
    except ImportError as e:
        raise SystemExit(f"could not import harness {name!r}: {e}") from e
    if not hasattr(module, "load_harness"):
        raise SystemExit(f"harness module {target!r} exposes no `load_harness`.")
    return module


# --------------------------------------------------------------------------- #
# Top-level config: only task, harness, model, rollout controls.
# `taskset` and `harness` sub-sections are typed dynamically per env/harness.
# --------------------------------------------------------------------------- #


class _EvalConfigBase(BaseConfig):
    """vf-eval-v1: evaluate a v1 environment via load_taskset + load_harness."""

    task: Annotated[str, tyro.conf.arg(help="Env package id.")]
    model: str = Field(default="openai/gpt-4.1-mini", description="Model id.")
    num_examples: int = Field(default=5, description="Examples to evaluate.")
    rollouts_per_example: int = Field(default=3, description="Rollouts per example.")
    max_concurrent: int = Field(default=32, description="Max concurrent rollouts.")
    max_tokens: int | None = Field(default=None, description="Max output tokens.")
    temperature: float | None = Field(default=None, description="Sampling temp.")


def _build_eval_config_cls(
    taskset_cls: type[vf1.TasksetConfig],
    harness_cls: type[vf1.HarnessConfig],
) -> type[BaseConfig]:
    return create_model(
        "EvalConfigV1",
        __base__=_EvalConfigBase,
        taskset=(taskset_cls, Field(default_factory=taskset_cls)),
        harness=(harness_cls, Field(default_factory=harness_cls)),
    )


# --------------------------------------------------------------------------- #
# Argv pre-scan: extract `--task` and `--harness <name>` before tyro parses.
# --------------------------------------------------------------------------- #


def _load_toml(path: str) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, ValueError):
        return {}


def _peek_task(argv: list[str]) -> str | None:
    for i, a in enumerate(argv):
        if a == "--task" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--task="):
            return a.split("=", 1)[1]
    for i, a in enumerate(argv):
        if a == "@" and i + 1 < len(argv):
            data = _load_toml(argv[i + 1])
            if "task" in data:
                return str(data["task"])
    return None


def _extract_harness_selector(argv: list[str]) -> tuple[str | None, list[str]]:
    """Pop `--harness <name>` from argv, leaving nested `--harness.<field>`."""
    out: list[str] = []
    selector: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--harness" and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            selector = argv[i + 1]
            i += 2
            continue
        if a.startswith("--harness=") and "." not in a.split("=", 1)[0]:
            selector = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    if selector is None:
        for i, a in enumerate(argv):
            if a == "@" and i + 1 < len(argv):
                data = _load_toml(argv[i + 1])
                harness = data.get("harness")
                if isinstance(harness, str):
                    selector = harness
                    break
    return selector, out


# --------------------------------------------------------------------------- #
# Eval execution.
# --------------------------------------------------------------------------- #


def _build_env(env_module: Any, harness_module: Any, cfg: Any) -> vf1.Env:
    taskset = env_module.load_taskset(cfg.taskset)
    harness = harness_module.load_harness(cfg.harness)
    if not isinstance(taskset, vf1.Taskset):
        raise SystemExit(
            f"{env_module.__name__}.load_taskset must return a vf.Taskset."
        )
    if not isinstance(harness, vf1.Harness):
        raise SystemExit(
            f"{harness_module.__name__}.load_harness must return a vf.Harness."
        )
    return vf1.Env(taskset=taskset, harness=harness)


def _summarize(outputs: GenerateOutputs) -> None:
    rewards = [o["reward"] for o in outputs["outputs"] if o["reward"] is not None]
    if not rewards:
        print("no rewards recorded")
        return
    mean = sum(rewards) / len(rewards)
    print(f"\nrollouts: {len(rewards)}    mean reward: {mean:.4f}")


async def _run(env_module: Any, harness_module: Any, cfg: Any) -> None:
    env = _build_env(env_module, harness_module, cfg)
    sampling_args: dict[str, Any] = {}
    if cfg.max_tokens is not None:
        sampling_args["max_tokens"] = cfg.max_tokens
    if cfg.temperature is not None:
        sampling_args["temperature"] = cfg.temperature
    client_config = ClientConfig(
        client_type="openai_chat_completions",
        api_base_url="https://api.pinference.ai/api/v1",
        api_key_var="PRIME_API_KEY",
    )
    outputs = await env.evaluate(
        client=client_config,
        model=cfg.model,
        sampling_args=sampling_args,
        num_examples=cfg.num_examples,
        rollouts_per_example=cfg.rollouts_per_example,
        max_concurrent=cfg.max_concurrent,
    )
    _summarize(outputs)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    harness_selector, argv = _extract_harness_selector(argv)

    task = _peek_task(argv)
    if task is None:
        env_module = None
        taskset_cls: type[vf1.TasksetConfig] = vf1.TasksetConfig
    else:
        env_module = _import_env_module(task)
        taskset_cls = _discover_taskset_config(env_module)

    if harness_selector is not None:
        harness_module = _resolve_harness_module(harness_selector)
    elif env_module is not None and hasattr(env_module, "load_harness"):
        harness_module = env_module
    else:
        harness_module = importlib.import_module(HARNESS_REGISTRY["base"])
    harness_cls = _discover_harness_config(harness_module)

    EvalConfigCls = _build_eval_config_cls(taskset_cls, harness_cls)
    cfg = cast(Any, cli(EvalConfigCls, args=argv))

    resolved_task = cfg.task
    if env_module is None or resolved_task != task:
        env_module = _import_env_module(resolved_task)

    asyncio.run(_run(env_module, harness_module, cfg))


if __name__ == "__main__":
    main()
