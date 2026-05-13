"""v1 eval entrypoint backed by pydantic-config (tyro + TOML).

Composes a `vf.Env` on the fly from the env package's `load_taskset(config)`
and a swappable harness package's `load_harness(config)`. Does not route
through `vf.load_environment`; legacy v0 `vf.Environment`-style envs are
not supported here — use `vf-eval` for those.

Examples:
    vf-eval-v1 --taskset reverse-text --help
    vf-eval-v1 --taskset reverse-text --harness opencode --help
    vf-eval-v1 --taskset reverse-text --taskset-config.dataset-split train
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

import verifiers.v1 as vf
from verifiers.types import ClientConfig, GenerateOutputs

# --------------------------------------------------------------------------- #
# Env / harness module discovery.
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


def _discover_taskset_config(module: Any) -> type[vf.TasksetConfig]:
    _check_load_signature(module.load_taskset, "load_taskset")
    return cast(
        type[vf.TasksetConfig],
        _config_annotation(module.load_taskset, vf.TasksetConfig),
    )


def _discover_harness_config(module: Any) -> type[vf.HarnessConfig]:
    if not hasattr(module, "load_harness"):
        return vf.HarnessConfig
    _check_load_signature(module.load_harness, "load_harness")
    return cast(
        type[vf.HarnessConfig],
        _config_annotation(module.load_harness, vf.HarnessConfig),
    )


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
# Top-level config. Pydantic-config owns required-field validation. The
# nested env-specific configs live under `taskset_config` / `harness_config`
# so they do not collide with the `--taskset` / `--harness` selector flags.
# --------------------------------------------------------------------------- #


class _EvalConfigBase(BaseConfig):
    """vf-eval-v1: evaluate a v1 environment via load_taskset + load_harness."""

    taskset: Annotated[
        str, tyro.conf.arg(help="Env package id (resolves load_taskset).")
    ]
    harness: str | None = Field(
        default=None,
        description=(
            "Harness package name from the registry (base / opencode / rlm / pi / "
            "mini-swe), or a Python module path. Defaults to the env's own "
            "load_harness when present, else base."
        ),
    )
    model: str = Field(default="openai/gpt-4.1-mini", description="Model id.")
    num_examples: int = Field(default=5, description="Examples to evaluate.")
    rollouts_per_example: int = Field(default=3, description="Rollouts per example.")


def _build_eval_config_cls(
    taskset_cls: type[vf.TasksetConfig],
    harness_cls: type[vf.HarnessConfig],
) -> type[BaseConfig]:
    return create_model(
        "EvalConfigV1",
        __base__=_EvalConfigBase,
        taskset_config=(taskset_cls, Field(default_factory=taskset_cls)),
        harness_config=(harness_cls, Field(default_factory=harness_cls)),
    )


# --------------------------------------------------------------------------- #
# Argv pre-scan: peek `--taskset` / `--harness` so we can resolve the right
# config types before tyro builds the help. We do NOT strip the flags — tyro
# still parses them and emits standard "missing required" / "unrecognized
# argument" errors.
# --------------------------------------------------------------------------- #


def _load_toml(path: str) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, ValueError):
        return {}


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


def _build_env(env_module: Any, harness_module: Any, cfg: Any) -> vf.Env:
    taskset = env_module.load_taskset(cfg.taskset_config)
    harness = harness_module.load_harness(cfg.harness_config)
    if not isinstance(taskset, vf.Taskset):
        raise SystemExit(
            f"{env_module.__name__}.load_taskset must return a vf.Taskset."
        )
    if not isinstance(harness, vf.Harness):
        raise SystemExit(
            f"{harness_module.__name__}.load_harness must return a vf.Harness."
        )
    return vf.Env(taskset=taskset, harness=harness)


def _summarize(outputs: GenerateOutputs) -> None:
    rewards = [o["reward"] for o in outputs["outputs"] if o["reward"] is not None]
    if not rewards:
        print("no rewards recorded")
        return
    mean = sum(rewards) / len(rewards)
    print(f"\nrollouts: {len(rewards)}    mean reward: {mean:.4f}")


async def _run(env_module: Any, harness_module: Any, cfg: Any) -> None:
    env = _build_env(env_module, harness_module, cfg)
    client_config = ClientConfig(
        client_type="openai_chat_completions",
        api_base_url="https://api.pinference.ai/api/v1",
        api_key_var="PRIME_API_KEY",
    )
    outputs = await env.evaluate(
        client=client_config,
        model=cfg.model,
        num_examples=cfg.num_examples,
        rollouts_per_example=cfg.rollouts_per_example,
    )
    _summarize(outputs)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Peek the selectors so we know which config types to use when building
    # the dynamic EvalConfig. Tyro still owns parsing/validation.
    taskset_selector = _peek_flag(argv, "taskset")
    harness_selector = _peek_flag(argv, "harness")

    if taskset_selector is not None:
        env_module = _import_env_module(taskset_selector)
        taskset_cls = _discover_taskset_config(env_module)
    else:
        env_module = None
        taskset_cls = vf.TasksetConfig

    if harness_selector is not None:
        harness_module = _resolve_harness_module(harness_selector)
    elif env_module is not None and hasattr(env_module, "load_harness"):
        harness_module = env_module
    else:
        harness_module = importlib.import_module(HARNESS_REGISTRY["base"])
    harness_cls = _discover_harness_config(harness_module)

    EvalConfigCls = _build_eval_config_cls(taskset_cls, harness_cls)
    cfg = cast(Any, cli(EvalConfigCls, args=argv))

    if env_module is None or cfg.taskset != taskset_selector:
        env_module = _import_env_module(cfg.taskset)
    if cfg.harness is not None and cfg.harness != harness_selector:
        harness_module = _resolve_harness_module(cfg.harness)

    asyncio.run(_run(env_module, harness_module, cfg))


if __name__ == "__main__":
    main()
