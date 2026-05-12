"""v1 eval entrypoint backed by pydantic-config (tyro + TOML).

Mirrors the v0 `vf-eval` interface against v1 `vf.Env` environments. The
top-level config exposes a `taskset` section (typed by the env package's
`load_taskset` signature) and a `harness` section (typed by `load_harness`).
The entrypoint calls those two functions directly — it does not go through
`vf.load_environment`.

Examples:
    vf-eval-v1 reverse-text --help
    vf-eval-v1 reverse-text --taskset.dataset-split train --num-examples 1
    vf-eval-v1 @ configs/eval/my-run.toml
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import sys
import tomllib
from typing import Annotated, Any, get_args, get_origin

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import tyro
from pydantic import Field, create_model
from pydantic_config import BaseConfig, cli

import verifiers as vf
import verifiers.v1 as vf1
from verifiers import setup_logging
from verifiers.scripts.eval import (
    DEFAULT_CLIENT_TYPE,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    PROVIDER_CONFIGS,
    merge_sampling_args,
)
from verifiers.types import ClientConfig, EvalConfig, EvalRunConfig
from verifiers.utils.eval_utils import (
    get_log_level,
    load_endpoints,
    run_evaluations,
    run_evaluations_tui,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Env package discovery: locate `load_taskset` and `load_harness`, and inspect
# their `config:` annotations to find the concrete TasksetConfig / HarnessConfig
# subclasses the env package declares.
# --------------------------------------------------------------------------- #


def _import_env_module(env_id: str) -> Any:
    """Import the env package, preferring the side-by-side `*_v1` module when
    the base module does not expose `load_taskset`."""
    base = env_id.replace("-", "_").split("/")[-1]
    candidates = [base, f"{base}_v1"]
    last_err: Exception | None = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            last_err = e
            continue
        if hasattr(module, "load_taskset"):
            return module
        last_err = AttributeError(f"{module_name!r} has no `load_taskset`")
    raise SystemExit(
        f"could not load env {env_id!r} (tried modules {candidates}): {last_err}"
    )


def _config_type_from_annotation(annotation: Any, fallback: type) -> type:
    """Resolve `Config | None` or bare `Config` annotations to the Config type."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return fallback
    origin = get_origin(annotation)
    if origin is None:
        return annotation if isinstance(annotation, type) else fallback
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        if isinstance(arg, type):
            return arg
    return fallback


def _resolved_config_annotation(fn: Any, fallback: type) -> type:
    """Read fn's `config` annotation, resolving PEP-563 string annotations."""
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except Exception:
        hints = {}
    annotation = hints.get("config", inspect.Parameter.empty)
    return _config_type_from_annotation(annotation, fallback)


def _single_positional_check(fn: Any, fn_name: str) -> None:
    params = list(inspect.signature(fn).parameters.values())
    if len(params) != 1 or params[0].name != "config":
        raise SystemExit(
            f"{fn.__module__}.{fn_name} must take exactly one positional "
            f"`config` argument (got {[p.name for p in params]})."
        )


def _discover_taskset_config(module: Any) -> type[vf1.TasksetConfig]:
    _single_positional_check(module.load_taskset, "load_taskset")
    return _resolved_config_annotation(module.load_taskset, vf1.TasksetConfig)


def _discover_harness_config(module: Any) -> type[vf1.HarnessConfig]:
    if not hasattr(module, "load_harness"):
        return vf1.HarnessConfig
    _single_positional_check(module.load_harness, "load_harness")
    return _resolved_config_annotation(module.load_harness, vf1.HarnessConfig)


# --------------------------------------------------------------------------- #
# Top-level EvalConfig built dynamically per env so `--taskset.*` and
# `--harness.*` flags reflect the env's concrete config types.
# --------------------------------------------------------------------------- #


class _EvalConfigBase(BaseConfig):
    """vf-eval-v1: evaluate a v1 environment via load_taskset + load_harness."""

    task: Annotated[
        str,
        tyro.conf.Positional,
        tyro.conf.arg(help="Environment id (positional, required)."),
    ]

    # Model / endpoint controls — mirror v0 vf-eval shape.
    model: str = Field(default=DEFAULT_MODEL, description="Model id.")
    provider: str | None = Field(
        default=None,
        description="Inference provider shorthand; resolves base url + key var.",
    )
    api_base_url: str | None = Field(default=None, description="API base URL.")
    api_key_var: str | None = Field(
        default=None, description="Env var holding the API key."
    )
    api_client_type: str | None = Field(
        default=None,
        description="Client type ('openai_chat_completions', 'anthropic_messages', ...).",
    )

    # Rollout controls.
    num_examples: int = Field(default=5, description="Examples to evaluate.")
    rollouts_per_example: int = Field(default=3, description="Rollouts per example.")
    max_concurrent: int = Field(default=32, description="Max concurrent rollouts.")
    max_tokens: int | None = Field(default=None, description="Max output tokens.")
    temperature: float | None = Field(default=None, description="Sampling temp.")

    # Output controls.
    save_results: bool = Field(default=False, description="Save results to disk.")
    output_dir: str | None = Field(
        default=None, description="Override results output directory."
    )
    disable_tui: bool = Field(
        default=False, description="Disable the Rich TUI display."
    )
    verbose: bool = Field(default=False, description="Verbose logging.")


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
# Pre-scan argv for the `task` positional so we can import the env module and
# discover its config types before tyro takes over.
# --------------------------------------------------------------------------- #


_BOOLEAN_FLAGS = {
    "--save-results",
    "--no-save-results",
    "--disable-tui",
    "--no-disable-tui",
    "--verbose",
    "--no-verbose",
    "-h",
    "--help",
}


def _load_toml(path: str) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, ValueError):
        return {}


def _peek_task_from_toml(argv: list[str]) -> str | None:
    for i, a in enumerate(argv):
        if a == "@" and i + 1 < len(argv):
            data = _load_toml(argv[i + 1])
            if "task" in data:
                return str(data["task"])
    return None


def _peek_task_from_argv(argv: list[str]) -> str | None:
    """Best-effort scan for the first positional argument."""
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "@":
            i += 2  # skip @ <path>
            continue
        if "=" in a and a.startswith("--"):
            i += 1
            continue
        if a in _BOOLEAN_FLAGS:
            i += 1
            continue
        if a.startswith("-"):
            # Value-taking flag: skip flag + value.
            i += 2
            continue
        return a
    return None


def _resolve_task(argv: list[str]) -> str:
    task = _peek_task_from_argv(argv) or _peek_task_from_toml(argv)
    if task is None:
        raise SystemExit(
            "usage: vf-eval-v1 <task> [options]\n"
            "  positional 'task' is required (or set `task = \"...\"` in a @ TOML)."
        )
    return task


# --------------------------------------------------------------------------- #
# Eval execution: build the env directly from load_taskset + load_harness.
# --------------------------------------------------------------------------- #


def _build_client_config(cfg: Any) -> tuple[str, ClientConfig]:
    model: str = cfg.model
    api_base_url = cfg.api_base_url
    api_key_var = cfg.api_key_var
    client_type = cfg.api_client_type
    provider = cfg.provider

    direct = api_base_url is not None and api_key_var is not None
    endpoints = {} if direct else load_endpoints(DEFAULT_ENDPOINTS_PATH)

    if model in endpoints:
        entry = endpoints[model][0]
        api_key_var = api_key_var or entry["key"]
        api_base_url = api_base_url or entry["url"]
        client_type = client_type or entry.get("api_client_type", DEFAULT_CLIENT_TYPE)
        model = entry["model"]
    else:
        pcfg = PROVIDER_CONFIGS[provider or DEFAULT_PROVIDER]
        api_key_var = api_key_var or pcfg["key"]
        api_base_url = api_base_url or pcfg["url"]
        client_type = client_type or pcfg.get("client_type", DEFAULT_CLIENT_TYPE)

    return model, ClientConfig(
        client_type=client_type,  # type: ignore[arg-type]
        api_key_var=api_key_var,
        api_base_url=api_base_url,
    )


def _build_env(module: Any, cfg: Any) -> vf1.Env:
    taskset = module.load_taskset(cfg.taskset)
    if hasattr(module, "load_harness"):
        harness = module.load_harness(cfg.harness)
    else:
        harness = vf1.Harness(config=cfg.harness)
    if not isinstance(taskset, vf1.Taskset):
        raise SystemExit(
            f"{module.__name__}.load_taskset must return a vf.Taskset "
            f"(got {type(taskset).__name__})."
        )
    if not isinstance(harness, vf1.Harness):
        raise SystemExit(
            f"{module.__name__}.load_harness must return a vf.Harness "
            f"(got {type(harness).__name__})."
        )
    return vf1.Env(taskset=taskset, harness=harness)


def _install_env_cache_override(target_env_id: str, env: vf1.Env) -> None:
    """Make `vf.load_environment(env_id, ...)` return our prepared env.

    The eval runner re-loads the env by id inside `run_evaluation`; intercept
    that single call instead of plumbing the env through `EvalConfig`.
    """
    original = vf.load_environment

    def _override(env_id: str | None = None, **kwargs: Any) -> vf.Environment:
        resolved = env_id if env_id is not None else kwargs.pop("env_id", None)
        if resolved == target_env_id:
            return env
        return original(resolved, **kwargs)  # type: ignore[arg-type]

    vf.load_environment = _override  # type: ignore[assignment]


async def _run(env_id: str, module: Any, cfg: Any) -> None:
    env = _build_env(module, cfg)

    model, client_config = _build_client_config(cfg)
    sampling_args = merge_sampling_args(
        None,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        include_none_max_tokens=True,
    )

    eval_cfg = EvalConfig(
        env_id=env_id,
        env_args={},
        env_dir_path=".",
        output_dir=cfg.output_dir,
        extra_env_kwargs={},
        endpoint_id=None,
        model=model,
        client_config=client_config,
        sampling_args=sampling_args,
        num_examples=cfg.num_examples,
        rollouts_per_example=cfg.rollouts_per_example,
        max_concurrent=cfg.max_concurrent,
        max_retries=0,
        num_workers="auto",
        disable_env_server=True,
        verbose=cfg.verbose,
        disable_tui=cfg.disable_tui,
        state_columns=[],
        save_results=cfg.save_results,
        resume_path=None,
        independent_scoring=False,
        save_to_hf_hub=False,
        hf_hub_dataset_name="",
    )

    _install_env_cache_override(target_env_id=env_id, env=env)

    eval_run_config = EvalRunConfig(evals=[eval_cfg], heartbeat_url=None)
    if cfg.disable_tui:
        await run_evaluations(eval_run_config)
    else:
        await run_evaluations_tui(eval_run_config, fullscreen=False, compact=False)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # If the user only asked for --help with no env, we still need *some*
    # config class. Use vf.TasksetConfig / vf.HarnessConfig as fallbacks.
    task = _peek_task_from_argv(argv) or _peek_task_from_toml(argv)
    if task is None:
        module = None
        taskset_cls: type[vf1.TasksetConfig] = vf1.TasksetConfig
        harness_cls: type[vf1.HarnessConfig] = vf1.HarnessConfig
    else:
        module = _import_env_module(task)
        taskset_cls = _discover_taskset_config(module)
        harness_cls = _discover_harness_config(module)

    EvalConfigCls = _build_eval_config_cls(taskset_cls, harness_cls)
    cfg = cli(EvalConfigCls, args=argv)

    if cfg.disable_tui:
        setup_logging(get_log_level(cfg.verbose))

    resolved_task = cfg.task
    if module is None or resolved_task != task:
        module = _import_env_module(resolved_task)

    asyncio.run(_run(resolved_task, module, cfg))


if __name__ == "__main__":
    main()
