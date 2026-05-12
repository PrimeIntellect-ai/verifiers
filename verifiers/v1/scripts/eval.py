"""v1 eval entrypoint backed by pydantic-config (tyro + TOML).

Mirrors the v0 `vf-eval` interface against v1 `vf.Env` environments. The
top-level config exposes a `taskset` section and a discriminated `harness`
section so `--help` reflects the harness variant chosen by `--harness.type`.

Examples:
    vf-eval-v1 reverse-text --help
    vf-eval-v1 reverse-text --harness.type opencode --help
    vf-eval-v1 reverse-text -n 1 -r 1
    vf-eval-v1 reverse-text @ configs/eval/my-run.toml
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tomllib
from typing import Annotated, Any, Literal

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import tyro
from pydantic import Field
from pydantic_config import BaseConfig, cli

import verifiers as vf
import verifiers.v1 as vf1
from verifiers import setup_logging
from verifiers.scripts.eval import (
    DEFAULT_CLIENT_TYPE,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_ENV_DIR_PATH,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    PROVIDER_CONFIGS,
    merge_sampling_args,
)
from verifiers.types import ClientConfig, EvalConfig, EvalRunConfig
from verifiers.utils.env_utils import load_environment
from verifiers.utils.eval_utils import (
    get_log_level,
    load_endpoints,
    run_evaluations,
    run_evaluations_tui,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Harness variants. Each carries its own help text via field descriptions.
# --------------------------------------------------------------------------- #


class BaseHarness(BaseConfig):
    """Default endpoint-backed tool loop (`vf.Harness`)."""

    type: Literal["base"] = "base"
    max_turns: int = Field(default=10, description="Per-rollout turn limit.")


class OpenCodeHarness(BaseConfig):
    """Sandboxed OpenCode CLI harness (`vf.OpenCode`)."""

    type: Literal["opencode"] = "opencode"
    disabled_tools: list[str] = Field(
        default_factory=list,
        description="Names of OpenCode tools to disable on every rollout.",
    )
    allow_git: bool | None = Field(
        default=None, description="Permit git operations inside the sandbox."
    )
    disable_compaction: bool | None = Field(
        default=None, description="Disable OpenCode context compaction."
    )
    max_turns: int | None = Field(
        default=None, description="Override OpenCode rollout turn limit."
    )


class RLMHarness(BaseConfig):
    """RLM (Recursive Language Model) CLI harness (`vf.RLM`)."""

    type: Literal["rlm"] = "rlm"
    rlm_max_turns: int = Field(default=100, description="Top-level RLM turn budget.")
    rlm_max_depth: int = Field(default=0, description="Sub-RLM recursion depth.")
    rlm_exec_timeout: int = Field(
        default=300, description="Exec timeout (seconds) per RLM step."
    )
    rlm_tools: list[str] = Field(
        default_factory=lambda: ["bash", "edit"],
        description="Tool surface exposed inside the RLM sandbox.",
    )


class MiniSWEHarness(BaseConfig):
    """mini-swe-agent CLI harness (`vf.MiniSWEAgent`)."""

    type: Literal["mini-swe"] = "mini-swe"


class PiHarness(BaseConfig):
    """Pi Coding Agent CLI harness (`vf.Pi`)."""

    type: Literal["pi"] = "pi"


HARNESS_CONFIGS: dict[str, type[BaseConfig]] = {
    "base": BaseHarness,
    "opencode": OpenCodeHarness,
    "rlm": RLMHarness,
    "mini-swe": MiniSWEHarness,
    "pi": PiHarness,
}


def _build_harness(cfg: BaseConfig) -> vf1.Harness:
    """Materialize a v1 Harness instance from a discriminated harness config."""
    if isinstance(cfg, BaseHarness):
        return vf1.Harness(max_turns=cfg.max_turns)
    if isinstance(cfg, OpenCodeHarness):
        kwargs: dict[str, Any] = {"disabled_tools": cfg.disabled_tools or None}
        if cfg.allow_git is not None:
            kwargs["allow_git"] = cfg.allow_git
        if cfg.disable_compaction is not None:
            kwargs["disable_compaction"] = cfg.disable_compaction
        if cfg.max_turns is not None:
            kwargs["max_turns"] = cfg.max_turns
        return vf1.OpenCode(**kwargs)
    if isinstance(cfg, RLMHarness):
        return vf1.RLM(
            rlm_max_turns=cfg.rlm_max_turns,
            rlm_max_depth=cfg.rlm_max_depth,
            rlm_exec_timeout=cfg.rlm_exec_timeout,
            rlm_tools=cfg.rlm_tools,
        )
    if isinstance(cfg, MiniSWEHarness):
        return vf1.MiniSWEAgent()
    if isinstance(cfg, PiHarness):
        return vf1.Pi()
    raise TypeError(f"unsupported harness config: {type(cfg).__name__}")


# --------------------------------------------------------------------------- #
# Taskset config. Minimal for now — only env_args is exposed; richer fields
# (split, system_prompt, etc.) can be added once we settle a per-taskset
# config story that does not duplicate each env package's TasksetConfig.
# --------------------------------------------------------------------------- #


class TasksetSection(BaseConfig):
    """Taskset-level overrides forwarded to the env package's `load_environment`."""

    env_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Kwargs forwarded to the env package's load_environment.",
    )


# --------------------------------------------------------------------------- #
# Top-level eval config, parametric over the chosen harness variant so that
# `--help` only shows the relevant harness fields.
# --------------------------------------------------------------------------- #


class _EvalConfigBase(BaseConfig):
    """vf-eval-v1: evaluate a v1 environment with a configurable harness."""

    env_id: Annotated[
        str, tyro.conf.Positional, tyro.conf.arg(help="Environment id to evaluate.")
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

    taskset: TasksetSection = Field(default_factory=TasksetSection)


class EvalConfigBase(_EvalConfigBase):
    harness: BaseHarness = Field(default_factory=BaseHarness)


class EvalConfigOpenCode(_EvalConfigBase):
    harness: OpenCodeHarness = Field(default_factory=OpenCodeHarness)


class EvalConfigRLM(_EvalConfigBase):
    harness: RLMHarness = Field(default_factory=RLMHarness)


class EvalConfigMiniSWE(_EvalConfigBase):
    harness: MiniSWEHarness = Field(default_factory=MiniSWEHarness)


class EvalConfigPi(_EvalConfigBase):
    harness: PiHarness = Field(default_factory=PiHarness)


EVAL_CONFIG_CLASSES: dict[str, type[_EvalConfigBase]] = {
    "base": EvalConfigBase,
    "opencode": EvalConfigOpenCode,
    "rlm": EvalConfigRLM,
    "mini-swe": EvalConfigMiniSWE,
    "pi": EvalConfigPi,
}


def _resolve_eval_config_cls(harness_type: str) -> type[_EvalConfigBase]:
    if harness_type not in EVAL_CONFIG_CLASSES:
        raise SystemExit(
            f"unknown --harness.type {harness_type!r}; "
            f"choose from {sorted(EVAL_CONFIG_CLASSES)}"
        )
    return EVAL_CONFIG_CLASSES[harness_type]


# --------------------------------------------------------------------------- #
# Argv pre-scan: resolve `--harness.type` *before* tyro sees the args so we
# can swap the EvalConfig class to the concrete harness variant. This is what
# makes `--harness.type opencode --help` print opencode fields instead of the
# default base-harness fields.
# --------------------------------------------------------------------------- #


def _resolve_harness_type(argv: list[str]) -> str:
    """Resolve harness.type from argv with CLI > @ TOML > default.

    Looks at `--harness.type X`, `--harness.type=X`, `@ file.toml`,
    `--harness @ file.toml`, and inline `[harness] type = X` inside any
    referenced TOML.
    """
    # 1. CLI flag wins.
    for i, a in enumerate(argv):
        if a == "--harness.type" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--harness.type="):
            return a.split("=", 1)[1]

    # 2. TOML config files referenced via @.
    def _load_toml(path: str) -> dict[str, Any]:
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except (OSError, ValueError):
            return {}

    for i, a in enumerate(argv):
        if a == "@" and i + 1 < len(argv):
            data = _load_toml(argv[i + 1])
            harness = data.get("harness")
            if isinstance(harness, dict) and "type" in harness:
                return str(harness["type"])
        if (
            a == "--harness"
            and i + 1 < len(argv)
            and argv[i + 1] == "@"
            and i + 2 < len(argv)
        ):
            data = _load_toml(argv[i + 2])
            if "type" in data:
                return str(data["type"])

    return "base"


# --------------------------------------------------------------------------- #
# Eval execution: translate the parsed config into the existing EvalConfig +
# run_evaluations machinery, swapping the env's harness for the configured
# one.
# --------------------------------------------------------------------------- #


def _build_client_config(cfg: BaseConfig) -> tuple[str, ClientConfig]:
    """Resolve provider/endpoint into (model, ClientConfig)."""
    model: str = cfg.model  # type: ignore[attr-defined]
    api_base_url = cfg.api_base_url  # type: ignore[attr-defined]
    api_key_var = cfg.api_key_var  # type: ignore[attr-defined]
    client_type = cfg.api_client_type  # type: ignore[attr-defined]
    provider = cfg.provider  # type: ignore[attr-defined]

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


async def _run(cfg: BaseConfig) -> None:
    env_id: str = cfg.env_id  # type: ignore[attr-defined]

    # Load the env. We assume v1 envs either return vf.v1.Env directly from
    # load_environment, or accept v1=True (the side-by-side reverse_text /
    # alphabet_sort / math_python pattern).
    try:
        env = load_environment(env_id, v1=True, **cfg.taskset.env_args)  # type: ignore[attr-defined]
    except TypeError:
        env = load_environment(env_id, **cfg.taskset.env_args)  # type: ignore[attr-defined]

    if not isinstance(env, vf1.Env):
        raise SystemExit(
            f"vf-eval-v1 expects a v1 environment (vf.v1.Env); "
            f"'{env_id}' returned {type(env).__name__}."
        )

    # Swap in the configured harness while keeping the taskset.
    harness = _build_harness(cfg.harness)  # type: ignore[attr-defined]
    env.harness = harness
    env.harness.attach_taskset(env.taskset)

    model, client_config = _build_client_config(cfg)
    sampling_args = merge_sampling_args(
        None,
        max_tokens=cfg.max_tokens,  # type: ignore[attr-defined]
        temperature=cfg.temperature,  # type: ignore[attr-defined]
        include_none_max_tokens=True,
    )

    eval_cfg = EvalConfig(
        env_id=env_id,
        env_args=cfg.taskset.env_args,  # type: ignore[attr-defined]
        env_dir_path=DEFAULT_ENV_DIR_PATH,
        output_dir=cfg.output_dir,  # type: ignore[attr-defined]
        extra_env_kwargs={},
        endpoint_id=None,
        model=model,
        client_config=client_config,
        sampling_args=sampling_args,
        num_examples=cfg.num_examples,  # type: ignore[attr-defined]
        rollouts_per_example=cfg.rollouts_per_example,  # type: ignore[attr-defined]
        max_concurrent=cfg.max_concurrent,  # type: ignore[attr-defined]
        max_retries=0,
        num_workers="auto",
        disable_env_server=True,
        verbose=cfg.verbose,  # type: ignore[attr-defined]
        disable_tui=cfg.disable_tui,  # type: ignore[attr-defined]
        state_columns=[],
        save_results=cfg.save_results,  # type: ignore[attr-defined]
        resume_path=None,
        independent_scoring=False,
        save_to_hf_hub=False,
        hf_hub_dataset_name="",
    )

    # `load_environment` registers env_id-keyed lookup; we already hold the env
    # in-memory but the v0 runner re-loads it. To keep the swapped harness, we
    # register the prepared env on the verifiers module load cache.
    _install_env_cache_override(target_env_id=env_id, env=env)

    eval_run_config = EvalRunConfig(evals=[eval_cfg], heartbeat_url=None)
    if cfg.disable_tui:  # type: ignore[attr-defined]
        await run_evaluations(eval_run_config)
    else:
        await run_evaluations_tui(eval_run_config, fullscreen=False, compact=False)


def _install_env_cache_override(target_env_id: str, env: vf1.Env) -> None:
    """Make `vf.load_environment(env_id, ...)` return our prepared env."""
    original = vf.load_environment

    def _override(env_id: str | None = None, **kwargs: Any) -> vf.Environment:
        resolved = env_id if env_id is not None else kwargs.pop("env_id", None)
        if resolved == target_env_id:
            return env
        return original(resolved, **kwargs)  # type: ignore[arg-type]

    vf.load_environment = _override  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    harness_type = _resolve_harness_type(argv)
    EvalConfigCls = _resolve_eval_config_cls(harness_type)

    cfg = cli(EvalConfigCls, args=argv)

    if cfg.disable_tui:  # type: ignore[attr-defined]
        setup_logging(get_log_level(cfg.verbose))  # type: ignore[attr-defined]

    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
