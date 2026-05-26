"""``vf-eval-v1`` — evaluation CLI for v1 taskset/harness environments.

This CLI is built around the ``verifiers.v1`` taskset/harness model. Compared
to the legacy ``vf-eval`` it:

* assumes the env module exposes ``load_taskset(config: TasksetConfig)``;
* runs in the env's *default harness* when nothing is configured (matching
  the legacy behavior);
* allows overriding any field on the default harness's config via
  ``--harness.<field>`` (e.g. ``--harness.max-turns 5``);
* allows swapping the harness class entirely via ``--harness.name`` (e.g.
  ``--harness.name rlm`` or any ``pkg.mod:Class`` import ref);
* keeps a v0 fallback: when the module only exposes ``load_environment`` the
  CLI calls it with ``--env-args`` and never tries to touch the bundled
  harness.

The full config surface is a :class:`EvalV1Config` Pydantic model. Anything
that can be set on the CLI can equally be loaded from TOML via the
``@ path/to/config.toml`` mechanism provided by ``pydantic-config``.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

from pydantic import AliasChoices, ConfigDict, Field, field_validator
from pydantic_config import BaseConfig, cli

from verifiers import setup_logging
from verifiers.types import (
    ClientConfig,
    ClientType,
    EvalConfig,
    EvalRunConfig,
)
from verifiers.utils.env_utils import import_env_module
from verifiers.utils.eval_utils import (
    get_log_level,
    run_evaluations,
    run_evaluations_tui,
)
from verifiers.utils.install_utils import check_hub_env_installed
from verifiers.utils.v1_loader_utils import (
    HARNESS_ALIASES,
    V1_HARNESS_KEY,
    V1_TASKSET_KEY,
    module_supports_v1_loader,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider shorthand (mirrors the legacy ``vf-eval`` --provider table).
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict[str, str]] = {
    "prime": {"url": "https://api.pinference.ai/api/v1", "key": "PRIME_API_KEY"},
    "openai": {"url": "https://api.openai.com/v1", "key": "OPENAI_API_KEY"},
    "anthropic": {
        "url": "https://api.anthropic.com",
        "key": "ANTHROPIC_API_KEY",
        "client_type": "anthropic_messages",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1",
        "key": "OPENROUTER_API_KEY",
    },
    "deepseek": {"url": "https://api.deepseek.com/v1", "key": "DEEPSEEK_API_KEY"},
    "local": {"url": "http://localhost:8000/v1", "key": "VLLM_API_KEY"},
}
DEFAULT_PROVIDER = "prime"


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------


class TasksetSpec(BaseConfig):
    """Taskset config overrides.

    Sub-fields are validated against the env's actual ``TasksetConfig``
    subclass at runtime, so e.g. ``--taskset.dataset-split test`` works
    whenever ``dataset_split`` exists on the env's taskset config.
    """

    model_config = ConfigDict(extra="allow")


class HarnessSpec(BaseConfig):
    """Harness selection and config overrides.

    By default (``name=None``) the env's own ``load_harness`` is used if
    present, otherwise the base ``verifiers.v1.Harness`` with
    ``HarnessConfig()`` defaults. Any extra fields are merged into the
    harness's actual config at runtime.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = Field(
        None,
        description=(
            "Harness identifier — alias from the built-in registry "
            "(e.g. ``rlm``, ``opencode``, ``pi``, ``terminus-2``, "
            "``mini-swe-agent``, ``base``) or a ``pkg.mod:Class`` import "
            "ref. When unset, falls back to the env's ``load_harness`` if "
            "present, otherwise the base ``verifiers.v1.Harness``."
        ),
    )


class EvalV1Config(BaseConfig):
    """``vf-eval-v1`` configuration."""

    env: str = Field(
        description=(
            "Environment id (the installed module name; ``-`` and ``_`` are "
            "interchangeable)."
        ),
    )
    name: str | None = Field(
        None, description="Optional human-readable run name (saved in metadata)."
    )

    taskset: TasksetSpec = TasksetSpec()
    """Overrides for the env's taskset config."""

    harness: HarnessSpec = HarnessSpec()
    """Selection + overrides for the env's harness."""

    env_args: dict = Field(
        default_factory=dict,
        validation_alias=AliasChoices("env_args", "a"),
        description=(
            "Extra kwargs passed to ``load_environment`` for v0 envs (the "
            "harness bundled by a v0 env is not swappable; use ``--harness`` "
            "instead for v1 tasksets)."
        ),
    )
    env_dir_path: str = Field(
        "./environments",
        description="Directory used by ``prime env install`` for local envs.",
    )

    # ---- model / inference --------------------------------------------------

    model: str = Field(
        "openai/gpt-4.1-mini",
        validation_alias=AliasChoices("model", "m"),
        description="Model id to send to the inference API.",
    )
    provider: str | None = Field(
        None,
        validation_alias=AliasChoices("provider", "p"),
        description=(
            "Inference provider shorthand. One of "
            f"{sorted(PROVIDERS)}. Resolves ``base-url`` / ``api-key-var`` / "
            "``client-type`` defaults; explicit flags still win."
        ),
    )
    base_url: str | None = Field(
        None,
        validation_alias=AliasChoices("base_url", "api_base_url", "b"),
        description="API base URL (overrides ``--provider``).",
    )
    api_key_var: str | None = Field(
        None,
        validation_alias=AliasChoices("api_key_var", "k"),
        description="Env var holding the API key (overrides ``--provider``).",
    )
    client_type: ClientType | None = Field(
        None,
        description="Inference client type. Default: ``openai_chat_completions``.",
    )
    max_tokens: int | None = Field(
        None,
        validation_alias=AliasChoices("max_tokens", "t"),
        description="Cap on completion tokens per turn (added to sampling args).",
    )
    temperature: float | None = Field(
        None,
        validation_alias=AliasChoices("temperature", "T"),
        description="Sampling temperature (added to sampling args).",
    )
    sampling_args: dict = Field(
        default_factory=dict,
        validation_alias=AliasChoices("sampling_args", "S"),
        description="Sampling args (deep-merged with ``--max-tokens`` / ``--temperature``).",
    )
    headers: dict = Field(
        default_factory=dict,
        description="Extra HTTP headers to attach to every inference request.",
    )

    # ---- run shape ----------------------------------------------------------

    num_examples: int = Field(
        5,
        validation_alias=AliasChoices("num_examples", "n"),
        description="Number of examples to evaluate.",
    )
    rollouts_per_example: int = Field(
        3,
        validation_alias=AliasChoices("rollouts_per_example", "r"),
        description="Rollouts per example (groups).",
    )
    max_concurrent: int = Field(
        32,
        validation_alias=AliasChoices("max_concurrent", "c"),
        description="Maximum concurrent rollouts.",
    )
    max_retries: int = Field(
        0,
        description="Max retries for transient infrastructure errors.",
    )
    timeout: float | None = Field(
        None, description="Per-rollout wall-clock timeout (seconds)."
    )
    num_workers: str | int = Field(
        "auto",
        validation_alias=AliasChoices("num_workers", "w"),
        description='Env server workers ("auto" or an integer).',
    )
    disable_env_server: bool = Field(
        False, description="Run rollouts in-process instead of starting env workers."
    )

    # ---- scoring ------------------------------------------------------------

    independent_scoring: bool = Field(
        False,
        validation_alias=AliasChoices("independent_scoring", "i"),
        description="Score each rollout individually instead of by group.",
    )

    # ---- output -------------------------------------------------------------

    output_dir: str | None = Field(
        None,
        validation_alias=AliasChoices("output_dir", "o"),
        description="Custom output directory for evaluation results and logs.",
    )
    save_results: bool = Field(
        False,
        validation_alias=AliasChoices("save_results", "s"),
        description="Save full rollouts to disk.",
    )
    state_columns: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("state_columns", "C"),
        description="State columns to include in the saved rollouts.",
    )
    resume: str | None = Field(
        None,
        validation_alias=AliasChoices("resume", "R"),
        description="Resume from an explicit results path.",
    )
    save_to_hf_hub: bool = Field(
        False,
        validation_alias=AliasChoices("save_to_hf_hub", "H"),
        description="Push the result dataset to the Hugging Face Hub.",
    )
    hf_hub_dataset_name: str = Field(
        "",
        validation_alias=AliasChoices("hf_hub_dataset_name", "D"),
        description="Hugging Face Hub dataset name (with --save-to-hf-hub).",
    )

    # ---- UI -----------------------------------------------------------------

    verbose: bool = Field(
        False,
        validation_alias=AliasChoices("verbose", "v"),
        description="Verbose logging.",
    )
    fullscreen: bool = Field(
        False,
        validation_alias=AliasChoices("fullscreen", "f"),
        description="Use Rich's alternate-screen TUI.",
    )
    disable_tui: bool = Field(
        False,
        validation_alias=AliasChoices("disable_tui", "d"),
        description="Disable the Rich TUI and fall back to plain logging.",
    )
    abbreviated_summary: bool = Field(
        False,
        validation_alias=AliasChoices("abbreviated_summary", "A"),
        description="Show the compact end-of-run summary (no per-example block).",
    )
    heartbeat_url: str | None = Field(
        None, description="Heartbeat URL pinged after each progress update."
    )

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str | None) -> str | None:
        if value is not None and value not in PROVIDERS:
            raise ValueError(
                f"Unknown provider {value!r}; pick one of {sorted(PROVIDERS)}."
            )
        return value


# ---------------------------------------------------------------------------
# Spec -> dict helpers (collapse explicit fields + ``extra="allow"`` extras
# into a plain mapping that survives JSON serialization, so we can hand it to
# worker subprocesses through ``env_args``).
# ---------------------------------------------------------------------------


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return value


def _spec_to_dict(
    spec: BaseConfig, *, drop_none: tuple[str, ...] = ()
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for name in spec.model_fields_set:
        value = getattr(spec, name)
        if name in drop_none and value is None:
            continue
        data[name] = value
    for name, value in (spec.model_extra or {}).items():
        data[name] = _maybe_parse_json(value)
    return data


# ---------------------------------------------------------------------------
# Eval pipeline glue
# ---------------------------------------------------------------------------


def _resolve_client_config(config: EvalV1Config) -> ClientConfig:
    provider = config.provider or DEFAULT_PROVIDER
    provider_cfg = PROVIDERS[provider]
    api_base_url = config.base_url or provider_cfg["url"]
    api_key_var = config.api_key_var or provider_cfg["key"]
    client_type = config.client_type or provider_cfg.get(
        "client_type", "openai_chat_completions"
    )
    headers = {str(k): str(v) for k, v in config.headers.items()}
    return ClientConfig(
        client_type=cast(ClientType, client_type),
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        extra_headers=headers,
        extra_headers_from_state={"X-Session-ID": "example_id"},
    )


def _resolve_sampling_args(config: EvalV1Config) -> dict[str, Any]:
    sampling_args: dict[str, Any] = dict(config.sampling_args or {})
    if config.max_tokens is not None and "max_tokens" not in sampling_args:
        sampling_args["max_tokens"] = config.max_tokens
    if config.temperature is not None and "temperature" not in sampling_args:
        sampling_args["temperature"] = config.temperature
    return sampling_args


def _resolve_env_args(config: EvalV1Config) -> dict[str, Any]:
    """Build the env_args dict handed to ``vf.load_environment``.

    For v1 envs, taskset/harness overrides are dispatched through the
    reserved ``__vf_v1_taskset__`` / ``__vf_v1_harness__`` keys so that worker
    subprocesses re-materialize the same env. For v0 envs, the user's
    ``--env-args`` is forwarded verbatim.
    """
    env_module = import_env_module(config.env)
    taskset_overrides = _spec_to_dict(config.taskset)
    harness_overrides = _spec_to_dict(config.harness, drop_none=("name",))

    if module_supports_v1_loader(env_module):
        if config.env_args:
            raise ValueError(
                f"Env {config.env!r} is a v1 env (load_taskset detected); use "
                "--taskset.* / --harness.* rather than --env-args."
            )
        env_args: dict[str, Any] = {}
        if taskset_overrides:
            env_args[V1_TASKSET_KEY] = taskset_overrides
        if harness_overrides:
            env_args[V1_HARNESS_KEY] = harness_overrides
        return env_args

    if taskset_overrides:
        raise ValueError(
            f"Env {config.env!r} only exposes load_environment; --taskset.* "
            "overrides are not supported for v0 envs."
        )
    if harness_overrides:
        raise ValueError(
            f"Env {config.env!r} only exposes load_environment; --harness.* "
            "overrides are not supported for v0 envs."
        )
    return dict(config.env_args)


def _build_eval_config(config: EvalV1Config) -> EvalConfig:
    extra_env_kwargs: dict[str, Any] = {}
    if config.timeout is not None:
        extra_env_kwargs["timeout_seconds"] = config.timeout

    return EvalConfig(
        env_id=config.env,
        name=config.name,
        env_args=_resolve_env_args(config),
        env_dir_path=config.env_dir_path,
        model=config.model,
        client_config=_resolve_client_config(config),
        sampling_args=_resolve_sampling_args(config),
        num_examples=config.num_examples,
        rollouts_per_example=config.rollouts_per_example,
        max_concurrent=config.max_concurrent,
        num_workers=config.num_workers,
        independent_scoring=config.independent_scoring,
        extra_env_kwargs=extra_env_kwargs,
        max_retries=config.max_retries,
        disable_env_server=config.disable_env_server,
        verbose=config.verbose,
        disable_tui=config.disable_tui,
        output_dir=config.output_dir,
        state_columns=config.state_columns,
        save_results=config.save_results,
        resume_path=Path(config.resume) if config.resume else None,
        save_to_hf_hub=config.save_to_hf_hub,
        hf_hub_dataset_name=config.hf_hub_dataset_name,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _preprocess_argv(argv: list[str]) -> list[str]:
    """Promote a bare leading positional to ``--env <value>`` for ergonomics."""
    if len(argv) <= 1:
        return argv
    first = argv[1]
    if first.startswith("-") or first.startswith("@") or first == "@":
        return argv
    return [argv[0], "--env", first, *argv[2:]]


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv) if argv is None else [sys.argv[0], *argv]
    pruned_argv = _preprocess_argv(raw_argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    saved_argv = sys.argv
    sys.argv = pruned_argv
    try:
        config = cli(EvalV1Config)
    finally:
        sys.argv = saved_argv

    if config.disable_tui and config.fullscreen:
        raise SystemExit(
            "error: --disable-tui and --fullscreen are mutually exclusive."
        )

    if config.disable_tui:
        setup_logging(get_log_level(config.verbose))

    if not check_hub_env_installed(config.env):
        raise SystemExit(
            f"Environment {config.env!r} is not installed.\n"
            f"  prime env install {config.env}"
        )

    eval_config = _build_eval_config(config)
    eval_run_config = EvalRunConfig(
        evals=[eval_config], heartbeat_url=config.heartbeat_url
    )

    if config.disable_tui:
        asyncio.run(run_evaluations(eval_run_config))
    else:
        asyncio.run(
            run_evaluations_tui(
                eval_run_config,
                fullscreen=config.fullscreen,
                compact=config.abbreviated_summary,
            )
        )


__all__ = [
    "EvalV1Config",
    "HarnessSpec",
    "TasksetSpec",
    "HARNESS_ALIASES",
    "main",
]


if __name__ == "__main__":
    main()
