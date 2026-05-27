"""``vf-eval-v1`` — evaluation CLI for v1 taskset/harness environments.

The CLI is shaped around two positionals and dotted overrides:

::

    vf-eval-v1 <task> [<harness>] [--taskset.<field> ...] [--harness.<field> ...]

- ``<task>`` is an installed env module name. The module must expose
  ``load_taskset(config: TasksetConfig)``; nothing else is required.
- ``<harness>`` is optional. If omitted the harness auto-resolves to the
  env's own ``load_harness`` if present, otherwise to the base
  ``verifiers.v1.Harness``. If provided it can be a registry alias
  (``rlm``, ``opencode``, ``pi``, ``terminus-2``, ``mini-swe-agent``,
  ``base``) or a ``pkg.mod:Class`` import ref.
- ``--taskset.<field>`` flows into the env's actual ``TasksetConfig``
  subclass, ``--harness.<field>`` into the resolved ``HarnessConfig``
  subclass. Both are typed: invalid fields fail at CLI-parse time, with
  the actual config schema rendered in ``--help``.

Anything settable on the CLI can equally be loaded from TOML via
``vf-eval-v1 @ path/to/eval.toml``.

The v0 fallback (modules that only expose ``load_environment``) keeps
working: the CLI calls the bundled loader with ``--env-args`` and rejects
``--taskset.*`` / ``--harness.*`` overrides for those envs.
"""

import asyncio
import json
import logging
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

from pydantic import AliasChoices, Field, create_model, field_validator
from pydantic_config import BaseConfig, cli

from verifiers import setup_logging
from verifiers.types import (
    ClientConfig,
    ClientType,
    EvalConfig,
    EvalRunConfig,
)
from verifiers.utils.env_utils import (
    factory_config_type,
    import_env_module,
)
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
    harness_config_type_from_class,
    module_supports_v1_loader,
    resolve_harness_class,
)
from verifiers.v1.config import HarnessConfig, TasksetConfig

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
# Static config base (shared between v0 and v1 dispatch).
# ---------------------------------------------------------------------------


class EvalConfigBase(BaseConfig):
    """Run/model/UI knobs shared by every ``vf-eval`` invocation.

    The taskset and harness fields are added per-invocation by
    :func:`_build_v1_eval_config`, so their types match the actual env.
    """

    env: str = Field(
        description=(
            "Environment id (the installed module name; ``-`` and ``_`` are "
            "interchangeable)."
        ),
    )
    harness_name: str | None = Field(
        None,
        description=(
            "Harness identifier — registry alias (``rlm``, ``opencode``, "
            "``pi``, ``terminus-2``, ``mini-swe-agent``, ``base``) or a "
            "``pkg.mod:Class`` import ref. Settable as the second positional. "
            "When unset, auto-resolves to the env's ``load_harness`` if "
            "present, otherwise the base ``verifiers.v1.Harness``."
        ),
    )
    name: str | None = Field(
        None, description="Optional human-readable run name (saved in metadata)."
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

    # ---- scoring + saving + UI ---------------------------------------------

    independent_scoring: bool = Field(
        False,
        validation_alias=AliasChoices("independent_scoring", "i"),
        description="Score each rollout individually instead of by group.",
    )
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


class EvalV0Config(EvalConfigBase):
    """Eval config for v0 envs (modules that expose only ``load_environment``)."""

    env_args: dict = Field(
        default_factory=dict,
        validation_alias=AliasChoices("env_args", "a"),
        description=(
            "Extra kwargs passed to ``load_environment`` for v0 envs (the "
            "harness bundled by a v0 env is not swappable)."
        ),
    )

    @field_validator("harness_name")
    @classmethod
    def _v0_rejects_harness_name(cls, value: str | None) -> str | None:
        if value is not None:
            raise ValueError(
                "Env is a v0 module (no load_taskset); harness selection is "
                "not supported. The bundled harness in load_environment is "
                "not swappable."
            )
        return value


# ---------------------------------------------------------------------------
# Dynamic v1 EvalConfig: typed taskset/harness fields per env.
# ---------------------------------------------------------------------------


def _build_v1_eval_config(
    taskset_cls: type[TasksetConfig],
    harness_cls: type[HarnessConfig],
) -> type[BaseConfig]:
    """Subclass ``EvalConfigBase`` with typed ``taskset`` and ``harness`` fields.

    The dynamic class is what ``pydantic_config.cli()`` validates against, so
    ``--taskset.<field>`` and ``--harness.<field>`` are checked against the
    env's actual ``TasksetConfig`` / ``HarnessConfig`` subclasses at parse
    time.
    """
    return create_model(
        "ResolvedEvalConfig",
        __base__=EvalConfigBase,
        taskset=(taskset_cls, Field(default_factory=taskset_cls)),
        harness=(harness_cls, Field(default_factory=harness_cls)),
    )


def _resolve_taskset_config_class(env_module: Any) -> type[TasksetConfig]:
    config_type = factory_config_type(env_module, "load_taskset", TasksetConfig)
    if config_type is None:
        raise TypeError(
            f"{env_module.__name__}.load_taskset must accept a typed config."
        )
    return cast(type[TasksetConfig], config_type)


def _resolve_harness_config_class(
    env_module: Any, harness_name: str | None
) -> type[HarnessConfig]:
    if harness_name is not None:
        harness_cls = resolve_harness_class(harness_name)
        return harness_config_type_from_class(harness_cls)
    if hasattr(env_module, "load_harness"):
        factory_type = factory_config_type(env_module, "load_harness", HarnessConfig)
        return cast(type[HarnessConfig], factory_type or HarnessConfig)
    return HarnessConfig


# ---------------------------------------------------------------------------
# Argv preprocessing: positional <task> [<harness>] + @ file.toml peek.
# ---------------------------------------------------------------------------


def _peek_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, ValueError):
        return {}


def _extract_initial_args(
    argv: list[str],
) -> tuple[list[str], str | None, str | None]:
    """Pull leading positionals + ``--env`` / ``--harness-name`` out of ``argv``.

    Returns ``(cleaned_argv, env_id, harness_name)``.

    Positionals are recognised at the front of the argv only: scanning stops
    at the first ``-``/``--``/``@`` token. The bare positionals are dropped
    from the cleaned argv and re-injected as ``--env`` / ``--harness-name``
    flags so the downstream pydantic-config parse sees them as ordinary
    fields.

    Also reads ``--env`` / ``--harness-name`` explicit flags from anywhere
    in argv, and peeks at any ``@ path/to.toml`` argument to backfill
    env/harness_name when the user hasn't already set them on the command
    line. Explicit flags / positionals win over TOML values.

    Raises ``SystemExit`` if more than two bare positionals appear.
    """
    env_id: str | None = None
    harness_name: str | None = None

    # Pass 1: leading positionals (stop at first non-positional).
    consumed_positionals = 0
    cursor = 1
    while cursor < len(argv):
        token = argv[cursor]
        if token.startswith("-") or token.startswith("@"):
            break
        if consumed_positionals == 0:
            env_id = token
        elif consumed_positionals == 1:
            harness_name = token
        else:
            raise SystemExit(
                "vf-eval-v1 takes at most two positionals (task + optional "
                f"harness); got {token!r}."
            )
        consumed_positionals += 1
        cursor += 1

    rest = argv[cursor:]
    cleaned = [argv[0], *rest]

    # Pass 2: explicit --env / --harness-name flags anywhere in the rest.
    j = 0
    while j < len(rest):
        token = rest[j]
        for flag, setter in (
            ("--env", "env"),
            ("--harness-name", "harness"),
        ):
            if token == flag and j + 1 < len(rest):
                if setter == "env":
                    env_id = rest[j + 1]
                else:
                    harness_name = rest[j + 1]
                break
            if token.startswith(flag + "="):
                value = token.split("=", 1)[1]
                if setter == "env":
                    env_id = value
                else:
                    harness_name = value
                break
        j += 1

    # Pass 3: TOML peek for any @ file references that don't already cover
    # env / harness_name.
    j = 0
    while j < len(rest):
        token = rest[j]
        toml_path: Path | None = None
        if token == "@" and j + 1 < len(rest):
            toml_path = Path(rest[j + 1])
            j += 2
        elif token.startswith("@") and token != "@":
            toml_path = Path(token[1:])
            j += 1
        else:
            j += 1
            continue
        if toml_path is None:
            continue
        data = _peek_toml(toml_path)
        if env_id is None and isinstance(data.get("env"), str):
            env_id = data["env"]
        if harness_name is None and isinstance(data.get("harness_name"), str):
            harness_name = data["harness_name"]

    # Re-inject the positionals as explicit flags if they aren't already on
    # the command line. The explicit flag form wins on conflict because
    # we already updated env_id / harness_name from it in pass 2.
    has_env_flag = any(a == "--env" or a.startswith("--env=") for a in cleaned[1:])
    has_harness_flag = any(
        a == "--harness-name" or a.startswith("--harness-name=") for a in cleaned[1:]
    )
    if env_id is not None and not has_env_flag:
        cleaned.extend(["--env", env_id])
    if harness_name is not None and not has_harness_flag:
        cleaned.extend(["--harness-name", harness_name])

    return cleaned, env_id, harness_name


# ---------------------------------------------------------------------------
# Helpers shared between v0 and v1 dispatch.
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


def _resolve_client_config(config: EvalConfigBase) -> ClientConfig:
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


def _resolve_sampling_args(config: EvalConfigBase) -> dict[str, Any]:
    sampling_args: dict[str, Any] = dict(config.sampling_args or {})
    if config.max_tokens is not None and "max_tokens" not in sampling_args:
        sampling_args["max_tokens"] = config.max_tokens
    if config.temperature is not None and "temperature" not in sampling_args:
        sampling_args["temperature"] = config.temperature
    return sampling_args


def _v1_env_args(config: BaseConfig) -> dict[str, Any]:
    """Build the env_args dict for a v1 dispatch.

    Carries the typed taskset/harness configs through the reserved
    ``__vf_v1_taskset__`` / ``__vf_v1_harness__`` keys so worker
    subprocesses can rebuild the exact same env in every process.
    """
    env_args: dict[str, Any] = {}
    taskset_data = cast(BaseConfig, config.taskset).model_dump(  # type: ignore[attr-defined]
        exclude_unset=True, exclude_defaults=True
    )
    if taskset_data:
        env_args[V1_TASKSET_KEY] = taskset_data
    harness_data = cast(BaseConfig, config.harness).model_dump(  # type: ignore[attr-defined]
        exclude_unset=True, exclude_defaults=True
    )
    harness_name = cast(str | None, config.harness_name)  # type: ignore[attr-defined]
    if harness_name is not None:
        harness_data["name"] = harness_name
    if harness_data:
        env_args[V1_HARNESS_KEY] = harness_data
    return env_args


def _build_eval_config(config: EvalConfigBase, env_args: dict[str, Any]) -> EvalConfig:
    extra_env_kwargs: dict[str, Any] = {}
    if config.timeout is not None:
        extra_env_kwargs["timeout_seconds"] = config.timeout

    return EvalConfig(
        env_id=config.env,
        name=config.name,
        env_args=env_args,
        env_dir_path="./environments",
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


def _resolve_config_class(
    env_id: str | None, harness_name: str | None
) -> type[BaseConfig]:
    """Pick the right BaseConfig subclass for this invocation."""
    if env_id is None:
        # No env yet — fall back to the base. cli() will report --env as
        # required (or render the top-level --help).
        return EvalConfigBase
    env_module = import_env_module(env_id)
    if module_supports_v1_loader(env_module):
        taskset_cls = _resolve_taskset_config_class(env_module)
        harness_cls = _resolve_harness_config_class(env_module, harness_name)
        return _build_v1_eval_config(taskset_cls, harness_cls)
    return EvalV0Config


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv) if argv is None else [sys.argv[0], *argv]
    cleaned_argv, env_id, harness_name = _extract_initial_args(raw_argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    if env_id is not None and not check_hub_env_installed(env_id):
        raise SystemExit(
            f"Environment {env_id!r} is not installed.\n  prime env install {env_id}"
        )

    config_cls = _resolve_config_class(env_id, harness_name)

    saved_argv = sys.argv
    sys.argv = cleaned_argv
    try:
        config = cli(config_cls)
    finally:
        sys.argv = saved_argv

    if config.disable_tui and config.fullscreen:
        raise SystemExit(
            "error: --disable-tui and --fullscreen are mutually exclusive."
        )
    if config.disable_tui:
        setup_logging(get_log_level(config.verbose))

    # Build env_args based on the dispatch path.
    if isinstance(config, EvalV0Config):
        env_args = dict(config.env_args)
    elif isinstance(config, EvalConfigBase):
        env_args = _v1_env_args(config)
    else:  # pragma: no cover — exhaustive
        raise SystemExit(
            f"Unexpected config type {type(config).__name__}; please report this."
        )

    eval_config = _build_eval_config(config, env_args)
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
    "EvalConfigBase",
    "EvalV0Config",
    "HARNESS_ALIASES",
    "main",
]


if __name__ == "__main__":
    main()
