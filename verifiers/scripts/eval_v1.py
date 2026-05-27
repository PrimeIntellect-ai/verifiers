"""``vf-eval-v1`` — evaluation CLI for v1 taskset/harness environments.

The CLI is shaped around two positionals and dotted overrides:

::

    vf-eval-v1 <taskset> [<harness>] [--taskset.<field> ...] [--harness.<field> ...]

- ``<taskset>`` is an installed Python module that exposes
  ``load_taskset(config: TasksetConfig)``; nothing else is required.
- ``<harness>`` is optional and is also an installed Python module — one
  that exposes ``load_harness(config: HarnessConfig)``. When omitted the
  harness auto-resolves to the taskset module's own ``load_harness`` if
  present, otherwise to the base ``verifiers.v1.Harness``.
- ``--taskset.<field>`` flows into the taskset's actual ``TasksetConfig``
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
    ClientConfig as LegacyClientConfig,
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
    V1_HARNESS_KEY,
    V1_TASKSET_KEY,
    harness_config_type_from_module,
    module_supports_v1_loader,
    resolve_harness_module,
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
# Sub-configs
# ---------------------------------------------------------------------------


class SamplingConfig(BaseConfig):
    """Sampling arguments handed to the inference API."""

    temperature: float | None = None
    """Sampling temperature."""

    max_tokens: int | None = None
    """Cap on completion tokens per turn."""

    top_p: float | None = None
    """Nucleus sampling top-p."""

    extras: dict = {}
    """Additional sampling args passed through to the inference API as-is."""


class ClientConfig(BaseConfig):
    """Inference client configuration."""

    model: str = "openai/gpt-4.1-mini"
    """Model id to send to the inference API."""

    provider: str | None = None
    """Provider shorthand: one of prime, openai, anthropic, openrouter, deepseek, local.
    Resolves ``base_url`` / ``api_key_var`` / ``client_type`` defaults; explicit
    sub-fields still win on conflict."""

    base_url: str | None = None
    """API base URL (overrides --client.provider)."""

    api_key_var: str | None = None
    """Env var holding the API key (overrides --client.provider)."""

    client_type: ClientType | None = None
    """Inference client type. Default: openai_chat_completions."""

    headers: dict[str, str] = {}
    """Extra HTTP headers to attach to every inference request."""

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str | None) -> str | None:
        if value is not None and value not in PROVIDERS:
            raise ValueError(
                f"Unknown provider {value!r}; pick one of {sorted(PROVIDERS)}."
            )
        return value


# ---------------------------------------------------------------------------
# Static config base (shared between v0 and v1 dispatch).
# ---------------------------------------------------------------------------


class EvalConfigBase(BaseConfig):
    """Run / client / UI knobs shared by every ``vf-eval-v1`` invocation.

    The ``taskset`` and ``harness`` fields are added per-invocation by
    :func:`_build_v1_eval_config`, so their types match the actual modules
    selected on the command line.
    """

    taskset_name: str = Field(
        validation_alias=AliasChoices("taskset_name", "taskset-name", "env"),
    )
    """Taskset module name — an installed Python module exposing
    ``load_taskset(config: TasksetConfig)``. Settable as the first positional
    (``vf-eval-v1 <taskset_name>``)."""

    harness_name: str | None = Field(
        None, validation_alias=AliasChoices("harness_name", "harness-name")
    )
    """Harness module name — an installed Python module exposing
    ``load_harness(config: HarnessConfig)``. Settable as the second positional.
    When unset, auto-resolves to the taskset module's ``load_harness`` if
    present, otherwise the base ``verifiers.v1.Harness``."""

    client: ClientConfig = ClientConfig()
    """Inference client configuration."""

    sampling: SamplingConfig = SamplingConfig()
    """Sampling arguments."""

    num_examples: int = 5
    """Number of examples to evaluate."""

    rollouts_per_example: int = 3
    """Rollouts per example (groups)."""

    max_concurrent: int = 32
    """Maximum concurrent rollouts."""

    max_retries: int = 0
    """Max retries for transient infrastructure errors."""

    timeout: float | None = None
    """Per-rollout wall-clock timeout (seconds)."""

    num_workers: str | int = "auto"
    """Env server workers (``auto`` or an integer)."""

    disable_env_server: bool = False
    """Run rollouts in-process instead of starting env workers."""

    independent_scoring: bool = False
    """Score each rollout individually instead of by group."""

    output_dir: str | None = None
    """Custom output directory for evaluation results and logs."""

    save_results: bool = False
    """Save full rollouts to disk."""

    state_columns: list[str] = []
    """State columns to include in the saved rollouts."""

    resume: str | None = None
    """Resume from an explicit results path."""

    save_to_hf_hub: bool = False
    """Push the result dataset to the Hugging Face Hub."""

    hf_hub_dataset_name: str = ""
    """Hugging Face Hub dataset name (used with --save-to-hf-hub)."""

    verbose: bool = False
    """Verbose logging."""

    fullscreen: bool = False
    """Use Rich's alternate-screen TUI."""

    disable_tui: bool = False
    """Disable the Rich TUI and fall back to plain logging."""

    abbreviated_summary: bool = False
    """Show the compact end-of-run summary (no per-example block)."""

    heartbeat_url: str | None = None
    """Heartbeat URL pinged after each progress update."""


class EvalV0Config(EvalConfigBase):
    """Eval config for v0 envs (modules that expose only ``load_environment``)."""

    env_args: dict = {}
    """Extra kwargs passed to ``load_environment`` for v0 envs (the bundled
    harness in a v0 env is not swappable)."""

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
    selected modules' actual ``TasksetConfig`` / ``HarnessConfig`` subclasses
    at parse time.
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
        harness_module = resolve_harness_module(harness_name)
        return harness_config_type_from_module(harness_module)
    if hasattr(env_module, "load_harness"):
        factory_type = factory_config_type(env_module, "load_harness", HarnessConfig)
        return cast(type[HarnessConfig], factory_type or HarnessConfig)
    return HarnessConfig


# ---------------------------------------------------------------------------
# Argv preprocessing: positional <taskset> [<harness>] + @ file.toml peek.
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
    """Pull leading positionals + ``--taskset-name`` / ``--harness-name`` out of argv.

    Returns ``(cleaned_argv, taskset_name, harness_name)``.

    Positionals are recognised at the front of argv only: scanning stops at
    the first ``-``/``--``/``@`` token. The bare positionals are dropped from
    the cleaned argv and re-injected as ``--taskset-name`` / ``--harness-name``
    flags so the downstream pydantic-config parse sees them as ordinary fields.

    Also reads ``--taskset-name`` / ``--harness-name`` explicit flags from
    anywhere in argv, and peeks at any ``@ path/to.toml`` argument to backfill
    when the user hasn't already set them on the command line. Explicit flags
    / positionals win over TOML values.

    Raises ``SystemExit`` if more than two bare positionals appear.
    """
    taskset_name: str | None = None
    harness_name: str | None = None

    # Pass 1: leading positionals (stop at first non-positional).
    consumed_positionals = 0
    cursor = 1
    while cursor < len(argv):
        token = argv[cursor]
        if token.startswith("-") or token.startswith("@"):
            break
        if consumed_positionals == 0:
            taskset_name = token
        elif consumed_positionals == 1:
            harness_name = token
        else:
            raise SystemExit(
                "vf-eval-v1 takes at most two positionals (taskset + optional "
                f"harness); got {token!r}."
            )
        consumed_positionals += 1
        cursor += 1

    rest = argv[cursor:]
    cleaned = [argv[0], *rest]

    # Pass 2: explicit --taskset-name / --harness-name flags anywhere in rest.
    j = 0
    while j < len(rest):
        token = rest[j]
        for flag, setter in (
            ("--taskset-name", "taskset"),
            ("--harness-name", "harness"),
        ):
            if token == flag and j + 1 < len(rest):
                if setter == "taskset":
                    taskset_name = rest[j + 1]
                else:
                    harness_name = rest[j + 1]
                break
            if token.startswith(flag + "="):
                value = token.split("=", 1)[1]
                if setter == "taskset":
                    taskset_name = value
                else:
                    harness_name = value
                break
        j += 1

    # Pass 3: TOML peek for any @ file references.
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
        if taskset_name is None and isinstance(data.get("taskset_name"), str):
            taskset_name = data["taskset_name"]
        if harness_name is None and isinstance(data.get("harness_name"), str):
            harness_name = data["harness_name"]

    # Re-inject the positionals as explicit flags if they aren't already on
    # the command line. The explicit flag form wins on conflict because we
    # already updated taskset_name / harness_name from it in pass 2.
    def _has_flag(name: str) -> bool:
        return any(a == name or a.startswith(name + "=") for a in cleaned[1:])

    if taskset_name is not None and not _has_flag("--taskset-name"):
        cleaned.extend(["--taskset-name", taskset_name])
    if harness_name is not None and not _has_flag("--harness-name"):
        cleaned.extend(["--harness-name", harness_name])

    return cleaned, taskset_name, harness_name


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


def _resolve_legacy_client_config(config: EvalConfigBase) -> LegacyClientConfig:
    client = config.client
    provider = client.provider or DEFAULT_PROVIDER
    provider_cfg = PROVIDERS[provider]
    api_base_url = client.base_url or provider_cfg["url"]
    api_key_var = client.api_key_var or provider_cfg["key"]
    client_type = client.client_type or provider_cfg.get(
        "client_type", "openai_chat_completions"
    )
    headers = {str(k): str(v) for k, v in client.headers.items()}
    return LegacyClientConfig(
        client_type=cast(ClientType, client_type),
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        extra_headers=headers,
        extra_headers_from_state={"X-Session-ID": "example_id"},
    )


def _resolve_sampling_args(config: EvalConfigBase) -> dict[str, Any]:
    sampling = config.sampling
    args: dict[str, Any] = dict(sampling.extras or {})
    if sampling.max_tokens is not None and "max_tokens" not in args:
        args["max_tokens"] = sampling.max_tokens
    if sampling.temperature is not None and "temperature" not in args:
        args["temperature"] = sampling.temperature
    if sampling.top_p is not None and "top_p" not in args:
        args["top_p"] = sampling.top_p
    return args


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
        env_id=config.taskset_name,
        name=None,
        env_args=env_args,
        env_dir_path="./environments",
        model=config.client.model,
        client_config=_resolve_legacy_client_config(config),
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
    taskset_name: str | None, harness_name: str | None
) -> type[BaseConfig]:
    """Pick the right BaseConfig subclass for this invocation."""
    if taskset_name is None:
        # No taskset yet — fall back to the base. cli() will report
        # --taskset-name as required (or render the top-level --help).
        return EvalConfigBase
    env_module = import_env_module(taskset_name)
    if module_supports_v1_loader(env_module):
        taskset_cls = _resolve_taskset_config_class(env_module)
        harness_cls = _resolve_harness_config_class(env_module, harness_name)
        return _build_v1_eval_config(taskset_cls, harness_cls)
    return EvalV0Config


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv) if argv is None else [sys.argv[0], *argv]
    cleaned_argv, taskset_name, harness_name = _extract_initial_args(raw_argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    if taskset_name is not None and not check_hub_env_installed(taskset_name):
        raise SystemExit(
            f"Taskset {taskset_name!r} is not installed.\n"
            f"  prime env install {taskset_name}"
        )

    config_cls = _resolve_config_class(taskset_name, harness_name)

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
    "ClientConfig",
    "EvalConfigBase",
    "EvalV0Config",
    "SamplingConfig",
    "main",
]


if __name__ == "__main__":
    main()
