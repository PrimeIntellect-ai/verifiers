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

v1-only by design: modules that don't expose ``load_taskset`` are
rejected up front. Use the legacy ``vf-eval`` for v0 environments.
"""

import asyncio
import json
import logging
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import AliasChoices, Field, create_model
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
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.taskset import TasksetConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider shorthand (mirrors the legacy ``vf-eval`` --provider table).
# ---------------------------------------------------------------------------

Provider = Literal["prime", "openai", "anthropic", "openrouter", "deepseek", "local"]

PROVIDERS: dict[Provider, dict[str, str]] = {
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
DEFAULT_PROVIDER: Provider = "prime"


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

    extras: dict[str, Any] = {}
    """Additional sampling args passed through to the inference API as-is."""


class ClientConfig(BaseConfig):
    """Inference client configuration."""

    model: str = "openai/gpt-4.1-mini"
    """Model id to send to the inference API."""

    provider: Provider | None = None
    """Provider shorthand. Resolves ``base_url`` / ``api_key_var`` /
    ``client_type`` defaults; explicit sub-fields still win on conflict."""

    base_url: str | None = None
    """API base URL (overrides --client.provider)."""

    api_key_var: str | None = None
    """Env var holding the API key (overrides --client.provider)."""

    client_type: ClientType | None = None
    """Inference client type. Default: openai_chat_completions."""

    headers: dict[str, str] = {}
    """Extra HTTP headers to attach to every inference request."""


# ---------------------------------------------------------------------------
# Static config base
# ---------------------------------------------------------------------------


class EvalConfigBase(BaseConfig):
    """Run / client / UI knobs shared by every ``vf-eval-v1`` invocation.

    The ``taskset`` and ``harness`` fields are added per-invocation by
    :func:`_build_v1_eval_config`, so their types match the actual modules
    selected on the command line.
    """

    taskset_name: str
    """Taskset module name — an installed Python module exposing
    ``load_taskset(config: TasksetConfig)``. Settable as the first positional
    (``vf-eval-v1 <taskset_name>``)."""

    harness_name: str | None = None
    """Harness module name — an installed Python module exposing
    ``load_harness(config: HarnessConfig)``. Settable as the second positional.
    When unset, auto-resolves to the taskset module's ``load_harness`` if
    present, otherwise the base ``verifiers.v1.Harness``."""

    client: ClientConfig = ClientConfig()
    """Inference client configuration."""

    sampling: SamplingConfig = SamplingConfig()
    """Sampling arguments."""

    num_examples: int = Field(5, validation_alias=AliasChoices("num_examples", "n"))
    """Number of examples to evaluate."""

    rollouts_per_example: int = Field(
        3, validation_alias=AliasChoices("rollouts_per_example", "r")
    )
    """Rollouts per example (groups)."""

    max_concurrent: int = 32
    """Maximum concurrent rollouts."""

    max_retries: int = 0
    """Max retries for transient infrastructure errors."""

    timeout: float | None = None
    """Per-rollout wall-clock timeout (seconds)."""

    num_workers: Literal["auto"] | int = "auto"
    """Env server workers (``auto`` or an integer)."""

    disable_env_server: bool = Field(
        False, validation_alias=AliasChoices("disable_env_server", "d")
    )
    """Run rollouts in-process instead of starting env workers."""

    independent_scoring: bool = False
    """Score each rollout individually instead of by group."""

    output_dir: Path | None = None
    """Custom output directory for evaluation results and logs."""

    save_results: bool = False
    """Save full rollouts to disk."""

    state_columns: list[str] = []
    """State columns to include in the saved rollouts."""

    resume: Path | None = None
    """Resume from an explicit results path."""

    save_to_hf_hub: bool = False
    """Push the result dataset to the Hugging Face Hub."""

    hf_hub_dataset_name: str = ""
    """Hugging Face Hub dataset name (used with --save-to-hf-hub)."""

    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Verbose logging."""

    fullscreen: bool = False
    """Use Rich's alternate-screen TUI."""

    disable_tui: bool = False
    """Disable the Rich TUI and fall back to plain logging."""

    abbreviated_summary: bool = False
    """Show the compact end-of-run summary (no per-example block)."""

    heartbeat_url: str | None = None
    """Heartbeat URL pinged after each progress update."""


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
    """Resolve the concrete ``TasksetConfig`` subclass a taskset module declares.

    Inspects the module's ``load_taskset(config: ...)`` factory signature so the
    CLI can type ``--taskset.*`` flags against the real config. This is the
    taskset twin of ``harness_config_type_from_module`` (no shared helper exists
    for tasksets yet); both delegate to ``factory_config_type``.
    """
    config_type = factory_config_type(env_module, "load_taskset", TasksetConfig)
    if config_type is None:
        raise TypeError(
            f"{env_module.__name__}.load_taskset must accept a typed config."
        )
    return cast(type[TasksetConfig], config_type)


def _resolve_harness_config_class(
    env_module: Any, harness_name: str | None
) -> type[HarnessConfig]:
    """Pick the ``HarnessConfig`` subclass to type ``--harness.*`` flags against.

    Precedence: an explicit ``--harness`` module's ``load_harness`` config; else
    the taskset module's own ``load_harness`` if it defines one; else the base
    ``HarnessConfig``.
    """
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


def _peek_toml_files(args: list[str]) -> list[dict[str, Any]]:
    """Parse every ``@ file`` / ``@file`` config reference found in ``args``."""
    files: list[dict[str, Any]] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "@" and i + 1 < len(args):
            files.append(_peek_toml(Path(args[i + 1])))
            i += 2
        elif token.startswith("@") and token != "@":
            files.append(_peek_toml(Path(token[1:])))
            i += 1
        else:
            i += 1
    return files


def _name_from_flag(args: list[str], flag: str) -> str | None:
    """Value of ``--flag value`` / ``--flag=value`` in ``args`` (or None)."""
    for i, token in enumerate(args):
        if token == flag and i + 1 < len(args):
            return args[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _has_flag(args: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in args)


def _extract_initial_args(
    argv: list[str],
) -> tuple[list[str], str | None, str | None]:
    """Resolve the taskset / harness names from argv before the typed parse.

    The ``--taskset.*`` / ``--harness.*`` schema is built per-module, so the
    names must be known up front. They can come from (in precedence order):
    leading positionals, explicit ``--taskset-name`` / ``--harness-name``
    flags, or a ``@ config.toml`` file. Leading positionals are re-injected as
    the matching flags so the downstream pydantic-config parse sees them as
    ordinary fields.

    Returns ``(cleaned_argv, taskset_name, harness_name)``. Raises
    ``SystemExit`` if more than two leading positionals appear.
    """
    prog, *args = argv

    # Leading positionals only — stop at the first flag / @-file token.
    positionals: list[str] = []
    for token in args:
        if token.startswith("-") or token.startswith("@"):
            break
        positionals.append(token)
    if len(positionals) > 2:
        raise SystemExit(
            "vf-eval-v1 takes at most two positionals (taskset + optional "
            f"harness); got {positionals[2]!r}."
        )
    rest = args[len(positionals) :]
    taskset_name = positionals[0] if positionals else None
    harness_name = positionals[1] if len(positionals) > 1 else None

    # Explicit flags win over positionals; TOML only backfills what's unset.
    taskset_name = _name_from_flag(rest, "--taskset-name") or taskset_name
    harness_name = _name_from_flag(rest, "--harness-name") or harness_name
    if taskset_name is None or harness_name is None:
        for data in _peek_toml_files(rest):
            if taskset_name is None and isinstance(data.get("taskset_name"), str):
                taskset_name = data["taskset_name"]
            if harness_name is None and isinstance(data.get("harness_name"), str):
                harness_name = data["harness_name"]

    # Re-inject positionals as flags (skip if the flag is already present).
    cleaned = [prog, *rest]
    if taskset_name is not None and not _has_flag(rest, "--taskset-name"):
        cleaned += ["--taskset-name", taskset_name]
    if harness_name is not None and not _has_flag(rest, "--harness-name"):
        cleaned += ["--harness-name", harness_name]
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
        output_dir=str(config.output_dir) if config.output_dir else None,
        state_columns=config.state_columns,
        save_results=config.save_results,
        resume_path=config.resume,
        save_to_hf_hub=config.save_to_hf_hub,
        hf_hub_dataset_name=config.hf_hub_dataset_name,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _resolve_config_class(
    taskset_name: str | None, harness_name: str | None
) -> type[BaseConfig]:
    """Pick the right BaseConfig subclass for this invocation.

    Raises ``SystemExit`` if the taskset module does not expose
    ``load_taskset``. ``vf-eval-v1`` is v1-only by design; the legacy
    ``vf-eval`` handles v0 envs.
    """
    if taskset_name is None:
        # No taskset yet — fall back to the base. cli() will report
        # --taskset-name as required (or render the top-level --help).
        return EvalConfigBase
    env_module = import_env_module(taskset_name)
    if not module_supports_v1_loader(env_module):
        raise SystemExit(
            f"Taskset {taskset_name!r} does not expose load_taskset. "
            "vf-eval-v1 only loads v1 taskset modules; use the legacy "
            "`vf-eval` for v0 environments."
        )
    taskset_cls = _resolve_taskset_config_class(env_module)
    harness_cls = _resolve_harness_config_class(env_module, harness_name)
    return _build_v1_eval_config(taskset_cls, harness_cls)


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
        config = cast(EvalConfigBase, cli(config_cls))
    finally:
        sys.argv = saved_argv

    if config.disable_tui and config.fullscreen:
        raise SystemExit(
            "error: --disable-tui and --fullscreen are mutually exclusive."
        )
    if config.disable_tui:
        setup_logging(get_log_level(config.verbose))

    eval_config = _build_eval_config(config, _v1_env_args(config))
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
    "SamplingConfig",
    "main",
]


if __name__ == "__main__":
    main()
