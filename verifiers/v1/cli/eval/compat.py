"""The frozen v0 eval config surface, normalized at the ingestion boundary.

Hosted-eval sandboxes and old local configs still speak the classic v0 eval
dialect: flat run fields plus a single ``[[eval]]`` table (``env_id`` or a
transitional ``taskset``/``group_size`` shape). The v0 evaluator is gone, so
this module maps that surface onto a native v1 :class:`EvalConfig` TOML — v0
env ids route through the legacy bridge via the usual auto-detection. Prime
reuses these helpers for its own frozen v0 argv surface.
"""

import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from verifiers.types import _validate_extra_headers_value

# v0 provider shorthand: resolves to a client base_url + API-key env var.
PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "prime": {"url": "https://api.pinference.ai/api/v1", "key": "PRIME_API_KEY"},
    "openrouter": {"url": "https://openrouter.ai/api/v1", "key": "OPENROUTER_API_KEY"},
    "openai": {"url": "https://api.openai.com/v1", "key": "OPENAI_API_KEY"},
    "anthropic": {
        "url": "https://api.anthropic.com",
        "key": "ANTHROPIC_API_KEY",
        "client_type": "anthropic_messages",
    },
    "minimax": {"url": "https://api.minimax.chat/v1", "key": "MINIMAX_API_KEY"},
    "deepseek": {"url": "https://api.deepseek.com/v1", "key": "DEEPSEEK_API_KEY"},
    "glm": {"url": "https://open.bigmodel.cn/api/paas/v4", "key": "GLM_API_KEY"},
    "local": {"url": "http://localhost:8000/v1", "key": "VLLM_API_KEY"},
    "vllm": {"url": "http://localhost:8000/v1", "key": "VLLM_API_KEY"},
}

# v0 fields with no v1 counterpart: dropping them changes at most what gets
# logged/saved, never what runs — surface a warning and continue.
_WARN_DROP_FIELDS = {
    "state_columns": "v1 traces always carry the full rollout state",
    "independent_scoring": "v1 scoring is defined by the taskset's rewards",
    "no_interleave_scoring": "v1 scoring is defined by the taskset's rewards",
    "shuffle_seed": "the v1 CLI has no shuffle seed",
    "header_from_state": "the v1 client has no per-state headers",
    "num_workers": "worker pools are configured via --pool.* in v1",
    "disable_env_server": "v1 runs rollouts in-process by default",
    "heartbeat_url": "the v1 CLI has no heartbeat",
    "abbreviated_summary": "the v1 CLI has no summary modes",
    "endpoints_path": "the v1 CLI has no endpoints registry",
}
# Purely presentational v0 fields; nothing to warn about.
_SILENT_DROP_FIELDS = {"save_results", "fullscreen", "name", "version"}
_TUI_FIELDS = {"disable_tui", "debug"}


def is_transitional_config(raw: dict[str, Any]) -> bool:
    """Whether a parsed TOML uses the v0 eval dialect (``env_id`` / ``[[eval]]``)."""
    return "env_id" in raw or "eval" in raw


def merge_sampling_args(
    sampling_args: dict[str, Any] | None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    prefer_existing_keys: bool = True,
    include_none_max_tokens: bool = False,
) -> dict[str, Any]:
    """Overlay v0's standalone --max-tokens/--temperature onto a sampling-args dict."""
    merged_sampling_args = dict(sampling_args or {})

    if (not prefer_existing_keys or "max_tokens" not in merged_sampling_args) and (
        include_none_max_tokens or max_tokens is not None
    ):
        merged_sampling_args["max_tokens"] = max_tokens

    if temperature is not None and (
        not prefer_existing_keys or "temperature" not in merged_sampling_args
    ):
        merged_sampling_args["temperature"] = temperature

    return merged_sampling_args


def build_extra_headers(raw: dict[str, Any]) -> dict[str, str]:
    """Merge v0's ``headers`` table and repeatable ``header`` "Name: Value" entries."""
    eval_headers_table: dict[str, str] = {}
    raw_headers = raw.get("headers")
    if raw_headers is not None:
        eval_headers_table = _validate_extra_headers_value(raw_headers)

    raw_header_values = raw.get("header")
    if raw_header_values is None:
        raw_header_values = []
    if not isinstance(raw_header_values, list):
        raise ValueError("'header' must be a list of 'Name: Value' strings")

    eval_headers_from_list: dict[str, str] = {}
    for header_value in raw_header_values:
        if not isinstance(header_value, str):
            raise ValueError(
                "Each 'header' entry must be a string 'Name: Value', "
                f"got: {header_value!r}"
            )
        if ":" not in header_value:
            raise ValueError(f"--header must be 'Name: Value', got: {header_value!r}")
        key, value = header_value.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError("--header name cannot be empty")
        eval_headers_from_list[key] = value

    return {**eval_headers_table, **eval_headers_from_list}


def transitional_config_to_fields(config_path: Path) -> dict[str, Any]:
    """Flatten a transitional eval TOML (``env_id`` and/or a single ``[[eval]]``).

    Hosted raw-v1 configs are exactly this shape: top-level run fields plus one
    ``[[eval]]`` table holding the env config.
    """
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if "ablation" in raw:
        raise ValueError(
            "[[ablation]] configs are not supported; expand them into v1 configs"
        )

    entries = raw.get("eval")
    if entries is None:
        merged = dict(raw)
    else:
        if (
            not isinstance(entries, list)
            or len(entries) != 1
            or not isinstance(entries[0], dict)
        ):
            raise ValueError(
                "only single-entry [[eval]] configs can run here; "
                "split multi-env configs into one file per environment"
            )
        merged = {k: v for k, v in raw.items() if k != "eval"}
        merged.update(entries[0])

    if "env_id" in merged:
        if merged.get("id"):
            raise ValueError("config cannot contain both id and env_id")
        merged["id"] = merged.pop("env_id")
    return merged


def build_v1_eval_config(
    fields: dict[str, Any], *, tui_disabled: bool = False
) -> tuple[dict[str, Any], list[str]]:
    """Map v0 fields onto a v1 eval config dict; returns (config, warnings).

    ``fields`` carries either a v1 ``taskset`` or a legacy ``id``; both must
    already name a locally importable package (hosts resolve hub refs first).
    """
    fields = dict(fields)
    warnings: list[str] = []
    config: dict[str, Any] = {}

    if fields.pop("resume", None):
        raise ValueError("--resume is a v1 flag now: eval --resume <output-dir>")
    if fields.pop("save_to_hf_hub", None) or fields.pop("hf_hub_dataset_name", None):
        raise ValueError("--save-to-hf-hub has no v1 equivalent")
    if fields.pop("endpoint_id", None):
        raise ValueError(
            "endpoint_id has no v1 equivalent; set model and client.base_url"
        )
    client_type = fields.pop("api_client_type", None)
    if client_type not in (None, "openai_chat_completions"):
        raise ValueError(f"--api-client-type {client_type} has no v1 equivalent")

    for field, reason in _WARN_DROP_FIELDS.items():
        if fields.pop(field, None) not in (None, False, []):
            warnings.append(f"ignoring v0-only `{field}`: {reason}")
    for field in _SILENT_DROP_FIELDS:
        fields.pop(field, None)
    for field in _TUI_FIELDS:
        tui_disabled = bool(fields.pop(field, False)) or tui_disabled
    if tui_disabled:
        config["rich"] = False

    # environment: taskset (v1) or legacy id; env_args attach to whichever is set
    taskset = fields.pop("taskset", None)
    legacy_id = fields.pop("id", None)
    env_args = fields.pop("env_args", None) or {}
    if legacy_id:
        config["id"] = legacy_id
        args = {**env_args, **(fields.pop("args", None) or {})}
        if args:
            config["args"] = args
    elif isinstance(taskset, dict) and taskset.get("id"):
        # env_args are taskset kwargs in v1; explicit taskset keys win
        config["taskset"] = {**env_args, **taskset}
    else:
        raise ValueError("config requires a v1 taskset.id or a legacy env id")

    for key in (
        "harness",
        "pool",
        "extra_env_kwargs",
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "multiplex",
        "model",
        "max_concurrent",
        "shuffle",
        "output_dir",
    ):
        value = fields.pop(key, None)
        if value is not None:
            config[key] = value
    if fields.pop("verbose", False):
        config["verbose"] = True

    num_examples = fields.pop("num_examples", None)
    if num_examples is not None and num_examples != -1:  # v0's -1 = all = v1's unset
        config["num_tasks"] = num_examples
    num_rollouts = fields.pop("group_size", None) or fields.pop(
        "rollouts_per_example", None
    )
    fields.pop("rollouts_per_example", None)
    if num_rollouts is not None:
        config["num_rollouts"] = num_rollouts

    # v0 --timeout is per-rollout seconds; transitional TOMLs may carry the v1 table
    timeout = fields.pop("timeout", None)
    if isinstance(timeout, dict):
        config["timeout"] = timeout
    elif timeout is not None:
        config["timeout"] = {"rollout": timeout}

    retries = dict(fields.pop("retries", None) or {})
    max_retries = fields.pop("max_retries", None)
    if max_retries is not None:
        retries["rollout"] = {**retries.get("rollout", {}), "max_retries": max_retries}
    if retries:
        config["retries"] = retries

    sampling = _build_sampling(fields)
    if sampling:
        config["sampling"] = sampling

    client = _build_client(fields)
    if client:
        config["client"] = client

    fields.pop("env_target", None)
    fields.pop("env_dir_path", None)
    for leftover in sorted(fields):
        warnings.append(f"ignoring unknown v0 field `{leftover}`")
    return config, warnings


def _build_sampling(fields: dict[str, Any]) -> dict[str, Any]:
    """Per-env v1 ``sampling`` overlaid with v0 sampling args; ``extra_body``
    flattens into the sampling table (v1 passes provider-specific keys through)."""
    sampling = dict(fields.pop("sampling", None) or {})
    sampling_args = merge_sampling_args(
        fields.pop("sampling_args", None),
        max_tokens=fields.pop("max_tokens", None),
        temperature=fields.pop("temperature", None),
        prefer_existing_keys=True,
    )
    extra_body = sampling_args.pop("extra_body", None)
    if isinstance(extra_body, dict):
        sampling_args = {**extra_body, **sampling_args}
    return {**sampling, **sampling_args}


def _build_client(fields: dict[str, Any]) -> dict[str, Any]:
    client: dict[str, Any] = {}
    provider = fields.pop("provider", None)
    if provider:
        provider_config = PROVIDER_CONFIGS.get(provider)
        if provider_config is None:
            raise ValueError(
                f"unknown provider `{provider}` "
                f"(known: {', '.join(sorted(PROVIDER_CONFIGS))})"
            )
        if provider_config.get("client_type"):
            raise ValueError(f"--provider {provider} has no v1 equivalent")
        client["base_url"] = provider_config["url"]
        client["api_key_var"] = provider_config["key"]

    base_url = fields.pop("api_base_url", None)
    if base_url:
        client["base_url"] = base_url
    api_key_var = fields.pop("api_key_var", None)
    if api_key_var:
        client["api_key_var"] = api_key_var

    header_list = fields.pop("header", None) or []
    headers_value = fields.pop("headers", None)
    if isinstance(headers_value, list):  # hosted TOMLs write "K: V" strings here
        header_list = [*header_list, *headers_value]
        headers_value = None
    raw_headers = {"header": header_list, "headers": headers_value}
    headers = build_extra_headers(
        {k: v for k, v in raw_headers.items() if v is not None}
    )
    if headers:
        client["headers"] = headers
    return client


def write_converted_eval_config(
    config: dict[str, Any], header_comment: str | None = None
) -> Path:
    """Write a converted config to a temp TOML the CLI can run via ``@ path``."""
    fd, name = tempfile.mkstemp(prefix="verifiers-eval-v1-", suffix=".toml")
    with os.fdopen(fd, "wb") as handle:
        if header_comment:
            handle.write(f"# {header_comment}\n".encode())
        tomli_w.dump(config, handle)
    return Path(name)


def convert_transitional_config(config_path: Path) -> tuple[Path, list[str]]:
    """Convert a transitional eval TOML into a runnable v1 config file.

    The CLI's ingestion boundary: ids must be locally importable (hosts install
    hub refs and pin the local name before handing the file to verifiers).
    """
    fields = transitional_config_to_fields(config_path)
    config, warnings = build_v1_eval_config(fields)
    path = write_converted_eval_config(
        config, header_comment=f"converted from v0 eval config {config_path}"
    )
    return path, warnings
