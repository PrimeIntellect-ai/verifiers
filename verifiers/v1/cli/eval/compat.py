"""The frozen v0 eval config surface, normalized at the ingestion boundary.

Hosted-eval sandboxes and old local configs still speak the classic v0 eval
dialect: flat run fields plus a single ``[[eval]]`` table (``env_id`` or a
transitional ``taskset``/``group_size`` shape). True v0 env ids dispatch to the
old evaluator before this converter runs; this module maps transitional v1
taskset configs onto a native v1 :class:`EvalConfig` TOML. The mapping leans on
``EvalConfig`` itself: any field the model accepts (by name or alias) passes
through untouched; only shape changes are hand-translated.
Prime reuses ``merge_sampling_args``, ``build_extra_headers``, and
``write_converted_eval_config`` for its hosted-eval surface.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import AliasChoices

from verifiers.types import _validate_extra_headers_value
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.types import local_env_id

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
_SILENT_DROP_FIELDS = {
    "name",
    "version",
    "env_target",
    "env_dir_path",
}
_TUI_FIELDS = ("disable_tui", "debug")


def is_transitional_config(raw: dict[str, Any]) -> bool:
    """Whether a parsed TOML uses the v0 eval dialect (``env_id`` / ``[[eval]]``)."""
    return "env_id" in raw or "eval" in raw


def merge_sampling_args(
    sampling_args: dict[str, Any] | None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Overlay v0's standalone --max-tokens/--temperature onto a sampling-args dict;
    explicit sampling-args keys win."""
    merged = dict(sampling_args or {})
    if max_tokens is not None:
        merged.setdefault("max_tokens", max_tokens)
    if temperature is not None:
        merged.setdefault("temperature", temperature)
    return merged


def build_extra_headers(
    headers: dict[str, str] | list[str] | None = None,
    header: list[str] | None = None,
) -> dict[str, str]:
    """Merge v0's ``headers`` table and repeatable ``header`` "Name: Value" entries.
    Hosted TOMLs write "Name: Value" strings under ``headers`` too; later entries win."""
    if header is not None and not isinstance(header, list):
        raise ValueError("'header' must be a list of 'Name: Value' strings")
    entries = list(header or [])
    if isinstance(headers, list):  # hosted TOMLs write "Name: Value" strings here
        entries += headers
        headers = None
    merged = _validate_extra_headers_value(headers) if headers is not None else {}
    for entry in entries:
        if not isinstance(entry, str) or ":" not in entry:
            raise ValueError(f"--header must be 'Name: Value', got: {entry!r}")
        name, value = entry.split(":", 1)
        if not name.strip():
            raise ValueError("--header name cannot be empty")
        merged[name.strip()] = value.strip()
    return merged


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a transitional eval TOML (``env_id`` and/or a single ``[[eval]]``).

    Hosted raw-v1 configs are exactly this shape: top-level run fields plus one
    ``[[eval]]`` table holding the env config.
    """
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


def build_v1_eval_config(fields: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Map v0 fields onto a v1 eval config dict; returns (config, warnings).

    ``fields`` carries either a v1 ``taskset`` or a legacy ``id``. Legacy hub refs
    normalize to the local package name; v1 taskset ids stay local-only. Only shape
    changes are translated here — anything ``EvalConfig`` accepts passes through.
    """
    fields = dict(fields)
    warnings: list[str] = []

    if fields.pop("resume", None):
        raise ValueError("--resume is a v1 flag now: eval --resume <output-dir>")
    if fields.pop("save_to_hf_hub", None) or fields.pop("hf_hub_dataset_name", None):
        raise ValueError("--save-to-hf-hub has no v1 equivalent")
    if fields.pop("endpoint_id", None):
        raise ValueError(
            "endpoint_id has no v1 equivalent; set model and client.base_url"
        )
    client_type = fields.pop("api_client_type", None)

    for field, reason in _WARN_DROP_FIELDS.items():
        if fields.pop(field, None) not in (None, False, []):
            warnings.append(f"ignoring v0-only `{field}`: {reason}")
    if fields.pop("save_results", None) is not None:
        warnings.append(
            "ignoring v0-only `save_results`: v1 evals always write artifacts"
        )
    for field in _SILENT_DROP_FIELDS:
        fields.pop(field, None)
    if any([fields.pop(field, False) for field in _TUI_FIELDS]):
        fields["rich"] = False

    # environment: taskset (v1) or legacy id; env_args attach to whichever is set
    taskset = fields.pop("taskset", None)
    legacy_id = fields.pop("id", None)
    if isinstance(legacy_id, str):
        legacy_id = local_env_id(legacy_id)
    env_args = fields.pop("env_args", None) or {}
    if legacy_id and isinstance(taskset, dict) and taskset.get("id"):
        raise ValueError("config cannot contain both a legacy env_id and a taskset")
    if legacy_id:
        fields["id"] = legacy_id
        args = {**env_args, **(fields.pop("args", None) or {})}
        if args:
            fields["args"] = args
    elif isinstance(taskset, dict) and taskset.get("id"):
        # env_args are taskset kwargs in v1; explicit taskset keys win
        fields["taskset"] = {**env_args, **taskset}
    else:
        raise ValueError("config requires a v1 taskset.id or a legacy env id")
    is_legacy = bool(legacy_id)

    # v0 counts: -1 num_examples means all (v1: unset); hosted TOMLs carry the rollout
    # count under both names, group_size wins. EvalConfig's aliases do the renames.
    if fields.get("num_examples") == -1:
        del fields["num_examples"]
    if "group_size" in fields:
        fields.pop("rollouts_per_example", None)

    # v0 --timeout is per-rollout seconds; transitional TOMLs may carry the v1 table
    timeout = fields.get("timeout")
    if timeout is not None and not isinstance(timeout, dict):
        fields["timeout"] = {"rollout": timeout}
    max_retries = fields.pop("max_retries", None)
    if max_retries is not None:
        retries = dict(fields.get("retries") or {})
        retries["rollout"] = {**retries.get("rollout", {}), "max_retries": max_retries}
        fields["retries"] = retries

    # sampling: v0 sampling_args (with extra_body flattened — v1 passes provider-specific
    # keys through) plus standalone flags, overlaid on any v1 `sampling` table; v0 wins.
    sampling_args = merge_sampling_args(
        fields.pop("sampling_args", None),
        max_tokens=fields.pop("max_tokens", None),
        temperature=fields.pop("temperature", None),
    )
    extra_body = sampling_args.pop("extra_body", None)
    if isinstance(extra_body, dict):
        sampling_args = {**extra_body, **sampling_args}
    sampling = {**(fields.pop("sampling", None) or {}), **sampling_args}
    if sampling:
        fields["sampling"] = sampling

    # client: provider shorthand, then explicit v0 fields, over any v1 `client` table
    client = dict(fields.pop("client", None) or {})
    provider = fields.pop("provider", None)
    if provider:
        provider_config = PROVIDER_CONFIGS.get(provider)
        if provider_config is None:
            raise ValueError(
                f"unknown provider `{provider}` "
                f"(known: {', '.join(sorted(PROVIDER_CONFIGS))})"
            )
        if provider_config.get("client_type") and not is_legacy:
            raise ValueError(f"--provider {provider} has no v1 equivalent")
        client["base_url"] = provider_config["url"]
        client["api_key_var"] = provider_config["key"]
        if provider_config.get("client_type"):
            client["v0_client_type"] = provider_config["client_type"]
    if client_type not in (None, "openai_chat_completions"):
        if not is_legacy:
            raise ValueError(f"--api-client-type {client_type} has no v1 equivalent")
        client["v0_client_type"] = client_type
    for src, dst in (("api_base_url", "base_url"), ("api_key_var", "api_key_var")):
        if value := fields.pop(src, None):
            client[dst] = value
    headers = build_extra_headers(
        fields.pop("headers", None), fields.pop("header", None)
    )
    if headers:
        client["headers"] = {**client.get("headers", {}), **headers}
    if client:
        fields["client"] = client

    # Everything EvalConfig accepts (field names + alias choices) passes through
    # unchanged — its aliases already speak v0. The rest is unknown v0: warn and drop.
    accepted: set[str] = set()
    for name, spec in EvalConfig.model_fields.items():
        accepted.add(name)
        if isinstance(spec.validation_alias, AliasChoices):
            accepted.update(
                a for a in spec.validation_alias.choices if isinstance(a, str)
            )
    config: dict[str, Any] = {}
    for key in sorted(fields):
        if key in accepted:
            config[key] = fields[key]
        else:
            warnings.append(f"ignoring unknown v0 field `{key}`")
    return config, warnings


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


def convert_transitional_config(
    raw: dict[str, Any], source: Path
) -> tuple[Path, list[str]]:
    """Convert a parsed transitional eval TOML into a runnable v1 config file.

    The CLI's ingestion boundary: ids must be locally importable (hosts install
    hub refs and pin the local name before handing the file to verifiers).
    """
    config, warnings = build_v1_eval_config(_flatten(raw))
    path = write_converted_eval_config(
        config, header_comment=f"converted from v0 eval config {source}"
    )
    return path, warnings
