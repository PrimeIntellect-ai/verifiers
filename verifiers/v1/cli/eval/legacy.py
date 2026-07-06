"""Legacy (v0) eval compatibility for the v1 `eval` entrypoint.

The v1 CLI owns input dispatch, but classic `load_environment(...)` envs should
still run through the v0 evaluator and write the v0 artifact shape
(`metadata.json` + v0 `results.jsonl` rows).
"""

import argparse
import asyncio
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, cast

from verifiers import setup_logging
from verifiers.types import (
    ClientConfig,
    ClientType,
    EndpointClientConfig,
    EndpointConfig,
    EvalConfig as LegacyEvalConfig,
    EvalRunConfig,
    _validate_extra_headers_value,
)
from verifiers.utils.eval_utils import (
    get_log_level,
    load_endpoints,
    load_toml_config,
    resolve_endpoints_file,
    run_evaluations,
)
from verifiers.utils.import_utils import load_toml
from verifiers.utils.path_utils import (
    find_latest_incomplete_eval_results_path,
    is_valid_eval_results_path,
)
from verifiers.v1.loaders import is_legacy_env

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_ENV_DIR_PATH = "./environments"
DEFAULT_ENDPOINTS_PATH = "./configs/endpoints.toml"
DEFAULT_NUM_EXAMPLES = 5
DEFAULT_ROLLOUTS_PER_EXAMPLE = 3
DEFAULT_MAX_CONCURRENT = 32
DEFAULT_CLIENT_TYPE = "openai_chat_completions"
DEFAULT_PROVIDER = "prime"

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


def is_legacy_eval_invocation(argv: list[str]) -> bool:
    """Return whether raw CLI args should use the old v0 eval runner."""
    if not argv or any(arg in ("-h", "--help") for arg in argv):
        return False

    target = _legacy_target(argv)
    if target is None:
        return False
    if target.endswith(".toml") and Path(target).is_file():
        return _legacy_toml(Path(target))
    return is_legacy_env(target)


def run_legacy_eval_cli(argv: list[str]) -> None:
    """Parse old v0 eval inputs and run the old evaluator."""
    args = _parse_args(_normalize_argv(argv))
    if args.disable_tui and args.fullscreen:
        raise SystemExit("error: --disable-tui and --fullscreen are mutually exclusive")
    setup_logging(get_log_level(args.verbose))
    asyncio.run(run_evaluations(_eval_run_config(args)))


def _legacy_target(argv: list[str]) -> str | None:
    for index, arg in enumerate(argv):
        if arg == "@" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("@") and len(arg) > 1:
            return arg[1:]
        if arg in ("--id", "--taskset.id") and index + 1 < len(argv):
            return argv[index + 1]
        for prefix in ("--id=", "--taskset.id="):
            if arg.startswith(prefix):
                return arg.split("=", 1)[1]
    for arg in argv:
        if not arg.startswith("-"):
            return arg
    return None


def _legacy_toml(path: Path) -> bool:
    with path.open("rb") as handle:
        raw = load_toml(handle)
    if not isinstance(raw, dict):
        return False
    if "env_id" in raw:
        return True
    entries = raw.get("eval", [])
    return isinstance(entries, list) and any(
        isinstance(entry, dict) and ("env_id" in entry or "id" in entry)
        for entry in entries
    )


def _normalize_argv(argv: list[str]) -> list[str]:
    if argv[:1] == ["@"] and len(argv) > 1:
        return [argv[1], *argv[2:]]
    if argv and argv[0].startswith("@") and len(argv[0]) > 1:
        return [argv[0][1:], *argv[1:]]
    for index, arg in enumerate(argv):
        if arg in ("--id", "--taskset.id") and index + 1 < len(argv):
            return [argv[index + 1], *argv[:index], *argv[index + 2 :]]
        for prefix in ("--id=", "--taskset.id="):
            if arg.startswith(prefix):
                return [arg.split("=", 1)[1], *argv[:index], *argv[index + 1 :]]
    return argv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval")
    parser.add_argument("env_id_or_config")
    parser.add_argument("--env-args", "-a", type=json.loads, default={})
    parser.add_argument("--env-dir-path", default=DEFAULT_ENV_DIR_PATH)
    parser.add_argument(
        "--provider", "-p", choices=list(PROVIDER_CONFIGS), default=None
    )
    parser.add_argument("--endpoints-path", "-e", default=DEFAULT_ENDPOINTS_PATH)
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument(
        "--api-client-type",
        choices=[
            "openai_completions",
            "openai_chat_completions",
            "openai_chat_completions_token",
            "openai_responses",
            "renderer",
            "anthropic_messages",
            "nemorl_chat_completions",
        ],
        default=None,
    )
    parser.add_argument("--api-key-var", "-k", default=None)
    parser.add_argument("--api-base-url", "-b", default=None)
    parser.add_argument("--header", action="append", default=None)
    parser.add_argument("--header-from-state", action="append", default=None)
    parser.add_argument("--num-examples", "-n", type=int, default=None)
    parser.add_argument("--rollouts-per-example", "-r", type=int, default=None)
    parser.add_argument("--shuffle", "-s", action="store_true", default=False)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument(
        "--max-concurrent", "-c", type=int, default=DEFAULT_MAX_CONCURRENT
    )
    parser.add_argument("--max-tokens", "-t", type=int, default=None)
    parser.add_argument("--temperature", "-T", type=float, default=None)
    parser.add_argument("--sampling-args", "-S", type=json.loads, default=None)
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument(
        "--no-interleave-scoring", "-N", action="store_true", default=False
    )
    parser.add_argument(
        "--state-columns",
        "-C",
        type=lambda text: [part.strip() for part in text.split(",")],
        default=[],
    )
    parser.add_argument(
        "--save-results", action="store_true", default=True, help=argparse.SUPPRESS
    )
    parser.add_argument("--resume", "-R", nargs="?", const=True, default=None)
    parser.add_argument(
        "--independent-scoring", "-i", action="store_true", default=False
    )
    parser.add_argument("--save-to-hf-hub", "-H", action="store_true", default=False)
    parser.add_argument("--hf-hub-dataset-name", "-D", default="")
    parser.add_argument("--extra-env-kwargs", "-x", type=json.loads, default={})
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--fullscreen", "-f", action="store_true", default=False)
    parser.add_argument("--disable-tui", "-d", action="store_true", default=False)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--disable-env-server", action="store_true", default=False)
    parser.add_argument("--num-workers", "-w", default="auto")
    parser.add_argument(
        "--abbreviated-summary", "-A", action="store_true", default=False
    )
    parser.add_argument("--heartbeat-url", default=None)
    return parser


def _parse_args(argv: list[str]) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _eval_run_config(args: argparse.Namespace) -> EvalRunConfig:
    if args.env_id_or_config.endswith(".toml"):
        path = Path(args.env_id_or_config)
        if not path.is_file():
            raise FileNotFoundError(f"TOML config file not found: {path}")
        raw_eval_configs = load_toml_config(path)
    else:
        raw_eval_configs = [{"env_id": args.env_id_or_config, **vars(args)}]
    return EvalRunConfig(
        evals=[_build_eval_config(raw) for raw in raw_eval_configs],
        heartbeat_url=args.heartbeat_url,
    )


def _build_eval_config(raw: dict[str, Any]) -> LegacyEvalConfig:
    env_id = raw["env_id"]
    name = raw.get("name")
    if name is not None and (not isinstance(name, str) or not name):
        raise ValueError("'name' must be a non-empty string when provided.")

    env_defaults = _env_eval_defaults(env_id)
    raw_num_examples = raw.get("num_examples")
    raw_rollouts = raw.get("rollouts_per_example")
    num_examples = (
        raw_num_examples
        if raw_num_examples is not None
        else env_defaults.get("num_examples", DEFAULT_NUM_EXAMPLES)
    )
    rollouts_per_example = (
        raw_rollouts
        if raw_rollouts is not None
        else env_defaults.get("rollouts_per_example", DEFAULT_ROLLOUTS_PER_EXAMPLE)
    )

    shuffle = bool(raw.get("shuffle", False))
    shuffle_seed = raw.get("shuffle_seed")
    if shuffle and shuffle_seed is None:
        shuffle_seed = 0
    if not shuffle:
        shuffle_seed = None

    endpoint_id = raw.get("endpoint_id")
    raw_model = raw.get("model")
    if endpoint_id is not None and raw_model is not None:
        raise ValueError("Cannot set both 'endpoint_id' and 'model' in eval config.")
    if endpoint_id is not None and not isinstance(endpoint_id, str):
        raise ValueError("'endpoint_id' must be a string when provided.")

    endpoints_path = raw.get("endpoints_path", DEFAULT_ENDPOINTS_PATH)
    endpoints_file = resolve_endpoints_file(str(endpoints_path))
    if endpoint_id is not None and (
        endpoints_file is None or endpoints_file.suffix != ".toml"
    ):
        raise ValueError("'endpoint_id' is only supported with endpoints.toml.")

    model, client_config, resolved_endpoint = _client_config(
        raw, endpoint_id, raw_model
    )

    resume_arg = (
        raw.get("resume_path") if raw.get("resume") is None else raw.get("resume")
    )
    resume_path = _resolve_resume(
        resume_arg,
        env_id=env_id,
        model=model,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        env_dir_path=raw.get("env_dir_path", DEFAULT_ENV_DIR_PATH),
        output_dir=raw.get("output_dir"),
        name=name,
    )

    extra_env_kwargs = dict(raw.get("extra_env_kwargs", {}))
    if raw.get("timeout") is not None:
        extra_env_kwargs["timeout_seconds"] = raw["timeout"]

    return LegacyEvalConfig(
        env_id=env_id,
        name=name,
        env_args=dict(raw.get("env_args", {})),
        env_dir_path=raw.get("env_dir_path", DEFAULT_ENV_DIR_PATH),
        output_dir=raw.get("output_dir"),
        extra_env_kwargs=extra_env_kwargs,
        endpoint_id=resolved_endpoint,
        model=model,
        client_config=client_config,
        sampling_args=_sampling_args(raw),
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        max_concurrent=raw.get("max_concurrent", DEFAULT_MAX_CONCURRENT),
        max_retries=raw.get("max_retries", 3),
        num_workers=raw.get("num_workers", "auto"),
        disable_env_server=raw.get("disable_env_server", False),
        verbose=raw.get("verbose", False),
        disable_tui=raw.get("disable_tui", False),
        state_columns=raw.get("state_columns", []),
        save_results=True,
        resume_path=resume_path,
        independent_scoring=raw.get("independent_scoring", False),
        save_to_hf_hub=raw.get("save_to_hf_hub", False),
        hf_hub_dataset_name=raw.get("hf_hub_dataset_name", ""),
    )


def _client_config(
    raw: dict[str, Any], endpoint_id: str | None, raw_model: str | None
) -> tuple[str, ClientConfig, str | None]:
    api_key_override = raw.get("api_key_var") is not None
    api_base_url_override = raw.get("api_base_url") is not None
    client_type_override = raw.get("api_client_type") is not None
    endpoints_path = raw.get("endpoints_path", DEFAULT_ENDPOINTS_PATH)
    endpoint_lookup_id = (
        endpoint_id if endpoint_id is not None else raw_model or DEFAULT_MODEL
    )
    direct_endpoint_config = (
        endpoint_id is None and api_key_override and api_base_url_override
    )
    endpoints = {} if direct_endpoint_config else load_endpoints(endpoints_path)
    endpoint_group: list[EndpointConfig] | None = None
    resolved_endpoint_id: str | None = None

    if endpoint_lookup_id in endpoints:
        endpoint_group = endpoints[endpoint_lookup_id]
        resolved_endpoint_id = endpoint_lookup_id
        endpoint = endpoint_group[0]
        api_key_var = endpoint.api_key_var
        api_base_url = endpoint.base_url
        client_type = endpoint.api_client_type or DEFAULT_CLIENT_TYPE
        model = endpoint.model
        endpoint_models = {entry.model for entry in endpoint_group}
        if len(endpoint_models) > 1:
            raise ValueError(
                f"Endpoint alias {endpoint_lookup_id!r} maps to multiple model ids."
            )
        if raw.get("provider") is not None:
            provider_config = PROVIDER_CONFIGS[raw["provider"]]
            api_key_var = provider_config["key"]
            api_base_url = provider_config["url"]
            client_type = provider_config.get("client_type", client_type)
    else:
        if endpoint_id is not None:
            raise ValueError(
                f"Endpoint id {endpoint_id!r} not found in {endpoints_path}"
            )
        provider_config = PROVIDER_CONFIGS[raw.get("provider") or DEFAULT_PROVIDER]
        model = raw_model or DEFAULT_MODEL
        api_key_var = provider_config["key"]
        api_base_url = provider_config["url"]
        client_type = provider_config.get("client_type", DEFAULT_CLIENT_TYPE)

    if api_key_override:
        api_key_var = raw["api_key_var"]
    if api_base_url_override:
        api_base_url = raw["api_base_url"]
    if client_type_override:
        client_type = raw["api_client_type"]
    if isinstance(api_base_url, list):
        raise ValueError("api_base_url lists are no longer supported.")

    eval_headers = _extra_headers(raw.get("headers"), raw.get("header"))
    headers_from_state = {"X-Session-ID": "trajectory_id"}
    headers_from_state.update(
        _extra_headers(raw.get("headers_from_state"), raw.get("header_from_state"))
    )
    registry_headers = dict(endpoint_group[0].extra_headers) if endpoint_group else {}

    endpoint_configs: list[EndpointClientConfig] = []
    if (
        endpoint_group is not None
        and not api_base_url_override
        and raw.get("provider") is None
        and len(endpoint_group) > 1
    ):
        endpoint_configs = [
            EndpointClientConfig(
                api_key_var=api_key_var if api_key_override else endpoint.api_key_var,
                api_base_url=endpoint.base_url,
                extra_headers={**dict(endpoint.extra_headers), **eval_headers},
            )
            for endpoint in endpoint_group
        ]

    return (
        model,
        ClientConfig(
            client_type=cast(ClientType, client_type),
            api_key_var=api_key_var,
            api_base_url=api_base_url,
            endpoint_configs=endpoint_configs,
            extra_headers={**registry_headers, **eval_headers},
            extra_headers_from_state=headers_from_state,
        ),
        resolved_endpoint_id,
    )


def _extra_headers(
    headers: dict[str, str] | list[str] | None,
    entries: list[str] | None,
) -> dict[str, str]:
    if entries is not None and not isinstance(entries, list):
        raise ValueError("'header' must be a list of 'Name: Value' strings")
    values = list(entries or [])
    if isinstance(headers, list):
        values += headers
        headers = None
    merged = _validate_extra_headers_value(headers) if headers is not None else {}
    for value in values:
        if not isinstance(value, str) or ":" not in value:
            raise ValueError(f"--header must be 'Name: Value', got: {value!r}")
        key, item = value.split(":", 1)
        if not key.strip():
            raise ValueError("--header name cannot be empty")
        merged[key.strip()] = item.strip()
    return merged


def _sampling_args(raw: dict[str, Any]) -> dict[str, Any]:
    sampling = dict(raw.get("sampling_args") or {})
    sampling.setdefault("max_tokens", raw.get("max_tokens"))
    if raw.get("temperature") is not None:
        sampling.setdefault("temperature", raw["temperature"])
    return sampling


def _resolve_resume(
    resume_arg: object,
    *,
    env_id: str,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    shuffle: bool,
    shuffle_seed: int | None,
    env_dir_path: str,
    output_dir: str | None,
    name: str | None,
) -> Path | None:
    if isinstance(resume_arg, str):
        resume_path = Path(resume_arg)
        if not is_valid_eval_results_path(resume_path):
            raise ValueError(
                f"Resume path {resume_path} is not a valid eval results path"
            )
        return resume_path
    if resume_arg is True:
        return find_latest_incomplete_eval_results_path(
            env_id=env_id,
            model=model,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            env_dir_path=env_dir_path,
            output_dir=output_dir,
            name=name,
        )
    if resume_arg in (None, False):
        return None
    raise ValueError(f"Invalid value for --resume: {resume_arg!r}")


def _env_eval_defaults(env_id: str) -> dict[str, Any]:
    module_name = env_id.replace("-", "_").split("/")[-1]
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {}
    if spec.submodule_search_locations:
        base_dir = Path(next(iter(spec.submodule_search_locations)))
    elif spec.origin:
        base_dir = Path(spec.origin).parent
    else:
        return {}
    pyproject = base_dir / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    with pyproject.open("rb") as handle:
        raw = load_toml(handle)
    eval_config = raw.get("tool", {}).get("verifiers", {}).get("eval", {})
    return {
        key: eval_config[key]
        for key in ("num_examples", "rollouts_per_example")
        if key in eval_config
    }
