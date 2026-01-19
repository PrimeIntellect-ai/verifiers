import argparse
import asyncio
import importlib.resources
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # type: ignore[unresolved-import]
except ImportError:
    import tomli as tomllib  # type: ignore[unresolved-import]

from verifiers import setup_logging
from verifiers.types import (
    ClientConfig,
    EvalConfig,
    EvalEnvConfig,
    EvalModelConfig,
    EvalRunConfig,
)
from verifiers.utils.eval_utils import (
    is_toml_config,
    load_endpoints,
    load_toml_config,
    run_evaluations,
)

logger = logging.getLogger(__name__)

DEFAULT_NUM_EXAMPLES = 5
DEFAULT_ROLLOUTS_PER_EXAMPLE = 3
DEFAULT_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"


def get_env_eval_defaults(env_id: str) -> Dict[str, Any]:
    """Get eval config defaults from environment package's pyproject.toml.

    Returns dict with 'num_examples' and 'rollouts_per_example' keys if found,
    otherwise returns empty dict. All errors are silently handled.
    """
    defaults: Dict[str, Any] = {}
    module_name = env_id.replace("-", "_").split("/")[-1]

    try:
        # read pyproject.toml from installed package
        package_ref = importlib.resources.files(module_name)
        pyproject_file = package_ref / "pyproject.toml"

        if not pyproject_file.is_file():
            logger.debug(f"pyproject.toml not found in installed package {module_name}")
            return defaults

        with pyproject_file.open("rb") as f:
            pyproject_data = tomllib.load(f)

        # Extract [tool.verifiers.eval] section
        eval_config = (
            pyproject_data.get("tool", {}).get("verifiers", {}).get("eval", {})
        )

        for key in (
            "env_args",
            "num_examples",
            "rollouts_per_example",
            "interleave_scoring",
            "state_columns",
            "extra_env_kwargs",
        ):
            if key in eval_config:
                defaults[key] = eval_config[key]

        if defaults:
            logger.debug(
                f"Loaded eval defaults from {module_name} pyproject.toml: {defaults}"
            )
    except ModuleNotFoundError:
        logger.debug(f"Package {module_name} not installed")
    except Exception as e:
        logger.debug(
            f"Could not load eval defaults from {module_name} pyproject.toml: {e}"
        )

    return defaults


def merge_nested_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_model_overrides(raw_model: dict | None) -> dict:
    if not raw_model:
        return {}
    model_overrides = dict(raw_model)
    client_config = dict(model_overrides.pop("client_config", {}))
    for key in ("api_key_var", "api_base_url", "extra_headers"):
        if key in model_overrides:
            client_config[key] = model_overrides.pop(key)
    if client_config:
        model_overrides["client_config"] = client_config
    return model_overrides


def resolve_client_config(
    model_name: str,
    endpoints: dict,
    api_key_var_override: str | None,
    api_base_url_override: str | None,
    extra_headers: Dict[str, str],
) -> tuple[str, ClientConfig]:
    api_key_override = api_key_var_override is not None
    api_base_url_override = api_base_url_override is not None

    if model_name in endpoints:
        endpoint = endpoints[model_name]
        api_key_var = api_key_var_override if api_key_override else endpoint["key"]
        api_base_url = (
            api_base_url_override if api_base_url_override else endpoint["url"]
        )
        resolved_model = endpoint["model"]
        if api_key_override or api_base_url_override:
            logger.debug(
                "Using endpoint registry for model '%s' with overrides (key: %s, url: %s)",
                resolved_model,
                "override" if api_key_override else "registry",
                "override" if api_base_url_override else "registry",
            )
        else:
            logger.debug(
                "Using endpoint configuration for model '%s' from registry",
                resolved_model,
            )
    else:
        logger.debug(
            "Model '%s' not found in endpoint registry, using defaults",
            model_name,
        )
        api_key_var = api_key_var_override or DEFAULT_API_KEY_VAR
        api_base_url = api_base_url_override or DEFAULT_API_BASE_URL
        resolved_model = model_name

    return resolved_model, ClientConfig(
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        extra_headers=extra_headers,
    )


def build_model_config(
    model_overrides: dict,
    model_defaults: dict,
    cli_defaults: dict,
    endpoints: dict,
    cli_api_key_var: str | None,
    cli_api_base_url: str | None,
    cli_headers: Dict[str, str],
) -> EvalModelConfig:
    normalized_defaults = normalize_model_overrides(model_defaults)
    normalized_overrides = normalize_model_overrides(model_overrides)
    model_name = (
        normalized_overrides.get("model")
        or normalized_defaults.get("model")
        or cli_defaults["model"]
    )
    merged_headers = merge_nested_dicts(
        cli_headers,
        merge_nested_dicts(
            normalized_defaults.get("client_config", {}).get("extra_headers", {}),
            normalized_overrides.get("client_config", {}).get("extra_headers", {}),
        ),
    )
    api_key_var_override = normalized_overrides.get("client_config", {}).get(
        "api_key_var",
        normalized_defaults.get("client_config", {}).get(
            "api_key_var", cli_api_key_var
        ),
    )
    api_base_url_override = normalized_overrides.get("client_config", {}).get(
        "api_base_url",
        normalized_defaults.get("client_config", {}).get(
            "api_base_url", cli_api_base_url
        ),
    )
    resolved_model, client_config = resolve_client_config(
        model_name,
        endpoints,
        api_key_var_override,
        api_base_url_override,
        merged_headers,
    )

    sampling_args = merge_nested_dicts(
        cli_defaults["sampling_args"],
        normalized_defaults.get("sampling_args", {}),
    )
    sampling_args = merge_nested_dicts(
        sampling_args,
        normalized_overrides.get("sampling_args", {}),
    )

    return EvalModelConfig(
        model=resolved_model,
        client_config=client_config,
        sampling_args=sampling_args,
        max_concurrent=normalized_overrides.get(
            "max_concurrent",
            normalized_defaults.get("max_concurrent", cli_defaults["max_concurrent"]),
        ),
        max_concurrent_generation=normalized_overrides.get(
            "max_concurrent_generation",
            normalized_defaults.get(
                "max_concurrent_generation", cli_defaults["max_concurrent_generation"]
            ),
        ),
        max_concurrent_scoring=normalized_overrides.get(
            "max_concurrent_scoring",
            normalized_defaults.get(
                "max_concurrent_scoring", cli_defaults["max_concurrent_scoring"]
            ),
        ),
    )


def build_env_config(
    env_id: str,
    env_overrides: dict,
    env_defaults: dict,
    cli_defaults: dict,
    pyproject_defaults: dict,
) -> EvalEnvConfig:
    env_overrides = dict(env_overrides)
    env_overrides.pop("env_id", None)
    merged_defaults = merge_nested_dicts(
        {
            "num_examples": DEFAULT_NUM_EXAMPLES,
            "rollouts_per_example": DEFAULT_ROLLOUTS_PER_EXAMPLE,
            "interleave_scoring": True,
            "env_args": {},
            "state_columns": None,
            "extra_env_kwargs": {},
        },
        pyproject_defaults,
    )
    merged_defaults = merge_nested_dicts(merged_defaults, cli_defaults)
    merged_defaults = merge_nested_dicts(merged_defaults, env_defaults)
    merged = merge_nested_dicts(merged_defaults, env_overrides)

    return EvalEnvConfig(
        env_id=env_id,
        env_args=merged.get("env_args", {}),
        num_examples=merged["num_examples"],
        rollouts_per_example=merged["rollouts_per_example"],
        interleave_scoring=merged.get("interleave_scoring", True),
        state_columns=merged.get("state_columns"),
        extra_env_kwargs=merged.get("extra_env_kwargs", {}),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_id_or_path",
        type=str,
        default="gsm8k",
        help="Environment module name(s) (comma-separated) or path to TOML config.",
    )
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default=None,
        help='Environment module arguments as JSON object (e.g., \'{"key": "value", "num": 42}\')',
    )
    parser.add_argument(
        "--env-dir-path",
        "-p",
        type=str,
        default="./environments",
        help="Path to environments directory",
    )
    parser.add_argument(
        "--endpoints-path",
        "-e",
        type=str,
        default="./configs/endpoints.py",
        help="Path to API endpoints registry",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openai/gpt-4.1-mini",
        help="Name of model to evaluate",
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        type=str,
        default=None,
        help=(
            "Environment variable name for API key "
            "(defaults to PRIME_API_KEY when not set and not in registry)"
        ),
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        type=str,
        default=None,
        help=(
            "Base URL for API "
            "(defaults to https://api.pinference.ai/api/v1 when not set and not in registry)"
        ),
    )
    parser.add_argument(
        "--header",
        action="append",
        default=None,
        help="Extra HTTP header to pass to inference API. 'Name: Value'. Repeatable.",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=None,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--rollouts-per-example",
        "-r",
        type=int,
        default=None,
        help="Number of rollouts per example",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=32,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--max-concurrent-generation",
        type=int,
        default=None,
        help="Maximum number of concurrent generation requests",
    )
    parser.add_argument(
        "--max-concurrent-scoring",
        type=int,
        default=None,
        help="Maximum number of concurrent scoring requests",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (unset to use model default)",
    )
    parser.add_argument(
        "--temperature", "-T", type=float, default=None, help="Temperature for sampling"
    )
    parser.add_argument(
        "--sampling-args",
        "-S",
        type=json.loads,
        default=None,
        help=(
            "Sampling arguments as JSON object. Keys here override --max-tokens/--temperature. "
            'Example: \'{"enable_thinking": false, "max_tokens": 256}\''
        ),
    )
    parser.add_argument(
        "--verbose", "-v", default=False, action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--no-interleave-scoring",
        "-N",
        default=None,
        action="store_true",
        help="Disable interleaving of scoring",
    )
    parser.add_argument(
        "--state-columns",
        "-C",
        type=lambda t: [s.strip() for s in t.split(",")],
        default=None,
        help="Comma-separated list of state columns to save (e.g., 'turn,timing')",
    )
    parser.add_argument(
        "--save-results",
        "-s",
        default=None,
        action="store_true",
        help="Save results to disk",
    )
    # save every n rollouts
    parser.add_argument(
        "--save-every",
        "-f",
        type=int,
        default=None,
        help="Save dataset every n rollouts",
    )
    parser.add_argument(
        "--independent-scoring",
        "-R",
        default=None,
        action="store_true",
        help="Score each rollout individually instead of scoring by group",
    )
    parser.add_argument(
        "--save-to-hf-hub",
        "-H",
        default=None,
        action="store_true",
        help="Save dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-hub-dataset-name",
        "-D",
        type=str,
        default=None,
        help="Name of dataset to save to Hugging Face Hub",
    )
    parser.add_argument(
        "--extra-env-kwargs",
        "-x",
        type=json.loads,
        default=None,
        help='Extra environment as JSON object (e.g., \'{"key": "value", "num": 42}\'). Passed to environment constructor.',
    )
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else os.getenv("VF_LOG_LEVEL", "INFO"))

    endpoints = load_endpoints(args.endpoints_path)

    # merge sampling args with precedence to JSON payload over explicit flags
    merged_sampling_args: dict = {}
    if args.sampling_args is not None:
        merged_sampling_args.update(args.sampling_args)
    if "max_tokens" not in merged_sampling_args:
        merged_sampling_args["max_tokens"] = args.max_tokens
    if args.temperature is not None and "temperature" not in merged_sampling_args:
        merged_sampling_args["temperature"] = args.temperature

    # Build headers from repeated --header flags
    merged_headers: Dict[str, str] = {}
    for h in args.header or []:
        if ":" not in h:
            raise ValueError(f"--header must be 'Name: Value', got: {h!r}")
        k, v = h.split(":", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError("--header name cannot be empty")
        merged_headers[k] = v

    cli_model_defaults = {
        "model": args.model,
        "sampling_args": merged_sampling_args,
        "max_concurrent": args.max_concurrent,
        "max_concurrent_generation": args.max_concurrent_generation,
        "max_concurrent_scoring": args.max_concurrent_scoring,
    }

    interleave_override = None
    if args.no_interleave_scoring:
        interleave_override = False
    if args.independent_scoring:
        interleave_override = False

    cli_env_defaults = {
        "env_args": args.env_args if args.env_args is not None else None,
        "num_examples": args.num_examples,
        "rollouts_per_example": args.rollouts_per_example,
        "interleave_scoring": interleave_override,
        "state_columns": args.state_columns,
        "extra_env_kwargs": args.extra_env_kwargs,
    }

    cli_env_defaults = {k: v for k, v in cli_env_defaults.items() if v is not None}

    if is_toml_config(args.env_id_or_path):
        path = Path(args.env_id_or_path)
        toml_config = load_toml_config(path)
        raw_eval_list = toml_config["evals"]
        toml_env_defaults = toml_config["env_defaults"]
        toml_model_defaults = toml_config["model_defaults"]
        toml_save_defaults = toml_config["save_defaults"]
    elif "," in args.env_id_or_path:
        env_ids = [
            env_id.strip()
            for env_id in args.env_id_or_path.split(",")
            if env_id.strip()
        ]
        if not env_ids:
            raise ValueError(
                f"No valid env_ids found in comma-separated list {args.env_id_or_path!r}"
            )
        raw_eval_list = [{"env": {"env_id": env_id}} for env_id in env_ids]
        toml_env_defaults = {}
        toml_model_defaults = {}
        toml_save_defaults = {}
    else:
        raw_eval_list = [{"env": {"env_id": args.env_id_or_path}}]
        toml_env_defaults = {}
        toml_model_defaults = {}
        toml_save_defaults = {}

    eval_configs: list[EvalConfig] = []
    for raw_eval in raw_eval_list:
        raw_env = raw_eval.get("env", {})
        if "env_id" in raw_eval:
            raw_env = {**raw_env, "env_id": raw_eval["env_id"]}
        env_id = raw_env["env_id"]
        raw_model = raw_eval.get("model", {})

        env_config = build_env_config(
            env_id,
            raw_env,
            toml_env_defaults,
            cli_env_defaults,
            get_env_eval_defaults(env_id),
        )

        model_config = build_model_config(
            raw_model,
            toml_model_defaults,
            cli_model_defaults,
            endpoints,
            args.api_key_var,
            args.api_base_url,
            merged_headers,
        )

        eval_config = EvalConfig(env=env_config, model=model_config)
        eval_configs.append(eval_config)
        logger.debug(f"Evaluation config: {eval_config.model_dump_json(indent=2)}")

    save_results = (
        args.save_results
        if args.save_results is not None
        else toml_save_defaults.get("save_results", False)
    )
    save_every = (
        args.save_every
        if args.save_every is not None
        else toml_save_defaults.get("save_every", -1)
    )
    save_to_hf_hub = (
        args.save_to_hf_hub
        if args.save_to_hf_hub is not None
        else toml_save_defaults.get("save_to_hf_hub", False)
    )
    hf_hub_dataset_name = (
        args.hf_hub_dataset_name
        if args.hf_hub_dataset_name is not None
        else toml_save_defaults.get("hf_hub_dataset_name")
    )

    run_config = EvalRunConfig(
        evals=eval_configs,
        env_dir_path=args.env_dir_path,
        save_results=save_results,
        save_every=save_every,
        save_to_hf_hub=save_to_hf_hub,
        hf_hub_dataset_name=hf_hub_dataset_name,
    )
    logger.debug(f"Evaluation run config: {run_config.model_dump_json(indent=2)}")

    asyncio.run(run_evaluations(run_config))


if __name__ == "__main__":
    main()
