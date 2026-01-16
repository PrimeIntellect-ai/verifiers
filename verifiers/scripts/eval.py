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
from verifiers.types import ClientConfig, EvalConfig, EvalRunConfig
from verifiers.utils.eval_utils import (
    is_toml_config,
    load_endpoints,
    load_toml_config,
    run_multi_evaluation,
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

        if "num_examples" in eval_config:
            defaults["num_examples"] = eval_config["num_examples"]
        if "rollouts_per_example" in eval_config:
            defaults["rollouts_per_example"] = eval_config["rollouts_per_example"]

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
        default={},
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
        default=False,
        action="store_true",
        help="Disable interleaving of scoring",
    )
    parser.add_argument(
        "--state-columns",
        "-C",
        type=lambda t: [s.strip() for s in t.split(",")],
        default=[],
        help="Comma-separated list of state columns to save (e.g., 'turn,timing')",
    )
    parser.add_argument(
        "--save-results",
        "-s",
        default=False,
        action="store_true",
        help="Save results to disk",
    )
    # save every n rollouts
    parser.add_argument(
        "--save-every",
        "-f",
        type=int,
        default=-1,
        help="Save dataset every n rollouts",
    )
    parser.add_argument(
        "--independent-scoring",
        "-R",
        default=False,
        action="store_true",
        help="Score each rollout individually instead of scoring by group",
    )
    parser.add_argument(
        "--save-to-hf-hub",
        "-H",
        default=False,
        action="store_true",
        help="Save dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-hub-dataset-name",
        "-D",
        type=str,
        default="",
        help="Name of dataset to save to Hugging Face Hub",
    )
    parser.add_argument(
        "--extra-env-kwargs",
        "-x",
        type=json.loads,
        default={},
        help='Extra environment as JSON object (e.g., \'{"key": "value", "num": 42}\'). Passed to environment constructor.',
    )
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else os.getenv("VF_LOG_LEVEL", "INFO"))

    # resolve env_id_or_path: TOML config > comma-separated list > single env ID
    if is_toml_config(args.env_id_or_path):
        # single/multi-env eval via single TOML config
        path = Path(args.env_id_or_path)
        raw_multi_env_config = load_toml_config(path)
    elif "," in args.env_id_or_path:
        # multi-env eval via comma-separated list
        env_ids = [
            env_id.strip()
            for env_id in args.env_id_or_path.split(",")
            if env_id.strip()
        ]
        if not env_ids:
            raise ValueError(
                f"No valid env_ids found in comma-separated list {args.env_id_or_path!r}"
            )
        raw_multi_env_config = [{"env_id": env_id} for env_id in env_ids]
    else:
        # single-eval env
        raw_multi_env_config = [{"env_id": args.env_id_or_path}]

    def resolve_eval_config(raw_env_config: dict) -> EvalConfig:
        """Resolve per-env eval config. TOML > CLI > Env Defaults > Global Defaults"""
        assert "env_id" in raw_env_config
        env_id = raw_env_config["env_id"]
        env_defaults = get_env_eval_defaults(env_id)
        num_examples = raw_env_config.get("num_examples", args.num_examples)
        rollouts_per_example = raw_env_config.get(
            "rollouts_per_example", args.rollouts_per_example
        )

        if num_examples is None:
            num_examples = env_defaults.get("num_examples", DEFAULT_NUM_EXAMPLES)
        if rollouts_per_example is None:
            rollouts_per_example = env_defaults.get(
                "rollouts_per_example", DEFAULT_ROLLOUTS_PER_EXAMPLE
            )

        if raw_env_config.get("num_examples") is None and args.num_examples is None:
            source = (
                "pyproject.toml" if "num_examples" in env_defaults else "global default"
            )
            logger.debug(f"Using num_examples={num_examples} from {source}")
        if (
            raw_env_config.get("rollouts_per_example") is None
            and args.rollouts_per_example is None
        ):
            source = (
                "pyproject.toml"
                if "rollouts_per_example" in env_defaults
                else "global default"
            )
            logger.debug(
                f"Using rollouts_per_example={rollouts_per_example} from {source}"
            )

        env_args = raw_env_config.get("env_args", args.env_args) or {}

        return EvalConfig(
            env_id=env_id,
            env_args=env_args,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
        )

    eval_configs: list[EvalConfig] = []
    for raw_env_config in raw_multi_env_config:
        eval_config = resolve_eval_config(raw_env_config)
        eval_configs.append(eval_config)
        logger.debug(f"Evaluation config: {eval_config.model_dump_json(indent=2)}")

    endpoints = load_endpoints(args.endpoints_path)
    api_key_override = args.api_key_var is not None
    api_base_url_override = args.api_base_url is not None

    model = args.model
    if args.model in endpoints:
        endpoint = endpoints[args.model]
        api_key_var = args.api_key_var if api_key_override else endpoint["key"]
        api_base_url = args.api_base_url if api_base_url_override else endpoint["url"]
        model = endpoint["model"]
        if api_key_override or api_base_url_override:
            logger.debug(
                "Using endpoint registry for model '%s' with CLI overrides (key: %s, url: %s)",
                model,
                "cli" if api_key_override else "registry",
                "cli" if api_base_url_override else "registry",
            )
        else:
            logger.debug(
                "Using endpoint configuration for model '%s' from registry",
                model,
            )
    else:
        logger.debug(
            "Model '%s' not found in endpoint registry, using command-line arguments",
            model,
        )
        api_key_var = args.api_key_var if api_key_override else DEFAULT_API_KEY_VAR
        api_base_url = (
            args.api_base_url if api_base_url_override else DEFAULT_API_BASE_URL
        )

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

    client_config = ClientConfig(
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        extra_headers=merged_headers,
    )

    run_config = EvalRunConfig(
        env=eval_configs,
        env_dir_path=args.env_dir_path,
        model=model,
        client_config=client_config,
        sampling_args=merged_sampling_args,
        max_concurrent=args.max_concurrent,
        max_concurrent_generation=args.max_concurrent_generation,
        max_concurrent_scoring=args.max_concurrent_scoring,
        independent_scoring=args.independent_scoring,
        extra_env_kwargs=args.extra_env_kwargs,
        verbose=args.verbose,
        state_columns=args.state_columns,
        save_results=args.save_results,
        save_every=args.save_every,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=args.hf_hub_dataset_name,
    )
    logger.debug(f"Evaluation run config: {run_config.model_dump_json(indent=2)}")

    asyncio.run(run_multi_evaluation(run_config))


if __name__ == "__main__":
    main()
