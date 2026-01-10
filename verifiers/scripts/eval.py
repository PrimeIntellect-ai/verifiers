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
from verifiers.types import ClientConfig, EvalConfig
from verifiers.utils.eval_utils import load_endpoints, run_evaluation

logger = logging.getLogger(__name__)

DEFAULT_NUM_EXAMPLES = 5
DEFAULT_ROLLOUTS_PER_EXAMPLE = 3
DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_PRIME_TEAM_HEADER = "X-Prime-Team-ID"
LEGACY_PRIME_TEAM_HEADERS = {"X-PI-TEAM-ID", "X-Prime-Team-ID"}
DEFAULT_PRIME_CONFIG_PATH = Path.home() / ".prime" / "config.json"


def _load_prime_config() -> dict[str, Any] | None:
    config_path = Path(
        os.getenv("PRIME_CONFIG_PATH", str(DEFAULT_PRIME_CONFIG_PATH))
    ).expanduser()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug(f"Failed to read Prime config at {config_path}: {exc}")
        return None
    if not isinstance(config, dict):
        logger.debug(f"Prime config at {config_path} is not a JSON object")
        return None
    return config


def _should_use_prime_team_header(
    api_base_url: str,
    api_key_var: str,
    prime_config: dict[str, Any],
) -> bool:
    if api_key_var == "PRIME_API_KEY":
        return True
    if "pinference.ai" in api_base_url:
        return True
    inference_url = prime_config.get("inference_url")
    if isinstance(inference_url, str) and inference_url:
        return api_base_url.rstrip("/") == inference_url.rstrip("/")
    return False


def _get_prime_inference_url() -> str | None:
    prime_config = _load_prime_config()
    if not prime_config:
        return None
    inference_url = prime_config.get("inference_url")
    if isinstance(inference_url, str) and inference_url.strip():
        return inference_url.strip()
    return None


def _get_prime_team_id(api_base_url: str, api_key_var: str) -> str | None:
    prime_config = _load_prime_config()
    if not prime_config:
        return None
    if not _should_use_prime_team_header(api_base_url, api_key_var, prime_config):
        return None
    team_id = prime_config.get("team_id")
    if isinstance(team_id, str) and team_id.strip():
        return team_id.strip()
    return None


def _merge_headers(header_args: list[str] | None, api_base_url: str, api_key_var: str):
    merged_headers: Dict[str, str] = {}
    merged_header_keys = set()
    for h in header_args or []:
        if ":" not in h:
            raise ValueError(f"--header must be 'Name: Value', got: {h!r}")
        k, v = h.split(":", 1)
        k, v = k.strip(), v.strip()
        if not k:
            raise ValueError("--header name cannot be empty")
        merged_headers[k] = v
        merged_header_keys.add(k.lower())

    team_header = os.getenv("PRIME_TEAM_ID_HEADER", DEFAULT_PRIME_TEAM_HEADER)
    known_team_headers = {team_header, *LEGACY_PRIME_TEAM_HEADERS}
    if not any(h.lower() in merged_header_keys for h in known_team_headers):
        team_id = _get_prime_team_id(api_base_url, api_key_var)
        if team_id:
            merged_headers[team_header] = team_id
            logger.debug(f"Using Prime team header {team_header} from local config")

    return merged_headers


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
        "env_id", type=str, default="gsm8k", help="Environment module name"
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
        default="gpt-4.1-mini",
        help="Name of model to evaluate",
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        type=str,
        default="PRIME_API_KEY",
        help="Environment variable name for API key",
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        type=str,
        default=None,
        help=f"Base URL for API (default: {DEFAULT_API_BASE_URL})",
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

    # apply defaults: CLI args take precedence, then env defaults, then global defaults
    env_defaults = get_env_eval_defaults(args.env_id)
    num_examples = (
        args.num_examples
        if args.num_examples is not None
        else env_defaults.get("num_examples", DEFAULT_NUM_EXAMPLES)
    )
    rollouts_per_example = (
        args.rollouts_per_example
        if args.rollouts_per_example is not None
        else env_defaults.get("rollouts_per_example", DEFAULT_ROLLOUTS_PER_EXAMPLE)
    )

    if args.num_examples is None:
        source = (
            "pyproject.toml" if "num_examples" in env_defaults else "global default"
        )
        logger.debug(f"Using num_examples={num_examples} from {source}")
    if args.rollouts_per_example is None:
        source = (
            "pyproject.toml"
            if "rollouts_per_example" in env_defaults
            else "global default"
        )
        logger.debug(f"Using rollouts_per_example={rollouts_per_example} from {source}")

    # load endpoints and get model config
    endpoints = load_endpoints(args.endpoints_path)
    if args.model in endpoints:
        api_key_var = endpoints[args.model]["key"]
        api_base_url = endpoints[args.model]["url"]
        args.model = endpoints[args.model]["model"]
        logger.debug(
            f"Using endpoint configuration for model '{args.model}' from registry"
        )
    else:
        logger.debug(
            f"Model '{args.model}' not found in endpoint registry, using command-line arguments"
        )
        api_key_var = args.api_key_var
        api_base_url = (
            args.api_base_url or _get_prime_inference_url() or DEFAULT_API_BASE_URL
        )

    # merge sampling args with precedence to JSON payload over explicit flags
    merged_sampling_args: dict = {}
    if args.sampling_args is not None:
        merged_sampling_args.update(args.sampling_args)
    if "max_tokens" not in merged_sampling_args:
        merged_sampling_args["max_tokens"] = args.max_tokens
    if args.temperature is not None and "temperature" not in merged_sampling_args:
        merged_sampling_args["temperature"] = args.temperature

    # Build headers from repeated --header flags, then augment from Prime config.
    merged_headers = _merge_headers(args.header, api_base_url, api_key_var)

    client_config = ClientConfig(
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        extra_headers=merged_headers or None,
    )

    # run evaluation
    eval_config = EvalConfig(
        # environment
        env_id=args.env_id,
        env_args=args.env_args,
        env_dir_path=args.env_dir_path,
        extra_env_kwargs=args.extra_env_kwargs,
        # evaluation
        model=args.model,
        client_config=client_config,
        sampling_args=merged_sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=args.max_concurrent,
        max_concurrent_generation=args.max_concurrent_generation,
        max_concurrent_scoring=args.max_concurrent_scoring,
        # logging
        print_results=True,
        verbose=args.verbose,
        # saving
        state_columns=args.state_columns,
        save_results=args.save_results,
        save_every=args.save_every,
        independent_scoring=args.independent_scoring,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=args.hf_hub_dataset_name,
    )
    logger.debug(f"Evaluation config: {eval_config.model_dump_json(indent=2)}")
    asyncio.run(run_evaluation(eval_config))


if __name__ == "__main__":
    main()
