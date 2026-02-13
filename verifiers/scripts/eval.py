import os

from verifiers.utils.path_utils import (
    find_latest_incomplete_eval_results_path,
    is_valid_eval_results_path,
)

# Suppress tokenizers parallelism warning (only prints when env var is unset)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import argparse
import asyncio
import importlib.resources
import json
import logging
from pathlib import Path
from typing import Any

from verifiers import setup_logging
from verifiers.types import (
    ClientConfig,
    EndpointClientConfig,
    EvalConfig,
    EvalRunConfig,
)
from verifiers.utils.eval_utils import (
    get_log_level,
    load_endpoints,
    load_toml_config,
    resolve_endpoints_file,
    run_evaluations,
    run_evaluations_tui,
)
from verifiers.utils.import_utils import load_toml
from verifiers.utils.install_utils import check_hub_env_installed

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_ENV_DIR_PATH = "./environments"
DEFAULT_ENDPOINTS_PATH = "./configs/endpoints.toml"
DEFAULT_NUM_EXAMPLES = 5
DEFAULT_ROLLOUTS_PER_EXAMPLE = 3
DEFAULT_MAX_CONCURRENT = 32
DEFAULT_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"


def get_env_eval_defaults(env_id: str) -> dict[str, Any]:
    """Get eval config defaults from environment package's pyproject.toml.

    Returns dict with 'num_examples' and 'rollouts_per_example' keys if found,
    otherwise returns empty dict. All errors are silently handled.
    """
    defaults: dict[str, Any] = {}
    module_name = env_id.replace("-", "_").split("/")[-1]

    try:
        # read pyproject.toml from installed package
        package_ref = importlib.resources.files(module_name)
        pyproject_file = package_ref / "pyproject.toml"

        if not pyproject_file.is_file():
            logger.debug(f"pyproject.toml not found in installed package {module_name}")
            return defaults

        with pyproject_file.open("rb") as f:
            pyproject_data = load_toml(f)

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
        "env_id_or_config",
        type=str,
        default="gsm8k",
        help="Environment module name or path to TOML config file.",
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
        default=DEFAULT_ENV_DIR_PATH,
        help="Path to environments directory",
    )
    parser.add_argument(
        "--endpoints-path",
        "-e",
        type=str,
        default=DEFAULT_ENDPOINTS_PATH,
        help="Path to API endpoints registry (.toml preferred, .py supported)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
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
            "Base URL for API. "
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
        default=DEFAULT_MAX_CONCURRENT,
        help="Maximum number of concurrent requests",
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
    parser.add_argument(
        "--resume",
        "-R",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help=(
            "Resume from a previous run. Optionally provide a PATH; "
            "if omitted, auto-detect the latest incomplete matching run."
        ),
    )
    parser.add_argument(
        "--independent-scoring",
        "-i",
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
    parser.add_argument(
        "--tui",
        "-u",
        default=False,
        action="store_true",
        help="Use TUI mode for live evaluation display",
    )
    parser.add_argument(
        "--debug",
        "-d",
        default=False,
        action="store_true",
        help="Disable Rich display; use normal logging and tqdm progress instead",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Max retries for transient infrastructure errors (default: 0)",
    )
    parser.add_argument(
        "--heartbeat-url",
        type=str,
        default=None,
        help="Heartbeat URL for uptime monitoring",
    )
    args = parser.parse_args()

    if args.debug:  # only set up console logging in debug mode
        setup_logging(get_log_level(args.verbose))

    # Build raw configs: both paths produce list[dict]
    if args.env_id_or_config.endswith(".toml"):
        path = Path(args.env_id_or_config)
        if not path.is_file():
            raise FileNotFoundError(
                f"TOML config file not found: {path}\nPlease check the path is correct."
            )
        raw_eval_configs = load_toml_config(path)
    else:
        # CLI path: convert args to dict
        raw_config = {"env_id": args.env_id_or_config}
        raw_config.update(vars(args))
        raw_eval_configs = [raw_config]

    def build_eval_config(raw: dict) -> EvalConfig:
        """Build EvalConfig from a raw config dict."""
        env_id = raw["env_id"]

        # Resolve num_examples and rollouts_per_example with env defaults
        env_defaults = get_env_eval_defaults(env_id)
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

        if raw_num_examples is None:
            source = (
                "pyproject.toml" if "num_examples" in env_defaults else "global default"
            )
            logger.debug(f"Using num_examples={num_examples} from {source}")
        if raw_rollouts is None:
            source = (
                "pyproject.toml"
                if "rollouts_per_example" in env_defaults
                else "global default"
            )
            logger.debug(
                f"Using rollouts_per_example={rollouts_per_example} from {source}"
            )

        # Resolve model and endpoint config
        endpoints_path = raw.get("endpoints_path", DEFAULT_ENDPOINTS_PATH)
        endpoints = load_endpoints(endpoints_path)

        raw_endpoint_id = raw.get("endpoint_id")
        raw_model_field = raw.get("model")
        if raw_endpoint_id is not None and raw_model_field is not None:
            raise ValueError(
                "Cannot set both 'endpoint_id' and 'model' in eval config; choose one."
            )
        if raw_endpoint_id is not None and not isinstance(raw_endpoint_id, str):
            raise ValueError("'endpoint_id' must be a string when provided.")
        if isinstance(raw_endpoint_id, str) and not raw_endpoint_id:
            raise ValueError("'endpoint_id' must be a non-empty string when provided.")
        resolved_endpoints_file = resolve_endpoints_file(str(endpoints_path))
        if raw_endpoint_id is not None and (
            resolved_endpoints_file is None or resolved_endpoints_file.suffix != ".toml"
        ):
            raise ValueError(
                "'endpoint_id' is only supported with TOML endpoint registries. "
                "Set endpoints_path to an endpoints.toml file."
            )

        raw_model = raw_model_field if raw_model_field is not None else DEFAULT_MODEL
        endpoint_lookup_id = (
            raw_endpoint_id if raw_endpoint_id is not None else raw_model
        )
        raw_api_key_var = raw.get("api_key_var")
        raw_api_base_url = raw.get("api_base_url")
        if isinstance(raw_api_base_url, list):
            raise ValueError(
                "api_base_url lists are no longer supported. "
                "Use endpoint_id + endpoints.toml for multi-endpoint configuration."
            )

        api_key_override = raw_api_key_var is not None
        api_base_url_override = raw_api_base_url is not None
        endpoint_group: list[dict[str, str]] | None = None
        resolved_endpoint_id: str | None = None

        if endpoint_lookup_id in endpoints:
            endpoint_group = endpoints[endpoint_lookup_id]
            resolved_endpoint_id = endpoint_lookup_id
            endpoint = endpoint_group[0]

            if api_key_override:
                api_key_var = raw_api_key_var
            else:
                api_key_var = endpoint["key"]

            if api_base_url_override:
                api_base_url = raw_api_base_url
            else:
                api_base_url = endpoint["url"]

            endpoint_models = {entry["model"] for entry in endpoint_group}
            if len(endpoint_models) > 1:
                raise ValueError(
                    f"Endpoint alias '{endpoint_lookup_id}' maps to multiple model ids {sorted(endpoint_models)}, "
                    "which is not yet supported by EvalConfig."
                )
            model = endpoint["model"]
            if api_key_override or api_base_url_override:
                logger.debug(
                    "Using endpoint registry for model '%s' with overrides (key: %s, url: %s)",
                    model,
                    "override" if api_key_override else "registry",
                    "override" if api_base_url_override else "registry",
                )
            else:
                logger.debug(
                    "Using endpoint configuration for model '%s' from registry (%d endpoint variant(s))",
                    model,
                    len(endpoint_group),
                )
        else:
            if raw_endpoint_id is not None:
                raise ValueError(
                    f"Endpoint id '{raw_endpoint_id}' not found in endpoint registry at {endpoints_path}"
                )
            logger.debug(
                "Model '%s' not found in endpoint registry, using defaults",
                raw_model,
            )
            model = raw_model
            api_key_var = raw_api_key_var if api_key_override else DEFAULT_API_KEY_VAR
            api_base_url = (
                raw_api_base_url if api_base_url_override else DEFAULT_API_BASE_URL
            )

        # Merge sampling args
        merged_sampling_args: dict = {}
        if raw.get("sampling_args") is not None:
            merged_sampling_args.update(raw["sampling_args"])
        if "max_tokens" not in merged_sampling_args:
            merged_sampling_args["max_tokens"] = raw.get("max_tokens")
        raw_temp = raw.get("temperature")
        if raw_temp is not None and "temperature" not in merged_sampling_args:
            merged_sampling_args["temperature"] = raw_temp
        # Build headers
        merged_headers: dict[str, str] = {}
        for h in raw.get("header") or []:
            if ":" not in h:
                raise ValueError(f"--header must be 'Name: Value', got: {h!r}")
            k, v = h.split(":", 1)
            k, v = k.strip(), v.strip()
            if not k:
                raise ValueError("--header name cannot be empty")
            merged_headers[k] = v

        primary_api_base_url = api_base_url
        if not isinstance(primary_api_base_url, str):
            raise ValueError("api_base_url must be a single string URL")
        assert api_key_var is not None
        resolved_api_key_var = api_key_var

        endpoint_configs: list[EndpointClientConfig] = []
        if (
            endpoint_group is not None
            and not api_base_url_override
            and len(endpoint_group) > 1
        ):
            endpoint_configs = [
                EndpointClientConfig(
                    api_key_var=(
                        resolved_api_key_var if api_key_override else endpoint["key"]
                    ),
                    api_base_url=endpoint["url"],
                    extra_headers=merged_headers,
                )
                for endpoint in endpoint_group
            ]

        assert primary_api_base_url is not None
        client_config = ClientConfig(
            api_key_var=resolved_api_key_var,
            api_base_url=primary_api_base_url,
            endpoint_configs=endpoint_configs,
            extra_headers=merged_headers,
        )

        # Backward-compatible TOML field: resume_path
        if raw.get("resume") is None and raw.get("resume_path") is not None:
            raw["resume"] = raw["resume_path"]

        # handle resume path resolution
        resume_arg = raw.get("resume")
        resume_path: Path | None = None
        if isinstance(resume_arg, str):
            resume_path = Path(resume_arg)
            if not is_valid_eval_results_path(resume_path):
                raise ValueError(
                    f"Resume path {resume_path} is not a valid evaluation results path"
                )
            logger.info(f"Resuming from explicit path: {resume_path}")
        elif resume_arg is True:
            auto_resume_path = find_latest_incomplete_eval_results_path(
                env_id=env_id,
                model=model,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
                env_dir_path=raw.get("env_dir_path", DEFAULT_ENV_DIR_PATH),
            )
            if auto_resume_path is not None:
                resume_path = auto_resume_path
                logger.info(f"Auto-resuming from: {resume_path}")
            else:
                logger.info(
                    "No matching incomplete run found for --resume; starting a new run"
                )
        elif resume_arg in (None, False):
            pass
        else:
            raise ValueError(f"Invalid value for --resume: {resume_arg!r}")

        return EvalConfig(
            env_id=env_id,
            env_args=raw.get("env_args", {}),
            env_dir_path=raw.get("env_dir_path", DEFAULT_ENV_DIR_PATH),
            extra_env_kwargs=raw.get("extra_env_kwargs", {}),
            endpoint_id=resolved_endpoint_id,
            model=model,
            client_config=client_config,
            sampling_args=merged_sampling_args,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            max_concurrent=raw.get("max_concurrent", DEFAULT_MAX_CONCURRENT),
            max_retries=raw.get("max_retries", 0),
            verbose=raw.get("verbose", False),
            debug=raw.get("debug", False),
            state_columns=raw.get("state_columns", []),
            save_results=raw.get("save_results", False),
            resume_path=resume_path,
            independent_scoring=raw.get("independent_scoring", False),
            save_to_hf_hub=raw.get("save_to_hf_hub", False),
            hf_hub_dataset_name=raw.get("hf_hub_dataset_name", ""),
        )

    # Check Hub environments are installed before running
    missing_envs = []
    for raw in raw_eval_configs:
        env_id = raw["env_id"]
        if not check_hub_env_installed(env_id):
            missing_envs.append(env_id)

    if missing_envs:
        logger.error("Missing environments. Install them first:")
        for env_id in missing_envs:
            logger.error(f"  prime env install {env_id}")
        raise SystemExit(1)

    eval_configs = [build_eval_config(raw) for raw in raw_eval_configs]
    for config in eval_configs:
        logger.debug(f"Evaluation config: {config.model_dump_json(indent=2)}")

    eval_run_config = EvalRunConfig(
        evals=eval_configs, heartbeat_url=args.heartbeat_url
    )
    if args.debug:
        asyncio.run(run_evaluations(eval_run_config))
    else:
        asyncio.run(run_evaluations_tui(eval_run_config, tui_mode=args.tui))


if __name__ == "__main__":
    main()
