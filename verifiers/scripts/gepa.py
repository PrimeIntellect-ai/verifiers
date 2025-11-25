#!/usr/bin/env python3
"""
GEPA optimization script for Verifiers environments.

Usage:
    vf-gepa wordle --budget light
    vf-gepa wiki-search --budget heavy --components system_prompt tool_descriptions
    vf-gepa my-env --max-metric-calls 1000 -n 100 --num-val 30
"""

import argparse
import asyncio
import json
import logging
import os
import sys

try:
    from gepa import optimize  # noqa: F401
except ImportError:
    print("Error: GEPA is not installed.")
    print("Install with: uv add 'verifiers[gepa]'")
    sys.exit(1)

from verifiers import setup_logging
from verifiers.types import ClientConfig, GEPAConfig
from verifiers.utils.eval_utils import load_endpoints
from verifiers.utils.gepa_utils import (
    auto_budget_to_metric_calls,
    ensure_env_dir_on_path,
    get_env_gepa_defaults,
    prepare_gepa_dataset,
    run_gepa_optimization,
)

import verifiers as vf

logger = logging.getLogger("gepa")

# Default constants
DEFAULT_NUM_EXAMPLES = 50
DEFAULT_NUM_VAL = 20
DEFAULT_ROLLOUTS_PER_EXAMPLE = 1


def main():
    parser = argparse.ArgumentParser(
        description="Run GEPA prompt optimization on Verifiers environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Light optimization (quick test)
  vf-gepa wordle --budget light

  # Heavy optimization with tool descriptions
  vf-gepa wiki-search --budget heavy --components system_prompt tool_descriptions

  # Custom configuration
  vf-gepa my-env --max-metric-calls 1000 -n 100 --num-val 30
        """,
    )

    # 1. Positional: env_id
    parser.add_argument(
        "env_id", type=str, help="Environment ID (e.g., wordle, wiki-search)"
    )

    # 2. Environment config
    parser.add_argument(
        "--env-args",
        "-a",
        default="{}",
        help="JSON dict of keyword args forwarded to vf.load_environment",
    )
    parser.add_argument(
        "--env-dir-path",
        "-p",
        type=str,
        default="./environments",
        help="Path to environments directory",
    )

    # 3. Dataset
    parser.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=None,
        help="Number of training examples",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=None,
        help="Number of validation examples",
    )

    # 4. Endpoints/Model
    parser.add_argument(
        "--endpoints-path",
        "-e",
        type=str,
        default="./configs/endpoints.py",
        help="Path to API endpoints registry",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-5-mini",
        help="Model to optimize (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        default="OPENAI_API_KEY",
        help="Environment variable containing the task model API key",
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        default="https://api.openai.com/v1",
        help="Base URL for the task model API (default: https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--header",
        action="append",
        dest="headers",
        default=None,
        help="Additional HTTP header for the task model client. Format: 'Name: Value'. Repeatable.",
    )

    # 5. Sampling
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for task model (default: 1.0)",
    )
    parser.add_argument(
        "-t",
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for task model (unset to use model default)",
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

    # 6. Rollouts
    parser.add_argument(
        "--rollouts-per-example",
        "-r",
        type=int,
        default=None,
        help="Number of rollouts per example",
    )

    # 7. Concurrency
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=32,
        help="Maximum number of concurrent requests",
    )

    # 8. GEPA budget (mutually exclusive)
    budget_group = parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument(
        "--budget",
        "-B",
        choices=["light", "medium", "heavy"],
        help="Budget preset: light (~6 candidates), medium (~12), heavy (~18)",
    )
    budget_group.add_argument(
        "--max-metric-calls", type=int, help="Maximum total metric calls budget"
    )

    # 9. GEPA configuration
    parser.add_argument(
        "--components",
        nargs="+",
        default=["system_prompt"],
        help="Components to optimize (default: system_prompt)",
    )
    parser.add_argument(
        "--reflection-model",
        default="gpt-5-mini",
        help="Model for reflection/proposal (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--reflection-temperature",
        type=float,
        default=1.0,
        help="Temperature for reflection model (default: 1.0)",
    )
    parser.add_argument(
        "--reflection-base-url",
        default=None,
        help="Base URL for reflection model API (default: task client base URL)",
    )
    parser.add_argument(
        "--reflection-api-key-var",
        default="OPENAI_API_KEY",
        help="Env var that stores the reflection API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--reflection-max-tokens",
        type=int,
        default=8000,
        help="Max tokens for reflection completions (default: 8000)",
    )
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=35,
        help="Number of examples per reflection step (default: 35)",
    )

    # 10. Output/Logging
    parser.add_argument(
        "--save-results",
        "-s",
        default=False,
        action="store_true",
        help="Save rollout trajectories to disk",
    )
    parser.add_argument(
        "--save-every",
        "-f",
        type=int,
        default=-1,
        help="Save rollout trajectories every n evaluations during optimization",
    )
    parser.add_argument(
        "--track-stats",
        action="store_true",
        help="Track detailed optimization statistics",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # 11. Experiment tracking - wandb
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity/team name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated from env_id)",
    )
    parser.add_argument(
        "--wandb-api-key-var",
        type=str,
        default="WANDB_API_KEY",
        help="Environment variable containing wandb API key (default: WANDB_API_KEY)",
    )
    parser.add_argument(
        "--wandb-init-kwargs",
        type=json.loads,
        default=None,
        help='Additional wandb.init() kwargs as JSON (e.g., \'{"tags": ["gepa"], "mode": "offline"}\')',
    )

    # 12. Experiment tracking - mlflow
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Enable mlflow logging",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    # Parse env_args
    try:
        env_args = json.loads(args.env_args)
        if not isinstance(env_args, dict):
            raise TypeError("env args must be a JSON object")
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            "--env-args must be valid JSON representing a dictionary"
        ) from exc

    # Parse headers
    task_client_headers: dict[str, str] | None = None
    if args.headers:
        task_client_headers = {}
        for header in args.headers:
            if ":" not in header:
                raise ValueError(
                    "Headers must be provided in the format 'Name: Value'."
                )
            key, value = header.split(":", 1)
            task_client_headers[key.strip()] = value.strip()

    # Setup logging
    setup_logging("DEBUG" if args.verbose else "INFO")

    # Silence noisy third-party loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info(f"Starting GEPA optimization for environment: {args.env_id}")
    logger.info(f"Components to optimize: {args.components}")

    if args.save_every > 0 and not args.save_results:
        logger.warning("--save-every is ignored unless --save-results is set")

    # Apply defaults: CLI > env pyproject.toml > hardcoded
    env_defaults = get_env_gepa_defaults(args.env_id)
    num_examples = (
        args.num_examples
        if args.num_examples is not None
        else env_defaults.get("num_examples", DEFAULT_NUM_EXAMPLES)
    )
    num_val = (
        args.num_val
        if args.num_val is not None
        else env_defaults.get("num_val", DEFAULT_NUM_VAL)
    )
    rollouts_per_example = (
        args.rollouts_per_example
        if args.rollouts_per_example is not None
        else env_defaults.get("rollouts_per_example", DEFAULT_ROLLOUTS_PER_EXAMPLE)
    )

    # Log sources
    if args.num_examples is None:
        source = "pyproject.toml" if "num_examples" in env_defaults else "default"
        logger.debug(f"Using num_examples={num_examples} from {source}")
    if args.num_val is None:
        source = "pyproject.toml" if "num_val" in env_defaults else "default"
        logger.debug(f"Using num_val={num_val} from {source}")
    if args.rollouts_per_example is None:
        source = (
            "pyproject.toml" if "rollouts_per_example" in env_defaults else "default"
        )
        logger.debug(f"Using rollouts_per_example={rollouts_per_example} from {source}")

    # Load endpoints and resolve model config
    endpoints = load_endpoints(args.endpoints_path)
    if args.model in endpoints:
        task_api_key_var = endpoints[args.model]["key"]
        task_api_base_url = endpoints[args.model]["url"]
        args.model = endpoints[args.model]["model"]
        logger.debug(f"Using endpoint configuration for task model '{args.model}'")
    else:
        logger.debug(f"Task model '{args.model}' not in registry, using CLI args")
        task_api_key_var = args.api_key_var
        task_api_base_url = args.api_base_url

    # Also check reflection model
    if args.reflection_model in endpoints:
        reflection_api_key_var = endpoints[args.reflection_model]["key"]
        reflection_base_url = endpoints[args.reflection_model]["url"]
        args.reflection_model = endpoints[args.reflection_model]["model"]
        logger.debug(f"Using endpoint for reflection model '{args.reflection_model}'")
    else:
        reflection_api_key_var = args.reflection_api_key_var
        reflection_base_url = args.reflection_base_url

    # Merge sampling args with precedence to JSON payload
    merged_sampling_args: dict = {}
    if args.sampling_args is not None:
        merged_sampling_args.update(args.sampling_args)
    if "max_tokens" not in merged_sampling_args:
        merged_sampling_args["max_tokens"] = args.max_tokens
    if args.temperature is not None and "temperature" not in merged_sampling_args:
        merged_sampling_args["temperature"] = args.temperature

    # Ensure local environments directory is available for imports
    ensure_env_dir_on_path(args.env_dir_path, args.env_id)

    # Setup client config
    client_config_kwargs = {
        "api_key_var": task_api_key_var,
        "api_base_url": task_api_base_url,
    }
    if task_client_headers is not None:
        client_config_kwargs["extra_headers"] = task_client_headers

    client_config = ClientConfig(**client_config_kwargs)

    # Load environment
    vf_env = vf.load_environment(env_id=args.env_id, **env_args)

    # Prepare datasets
    logger.info(f"Loading {num_examples} training examples")
    logger.info(f"Loading {num_val} validation examples")
    if vf_env.eval_dataset is not None:
        train_dataset_raw = vf_env.get_dataset(n=num_examples, seed=args.seed)
        val_dataset_raw = vf_env.get_eval_dataset(n=num_val, seed=args.seed + 1)
    else:
        total_requested = max(num_examples, 0) + max(num_val, 0)
        base_dataset = vf_env.get_dataset(n=total_requested, seed=args.seed)
        base_examples = (
            base_dataset.to_list()
            if hasattr(base_dataset, "to_list")
            else list(base_dataset)
        )
        train_dataset_raw = (
            base_examples[:num_examples] if num_examples > 0 else base_examples
        )
        val_dataset_raw = (
            base_examples[num_examples : num_examples + num_val] if num_val > 0 else []
        )
        logger.debug(
            "Eval dataset missing; derived %s validation examples from train split",
            len(val_dataset_raw),
        )

    trainset = prepare_gepa_dataset(train_dataset_raw)
    valset = prepare_gepa_dataset(val_dataset_raw)

    if num_examples > 0 and not trainset:
        raise ValueError(
            "Training dataset is empty - check environment configuration and filters"
        )
    if num_val > 0 and not valset:
        raise ValueError(
            "Validation dataset is empty - check environment configuration and filters"
        )

    logger.info(f"Training set: {len(trainset)} examples")
    logger.info(f"Validation set: {len(valset)} examples")

    # Get reflection API key
    reflection_api_key = os.getenv(reflection_api_key_var)
    if not reflection_api_key:
        raise ValueError(
            f"{reflection_api_key_var} environment variable not set for reflection client"
        )

    # Use resolved reflection_base_url or fall back to task client base URL
    if not reflection_base_url:
        reflection_base_url = task_api_base_url

    # Extract seed candidate (initial component values)
    seed_candidate = {}
    for comp in args.components:
        if comp == "tool_descriptions":
            # Extract tool descriptions
            if hasattr(vf_env, "oai_tools") and vf_env.oai_tools:
                for i, tool in enumerate(vf_env.oai_tools):
                    seed_candidate[f"tool_{i}_description"] = tool["function"][
                        "description"
                    ]
        elif hasattr(vf_env, comp):
            seed_candidate[comp] = getattr(vf_env, comp)
        else:
            raise ValueError(
                f"Environment '{args.env_id}' does not have component '{comp}'. "
                f"Available components: system_prompt, tool_descriptions"
            )

    if not seed_candidate:
        raise ValueError(
            f"No valid components found to optimize for environment '{args.env_id}'"
        )

    # Convert budget preset to max_metric_calls if needed
    if args.budget:
        max_metric_calls = auto_budget_to_metric_calls(
            auto=args.budget,
            num_components=len(seed_candidate),
            valset_size=len(valset),
            minibatch_size=args.reflection_minibatch_size,
        )
    else:
        max_metric_calls = args.max_metric_calls

    logger.info(f"Budget: {max_metric_calls} metric calls total")

    # Build GEPA config
    gepa_config = GEPAConfig(
        # environment
        env_id=args.env_id,
        env_args=env_args,
        env_dir_path=args.env_dir_path,
        # task model
        model=args.model,
        client_config=client_config,
        sampling_args=merged_sampling_args,
        # reflection model
        reflection_model=args.reflection_model,
        reflection_api_key=reflection_api_key,
        reflection_base_url=reflection_base_url,
        reflection_temperature=args.reflection_temperature,
        reflection_max_tokens=args.reflection_max_tokens,
        reflection_minibatch_size=args.reflection_minibatch_size,
        # datasets
        num_examples=num_examples,
        num_val=num_val,
        rollouts_per_example=rollouts_per_example,
        trainset=trainset,
        valset=valset,
        # optimization
        components_to_optimize=args.components,
        seed_candidate=seed_candidate,
        max_metric_calls=max_metric_calls,
        # execution
        max_concurrent=args.max_concurrent,
        seed=args.seed,
        # output
        save_results=args.save_results,
        save_every=args.save_every,
        track_stats=args.track_stats,
        verbose=args.verbose,
        # experiment tracking
        use_wandb=args.use_wandb,
        wandb_api_key_var=args.wandb_api_key_var,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_init_kwargs=args.wandb_init_kwargs,
        use_mlflow=args.use_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
    )

    # Run GEPA optimization
    asyncio.run(run_gepa_optimization(gepa_config))


if __name__ == "__main__":
    main()
