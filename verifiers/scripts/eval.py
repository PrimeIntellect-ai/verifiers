import argparse
import asyncio
import importlib
import importlib.util
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, cast

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import numpy as np
from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import Endpoints, GenerateOutputs
from verifiers.utils.client_utils import setup_client
from verifiers.utils.logging_utils import setup_logging
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls

# Setup logger for eval script using verifiers logging format
logger = logging.getLogger("verifiers.scripts.eval")


def push_eval_to_env_hub(
    eval_name: str,
    model_name: str,
    dataset: str,
    metrics: dict[str, float],
    metadata: dict[str, Any],
    results: list[dict[str, Any]] | None = None,
    skip_env_check: bool = False,
    run_id: str | None = None,
    version_id: str | None = None,
    framework: str = "verifiers",
) -> dict[str, Any] | None:
    try:
        from prime_evals import APIClient, EvalsClient, EvalsAPIError

        api_client = APIClient()
        client = EvalsClient(api_client)

        # Try to load environment metadata from local .env-metadata.json file
        env_id_to_use = None
        version_id_to_use = version_id

        # Look for .env-metadata.json in the environment directory
        env_dir = (
            Path(__file__).parent.parent.parent
            / "environments"
            / dataset.replace("-", "_")
        )
        hub_metadata_file = env_dir / ".env-metadata.json"

        if hub_metadata_file.exists():
            try:
                with open(hub_metadata_file) as f:
                    hub_metadata = json.load(f)
                    env_id_to_use = hub_metadata.get("environment_id")
                    if not version_id_to_use:
                        version_id_to_use = hub_metadata.get("version_id")
                    logger.debug(
                        f"✓ Loaded environment metadata from {hub_metadata_file.name}: "
                        f"env_id={env_id_to_use[:8] if env_id_to_use else 'None'}..."
                    )
            except Exception as e:
                logger.debug(f"Could not load {hub_metadata_file}: {e}")

        if not skip_env_check and not env_id_to_use:
            logger.debug(
                f"Checking if environment '{dataset}' exists on Environment Hub..."
            )
            try:
                env_exists = client.check_environment_exists(dataset)
                if not env_exists:
                    if "/" in dataset:
                        env_hint = f"owner/{dataset}"
                    else:
                        env_hint = f"<owner>/{dataset}"

                    logger.error(
                        f"✗ Cannot push eval: Environment '{dataset}' not found on Environment Hub.\n"
                        f"  Please push the environment first using one of these methods:\n"
                        f"  1. Using verifiers: env.push_to_env_hub(hub_name='{env_hint}')\n"
                        f"  2. Using prime CLI: prime env push {dataset}\n"
                        f"  3. Visit: https://app.primeintellect.ai/environments\n"
                    )
                    return None
                logger.debug(f"✓ Environment '{dataset}' found in Prime Hub")
            except EvalsAPIError as e:
                logger.warning(
                    f"Could not verify environment existence: {e}\n"
                    f"Proceeding with eval push anyway..."
                )

        create_response = client.create_evaluation(
            name=eval_name,
            environment_ids=[env_id_to_use] if env_id_to_use else [dataset],
            run_id=run_id,
            model_name=model_name,
            dataset=dataset,
            framework=framework,
            metadata=metadata,
            metrics=metrics,
        )

        evaluation_id = create_response["evaluation_id"]
        logger.debug(f"✓ Created evaluation {evaluation_id}")

        if results:
            samples = []
            for result in results:
                sample = {
                    "example_id": result.get("example_id", 0),
                    "reward": result.get("reward", 0.0),
                    "task": result.get("task"),
                    "answer": result.get("answer"),
                    "prompt": result.get("prompt"),
                    "completion": result.get("completion"),
                    "score": result.get("score"),
                    "correct": result.get("correct"),
                    "num_steps": result.get("num_steps"),
                    "total_time": result.get("total_time"),
                    "latency_ms": result.get("latency_ms"),
                    "rollout_number": result.get("rollout_number"),
                    "metadata": {
                        k: v
                        for k, v in result.items()
                        if k
                        not in [
                            "example_id",
                            "reward",
                            "task",
                            "answer",
                            "prompt",
                            "completion",
                            "score",
                            "correct",
                            "num_steps",
                            "total_time",
                            "latency_ms",
                            "rollout_number",
                        ]
                    },
                }
                samples.append(sample)

            client.push_samples(evaluation_id, samples)
            logger.debug(f"✓ Pushed {len(samples)} samples")

        finalize_response = client.finalize_evaluation(evaluation_id, metrics=metrics)

        viewer_url = finalize_response.get("viewer_url")
        if viewer_url:
            logger.info(
                f"✓ Pushed eval '{eval_name}' to Prime Hub\n  View at: {viewer_url}"
            )
        else:
            logger.info(f"✓ Pushed eval '{eval_name}' to Prime Hub")

        return finalize_response

    except ImportError:
        logger.warning(
            "prime-evals not found. Install with: pip install prime-evals\n"
            "Skipping push to Prime Hub."
        )
    except Exception as e:
        logger.warning(f"Failed to push eval to Prime Hub: {e}")

    return None


async def eval_environment_async(
    env: str,
    env_args: dict,
    client: AsyncOpenAI,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    sampling_args: dict | None,
) -> tuple[str, GenerateOutputs]:
    logger.info(f"Loading environment: {env}")
    vf_env = vf.load_environment(env_id=env, **env_args)

    if vf_env.eval_dataset is None:
        logger.debug(f"No eval dataset for {env}, using train dataset")
        dataset = vf_env.get_dataset(n=num_examples)
    else:
        dataset = vf_env.get_eval_dataset(n=num_examples)

    assert dataset is not None
    if rollouts_per_example > 1:
        dataset = dataset.repeat(rollouts_per_example)

    logger.info(f"Evaluating {env} with {len(dataset)} samples...")

    results = await vf_env.a_generate(
        inputs=dataset,
        client=client,
        model=model,
        sampling_args=sampling_args,
        score_rollouts=True,
        max_concurrent=max_concurrent,
    )

    return env, results


async def eval_environments_parallel(
    envs: list[str],
    env_args_dict: dict[str, dict],
    client: AsyncOpenAI,
    model: str,
    num_examples: list[int],
    rollouts_per_example: list[int],
    max_concurrent: list[int],
    sampling_args: dict | None = None,
    sampling_args_dict: dict[str, dict] | None = None,
) -> dict[str, GenerateOutputs]:
    tasks = []
    for env, n, r, c in zip(envs, num_examples, rollouts_per_example, max_concurrent):
        env_sampling_args = sampling_args
        if sampling_args_dict and env in sampling_args_dict:
            env_sampling_args = sampling_args_dict[env]

        tasks.append(
            eval_environment_async(
                env=env,
                env_args=env_args_dict.get(env, {}),
                client=client,
                model=model,
                num_examples=n,
                rollouts_per_example=r,
                max_concurrent=c,
                sampling_args=env_sampling_args,
            )
        )

    results = await asyncio.gather(*tasks)

    return dict(results)


def eval_environment(
    env: str,
    env_args: dict,
    env_dir_path: str,
    endpoints_path: str,
    model: str,
    api_key_var: str,
    api_base_url: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int | None,
    temperature: float | None,
    sampling_args: dict | None,
    verbose: bool,
    save_dataset: bool,
    save_to_hf_hub: bool,
    hf_hub_dataset_name: str,
    extra_headers: Dict[str, str],
):
    setup_logging("DEBUG" if verbose else "INFO")
    try:
        endpoints_path_obj = Path(endpoints_path)
        if endpoints_path_obj.is_dir():
            endpoints_file = endpoints_path_obj / "endpoints.py"
        else:
            endpoints_file = endpoints_path_obj

        if endpoints_file.exists():
            logger.debug(f"Loading endpoint registry from {endpoints_file}")
            spec = importlib.util.spec_from_file_location("endpoints", endpoints_file)
            assert spec and spec.loader
            endpoints_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(endpoints_module)
            # check that module exposes ENDPOINTS
            if not hasattr(endpoints_module, "ENDPOINTS"):
                raise AttributeError(
                    f"Module '{endpoints_file}' does not have a 'ENDPOINTS' attribute"
                )
            ENDPOINTS = cast(Endpoints, endpoints_module.ENDPOINTS)
            logger.debug(
                f"Successfully loaded {len(ENDPOINTS)} endpoints from registry"
            )
        else:
            raise ImportError(f"endpoints.py not found at {endpoints_file}")
    except (ImportError, AttributeError) as e:
        logger.warning(
            f"No local endpoint registry found at {endpoints_path}. "
            f"Please specify the model name (-m), API host base URL (-b), and API key variable name (-k). "
            f"Error details: {str(e)}"
        )
        logger.debug("Using default empty endpoints registry")
        ENDPOINTS: Endpoints = {}

    if model in ENDPOINTS:
        api_key_var = ENDPOINTS[model]["key"]
        api_base_url = ENDPOINTS[model]["url"]
        model = ENDPOINTS[model]["model"]
        logger.debug(f"Using endpoint configuration for model '{model}' from registry")
    else:
        logger.debug(
            f"Model '{model}' not found in endpoint registry, using command-line arguments"
        )

    # Setup eval client with high limits to prevent API timeout errors
    client = setup_client(
        api_base_url,
        api_key_var,
        timeout=3600.0,  # 1h
        max_connections=28000,  # Number of available ports
        max_keepalive_connections=28000,  # Number of available ports
        max_retries=10,  # 10 retries (w/ exponential backoffs)
        extra_headers=extra_headers,
    )
    logger.debug(f"Initialized OpenAI client with base_url: {api_base_url}")
    vf_env = vf.load_environment(env_id=env, **env_args)
    # Merge sampling args with precedence to JSON payload over explicit flags
    merged_sampling_args: dict = {}
    if sampling_args is not None:
        merged_sampling_args.update(sampling_args)
    if "max_tokens" not in merged_sampling_args:
        merged_sampling_args["max_tokens"] = max_tokens
    if temperature is not None and "temperature" not in merged_sampling_args:
        merged_sampling_args["temperature"] = temperature

    logger.info(f"Starting evaluation with model: {model}")
    logger.info(
        f"Configuration: num_examples={num_examples}, rollouts_per_example={rollouts_per_example}, max_concurrent={max_concurrent}"
    )
    start_time = time.time()
    results = vf_env.evaluate(
        client=client,
        model=model,
        sampling_args=merged_sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
    )
    end_time = time.time()
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    print("--- Evaluation ---")
    print(f"Environment: {env}")
    print(f"Model: {model}")
    print(f"Provider: {api_base_url}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")
    print("--- Example ---")
    printable_prompts = [messages_to_printable(p) for p in results.prompt]
    printable_completions = [messages_to_printable(c) for c in results.completion]
    vf.print_prompt_completions_sample(
        printable_prompts, printable_completions, results.reward, step=0
    )
    print("--- All ---")
    print("Rewards:")
    print(
        f"reward: avg - {sum(results.reward) / len(results.reward):.3f}, std - {np.std(results.reward):.3f}"
    )
    r = rollouts_per_example
    n = len(results.reward) // r
    for i in range(r):
        # rounded to 3 decimal places
        trials = [round(results.reward[(i * n) + j], 3) for j in range(n)]
        out = f"r{i + 1}: {trials}"
        print(out)
    for k in results.metrics:
        v = results.metrics[k]
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
        for i in range(r):
            # rounded to 3 decimal places
            trials = [round(v[(i * n) + j], 3) for j in range(n)]
            out = f"r{i + 1}: {trials}"
            print(out)

    if save_dataset or save_to_hf_hub:
        ids = [i // rollouts_per_example for i in range(n * rollouts_per_example)]
        rewards = results.reward
        tasks = results.task
        data_dict = {
            "id": ids,
            "prompt": [sanitize_tool_calls(p) for p in printable_prompts],
            "completion": [sanitize_tool_calls(c) for c in printable_completions],
            "task": tasks,
            "generation_ms": [s["timing"]["generation_ms"] for s in results.state],
            "scoring_ms": [s["timing"]["scoring_ms"] for s in results.state],
            "total_ms": [s["timing"]["total_ms"] for s in results.state],
        }
        if results.info[0] != {}:
            data_dict["info"] = results.info
        if results.answer[0] != "":
            data_dict["answer"] = results.answer
        data_dict["reward"] = rewards
        for k in results.metrics:
            v = results.metrics[k]
            data_dict[k] = v

        dataset = Dataset.from_dict(data_dict)
        metadata = {
            "env": env,
            "model": model,
            "num_examples": n,
            "rollouts_per_example": rollouts_per_example,
            "sampling_args": merged_sampling_args,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "time_ms": (end_time - start_time) * 1000,
            "avg_reward": sum(results.reward) / len(results.reward),
        }
        for k in results.metrics:
            metadata[f"avg_{k}"] = sum(results.metrics[k]) / len(results.metrics[k])

        uuid_str = str(uuid.uuid4())[:8]
        env_model_str = f"{env}--{model.replace('/', '--')}"
        if save_dataset:
            module_name = env.replace("-", "_")
            local_env_dir = Path(env_dir_path) / module_name
            if local_env_dir.exists():
                results_path = (
                    local_env_dir / "outputs" / "evals" / env_model_str / uuid_str
                )
            else:
                results_path = Path("./outputs") / "evals" / env_model_str / uuid_str
            results_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_json(results_path / "results.jsonl")
            with open(results_path / "metadata.json", "w") as f:
                json.dump(metadata, f)

            logger.info(f"Saved dataset to {results_path}")
        if save_to_hf_hub:
            if hf_hub_dataset_name == "":
                dataset_name = (
                    f"{env}_{model.replace('/', '-')}_n{n}_r{rollouts_per_example}"
                )
            else:
                dataset_name = hf_hub_dataset_name
            dataset.push_to_hub(dataset_name)
            logger.info(f"Saved dataset to Hugging Face Hub: {dataset_name}")


def display_and_push_results(
    env,
    results,
    model,
    args,
    num_examples,
    rollouts_per_example,
    max_concurrent,
    sampling_args,
):
    """Display results and optionally push to Prime Hub."""
    logger.info(f"\n--- {env} ---")
    logger.info(
        f"Rewards: avg={np.mean(results.reward):.3f}, std={np.std(results.reward):.3f}"
    )

    for metric_name, metric_values in results.metrics.items():
        logger.info(
            f"{metric_name}: avg={np.mean(metric_values):.3f}, std={np.std(metric_values):.3f}"
        )

    if args.save_to_env_hub:
        eval_name = (
            args.eval_name
            or f"{model.replace('/', '-')}-{env}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        metrics = {
            "avg_reward": float(np.mean(results.reward)),
            "std_reward": float(np.std(results.reward)),
            "num_samples": len(results.reward),
        }
        for metric_name, metric_values in results.metrics.items():
            metrics[f"avg_{metric_name}"] = float(np.mean(metric_values))
            metrics[f"std_{metric_name}"] = float(np.std(metric_values))

        metadata = {
            "environment": env,
            "model": model,
            "num_examples": num_examples,
            "rollouts_per_example": rollouts_per_example,
            "max_concurrent": max_concurrent,
            "sampling_args": sampling_args,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

        sample_results = []
        for i in range(len(results.reward)):
            result_entry = {
                "example_id": i // rollouts_per_example,
                "rollout_number": i % rollouts_per_example,
                "reward": float(results.reward[i]),
                "prompt": results.prompt[i] if i < len(results.prompt) else [],
                "completion": results.completion[i]
                if i < len(results.completion)
                else [],
            }
            if i < len(results.task):
                result_entry["task"] = str(results.task[i])
            if i < len(results.answer):
                result_entry["answer"] = str(results.answer[i])
            if i < len(results.info):
                info = results.info[i]
                if isinstance(info, dict):
                    if "score" in info:
                        result_entry["score"] = float(info["score"])
                    if "correct" in info:
                        result_entry["correct"] = bool(info["correct"])
            sample_results.append(result_entry)

        push_eval_to_env_hub(
            eval_name=eval_name,
            model_name=model,
            dataset=env,
            metrics=metrics,
            metadata=metadata,
            results=sample_results,
        )


def load_config_file(config_path: str) -> dict:
    """Load configuration from TOML or JSON file."""

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_file.suffix == ".toml":
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
    elif config_file.suffix == ".json":
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config file format: {config_file.suffix}. Use .toml or .json"
        )

    return config


def parse_env_spec(spec: str) -> tuple[str, dict]:
    """Parse environment specification like 'id=gsm8k,num_examples=10'."""
    parts = spec.split(",")
    env_id = None
    config = {}

    for part in parts:
        if "=" not in part:
            if env_id is None:
                env_id = part
            continue

        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "id":
            env_id = value
        elif key in ["num_examples", "rollouts_per_example", "max_concurrent"]:
            config[key] = int(value)
        elif key == "temperature":
            config[key] = float(value)
        else:
            try:
                config[key] = json.loads(value)
            except json.JSONDecodeError:
                config[key] = value

    if env_id is None:
        raise ValueError(f"Environment spec must include 'id': {spec}")

    return env_id, config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate environment(s) using verifiers. Config file recommended for complex multi-environment setups.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple single environment eval
  vf-eval gsm8k --num-examples 10 --model gpt-4o
  
  # Multiple environments with global settings
  vf-eval gsm8k math500 --num-examples 100 --model gpt-4o
  
  # Per-environment configuration
  vf-eval --env id=gsm8k,num_examples=100 --env id=math500,num_examples=50,rollouts_per_example=5
  
  # Config file (recommended for complex setups)
  vf-eval --config eval_config.toml
  
  # Config file with CLI overrides
  vf-eval --config eval_config.toml --num-examples 20
        """,
    )
    parser.add_argument(
        "positional_envs",
        type=str,
        nargs="*",
        metavar="env",
        help="Environment module name(s). Use --env for per-environment settings.",
    )
    parser.add_argument(
        "--env",
        action="append",
        dest="env_specs",
        help="Environment specification: 'id=name,key=val,...'. Repeatable for multiple environments.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML or JSON config file. CLI args override config file values.",
    )
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default=None,
        help='Environment module arguments as JSON. For multi-env: \'{"env1": {"key": "val"}, "env2": {...}}\'',
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
        help="Name of model to evaluate (global default)",
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for API key",
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for API",
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
        help="Number of examples to evaluate (global default)",
    )
    parser.add_argument(
        "--rollouts-per-example",
        "-r",
        type=int,
        default=None,
        help="Number of rollouts per example (global default)",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=None,
        help="Maximum number of concurrent requests (global default)",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=None,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        "-T",
        type=float,
        default=None,
        help="Temperature for sampling (global default)",
    )
    parser.add_argument(
        "--sampling-args",
        "-S",
        type=json.loads,
        default=None,
        help="Sampling arguments as JSON object. Example: '{\"enable_thinking\": false}'",
    )
    parser.add_argument(
        "--verbose", "-v", default=None, action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--save-dataset",
        "-s",
        default=None,
        action="store_true",
        help="Save dataset to disk",
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
        default="",
        help="Name of dataset to save to Hugging Face Hub",
    )
    parser.add_argument(
        "--save-to-env-hub",
        "-P",
        default=None,
        action="store_true",
        help=(
            "Save evaluation results to Prime Hub (requires prime-evals). "
            "NOTE: The environment must be pushed to Prime Hub first."
        ),
    )
    parser.add_argument(
        "--eval-name",
        type=str,
        default=None,
        help="Name for the evaluation run (used when saving to Prime Hub)",
    )
    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config_file(args.config)

        # Merge config values with CLI args (CLI takes precedence)
        args.save_to_env_hub = (
            args.save_to_env_hub
            if args.save_to_env_hub is not None
            else config.get("save_to_env_hub", False)
        )
        args.save_dataset = (
            args.save_dataset
            if args.save_dataset is not None
            else config.get("save_dataset", False)
        )
        args.save_to_hf_hub = (
            args.save_to_hf_hub
            if args.save_to_hf_hub is not None
            else config.get("save_to_hf_hub", False)
        )
        args.verbose = (
            args.verbose if args.verbose is not None else config.get("verbose", False)
        )
        args.eval_name = args.eval_name or config.get("eval_name")
        args.hf_hub_dataset_name = args.hf_hub_dataset_name or config.get(
            "hf_hub_dataset_name", ""
        )
    else:
        args.save_to_env_hub = args.save_to_env_hub or False
        args.save_dataset = args.save_dataset or False
        args.save_to_hf_hub = args.save_to_hf_hub or False
        args.verbose = args.verbose or False

    envs = []
    env_spec_configs = {}

    if args.env_specs:
        for spec in args.env_specs:
            env_id, env_config = parse_env_spec(spec)
            envs.append(env_id)
            if env_config:
                env_spec_configs[env_id] = env_config
    elif args.positional_envs:
        envs = args.positional_envs
    else:
        envs = config.get("environment_ids", [])

    if not envs:
        parser.error(
            "No environments specified. Use positional args, --env, or --config file with 'environment_ids'."
        )

    default_model = (
        args.model or config.get("model", {}).get("name")
        if isinstance(config.get("model"), dict)
        else args.model or config.get("model", "gpt-4.1-mini")
    )
    default_num_examples = (
        args.num_examples
        if args.num_examples is not None
        else config.get("num_examples", 5)
    )
    default_rollouts = (
        args.rollouts_per_example
        if args.rollouts_per_example is not None
        else config.get("rollouts_per_example", 3)
    )
    default_max_concurrent = (
        args.max_concurrent
        if args.max_concurrent is not None
        else config.get("max_concurrent", 32)
    )
    default_api_key_var = args.api_key_var or config.get(
        "api_key_var", "OPENAI_API_KEY"
    )
    default_api_base_url = args.api_base_url or config.get(
        "api_base_url", "https://api.openai.com/v1"
    )

    config_sampling = config.get("sampling", {})
    default_sampling_args = {}
    if args.sampling_args:
        default_sampling_args.update(args.sampling_args)
    else:
        default_sampling_args.update(config_sampling)

    if args.max_tokens is not None:
        default_sampling_args["max_tokens"] = args.max_tokens
    elif "max_tokens" not in default_sampling_args and config_sampling.get(
        "max_tokens"
    ):
        default_sampling_args["max_tokens"] = config_sampling["max_tokens"]

    if args.temperature is not None:
        default_sampling_args["temperature"] = args.temperature
    elif "temperature" not in default_sampling_args and config_sampling.get(
        "temperature"
    ):
        default_sampling_args["temperature"] = config_sampling["temperature"]

    # Extract per-environment overrides from config (supports new [env.X] and legacy *_per_env formats)
    env_configs = config.get("env", {})

    num_examples_per_env = {}
    rollouts_per_env = {}
    max_concurrent_per_env = {}
    models_per_env = {}
    sampling_args_per_env = {}
    env_args_dict_from_config = {}

    # Parse new [env.X] format
    for env_id, env_config in env_configs.items():
        if isinstance(env_config, dict):
            if "num_examples" in env_config:
                num_examples_per_env[env_id] = env_config["num_examples"]
            if "rollouts_per_example" in env_config:
                rollouts_per_env[env_id] = env_config["rollouts_per_example"]
            if "max_concurrent" in env_config:
                max_concurrent_per_env[env_id] = env_config["max_concurrent"]
            if "model" in env_config:
                models_per_env[env_id] = env_config["model"]

            # Sampling args
            sampling_keys = {
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
            }
            env_sampling = {k: v for k, v in env_config.items() if k in sampling_keys}
            if env_sampling:
                sampling_args_per_env[env_id] = env_sampling

            # Filter out script-level args from environment init args
            reserved_keys = {
                "num_examples",
                "rollouts_per_example",
                "max_concurrent",
                "model",
                "save_to_env_hub",
                "save_dataset",
                "save_to_hf_hub",
                "verbose",
                "eval_name",
                "api_key_var",
                "api_base_url",
            } | sampling_keys
            env_args = {k: v for k, v in env_config.items() if k not in reserved_keys}
            if env_args:
                env_args_dict_from_config[env_id] = env_args

    # Merge with legacy format (legacy takes precedence if both exist)
    num_examples_per_env.update(config.get("num_examples_per_env", {}))
    rollouts_per_env.update(config.get("rollouts_per_example_per_env", {}))
    max_concurrent_per_env.update(config.get("max_concurrent_per_env", {}))
    models_per_env.update(config.get("models_per_env", {}))
    sampling_args_per_env.update(config.get("sampling_args_per_env", {}))
    env_args_dict_from_config.update(
        config.get("environment_args", config.get("env_args_per_env", {}))
    )

    # Merge CLI --env spec configs (they override config file)
    for env_id, env_config in env_spec_configs.items():
        if "num_examples" in env_config:
            num_examples_per_env[env_id] = env_config["num_examples"]
        if "rollouts_per_example" in env_config:
            rollouts_per_env[env_id] = env_config["rollouts_per_example"]
        if "max_concurrent" in env_config:
            max_concurrent_per_env[env_id] = env_config["max_concurrent"]
        if "temperature" in env_config:
            if env_id not in sampling_args_per_env:
                sampling_args_per_env[env_id] = {}
            sampling_args_per_env[env_id]["temperature"] = env_config["temperature"]

    # Handle env_args from CLI
    if args.env_args:
        env_args_dict = args.env_args
    else:
        env_args_dict = env_args_dict_from_config

    # Setup client (reuse headers from args if provided)
    headers_dict = None
    if args.header:
        headers_dict = {}
        for header_str in args.header:
            if ":" not in header_str:
                logger.warning(f"Skipping malformed header: {header_str}")
                continue
            key, value = header_str.split(":", 1)
            headers_dict[key.strip()] = value.strip()

    api_key_value = os.getenv(default_api_key_var)
    if not api_key_value:
        raise ValueError(
            f"API key not found in environment variable: {default_api_key_var}"
        )

    client = AsyncOpenAI(
        api_key=api_key_value,
        base_url=default_api_base_url,
        default_headers=headers_dict,
    )

    # Build per-environment lists
    num_examples_list = []
    rollouts_list = []
    max_concurrent_list = []
    model_list = []
    sampling_args_dict = {}

    for env in envs:
        # Determine values for this environment
        num_examples_list.append(num_examples_per_env.get(env, default_num_examples))
        rollouts_list.append(rollouts_per_env.get(env, default_rollouts))
        max_concurrent_list.append(
            max_concurrent_per_env.get(env, default_max_concurrent)
        )
        model_list.append(models_per_env.get(env, default_model))

        # Merge sampling args
        if env in sampling_args_per_env:
            env_sampling = default_sampling_args.copy() if default_sampling_args else {}
            env_sampling.update(sampling_args_per_env[env])
            sampling_args_dict[env] = env_sampling

    logger.info(
        f"Evaluating {len(envs)} environment{'s' if len(envs) > 1 else ''}: {', '.join(envs)}"
    )
    if len(envs) > 1:
        logger.info("Running evaluations in parallel...")

    async def run_multi_model_eval():
        tasks = []
        for idx, env in enumerate(envs):
            env_model = model_list[idx]
            env_sampling = (
                sampling_args_dict.get(env, default_sampling_args)
                if sampling_args_dict
                else default_sampling_args
            )

            task = eval_environment_async(
                env=env,
                env_args=env_args_dict.get(env, {}),
                client=client,
                model=env_model,
                num_examples=num_examples_list[idx],
                rollouts_per_example=rollouts_list[idx],
                max_concurrent=max_concurrent_list[idx],
                sampling_args=env_sampling,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        all_results = {}
        for env_name, env_result in results:
            all_results[env_name] = env_result

        return all_results

    results_dict = asyncio.run(run_multi_model_eval())

    if len(envs) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)

    for idx, (env, results) in enumerate(results_dict.items()):
        env_model = model_list[idx]
        env_sampling_args = (
            sampling_args_dict.get(env, default_sampling_args)
            if sampling_args_dict
            else default_sampling_args
        )

        display_and_push_results(
            env=env,
            results=results,
            model=env_model,
            args=args,
            num_examples=num_examples_list[idx],
            rollouts_per_example=rollouts_list[idx],
            max_concurrent=max_concurrent_list[idx],
            sampling_args=env_sampling_args,
        )

    if len(envs) > 1:
        logger.info("\n" + "=" * 80)
        logger.info(f"✓ Completed evaluation of {len(envs)} environments")
        logger.info("=" * 80)
    else:
        logger.info("✓ Evaluation complete")


if __name__ == "__main__":
    main()
