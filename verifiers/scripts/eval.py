import argparse
import asyncio
import importlib
import importlib.util
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, cast

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


def push_eval_to_prime_hub(
    eval_name: str,
    model_name: str,
    dataset: str,
    metrics: dict[str, float],
    metadata: dict[str, Any],
    results: list[dict[str, Any]] | None = None,
    skip_env_check: bool = False,
) -> None:
    try:
        from prime_cli.api.evals import EvalsClient, EvalsAPIError
        
        client = EvalsClient()
        
        if not skip_env_check:
            logger.debug(f"Checking if environment '{dataset}' exists on Environment Hub...")
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
                        f"  1. Using verifiers: env.push_to_hub(hub_name='{env_hint}')\n"
                        f"  2. Using prime CLI: prime env push {dataset}\n"
                        f"  3. Visit: https://app.primeintellect.ai/environments\n"
                    )
                    return
                logger.debug(f"✓ Environment '{dataset}' found in Prime Hub")
            except EvalsAPIError as e:
                logger.warning(
                    f"Could not verify environment existence: {e}\n"
                    f"Proceeding with eval push anyway..."
                )
        
        eval_data = {
            "eval_name": eval_name,
            "model_name": model_name,
            "dataset": dataset,
            "metrics": metrics,
            "metadata": metadata,
        }
        
        if results is not None:
            eval_data["results"] = results
        
        response = client.push_eval(eval_data)
        
        viewer_url = response.get("viewer_url")
        
        if viewer_url:
            logger.info(
                f"✓ Pushed eval '{eval_name}' to Prime Hub\n"
                f"  View at: {viewer_url}"
            )
        else:
            logger.info(f"✓ Pushed eval '{eval_name}' to Prime Hub")
            
    except ImportError:
        logger.warning(
            "prime-cli not found. Install with: pip install prime-cli\n"
            "Skipping push to Prime Hub."
        )
    except Exception as e:
        logger.warning(f"Failed to push eval to Prime Hub: {e}")


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


def setup_client_from_args(args):
    try:
        endpoints_path_obj = Path(args.endpoints_path)
        if endpoints_path_obj.is_dir():
            endpoints_file = endpoints_path_obj / "endpoints.py"
        else:
            endpoints_file = endpoints_path_obj

        if endpoints_file.exists():
            logger.info(f"Loading endpoint registry from {endpoints_file}")
            spec = importlib.util.spec_from_file_location("endpoints", endpoints_file)
            assert spec and spec.loader
            endpoints_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(endpoints_module)
            ENDPOINTS = endpoints_module.ENDPOINTS
        else:
            raise ImportError(f"endpoints.py not found at {endpoints_file}")
    except (ImportError, AttributeError):
        ENDPOINTS = {}
    
    # Resolve model/API config
    model = args.model
    api_key_var = args.api_key_var
    api_base_url = args.api_base_url
    
    if model in ENDPOINTS:
        api_key_var = ENDPOINTS[model]["key"]
        api_base_url = ENDPOINTS[model]["url"]
        model = ENDPOINTS[model]["model"]
    
    api_key_value = os.getenv(api_key_var, "EMPTY")
    client = AsyncOpenAI(api_key=api_key_value, base_url=api_base_url)
    
    return client, model


def prepare_sampling_args_from_cli(args):
    """Prepare sampling arguments from CLI args."""
    merged_sampling_args = {}
    if args.sampling_args is not None:
        merged_sampling_args.update(args.sampling_args)
    if "max_tokens" not in merged_sampling_args:
        merged_sampling_args["max_tokens"] = args.max_tokens
    if args.temperature is not None and "temperature" not in merged_sampling_args:
        merged_sampling_args["temperature"] = args.temperature
    return merged_sampling_args


def display_and_push_results(env, results, model, args, num_examples, rollouts_per_example, max_concurrent, sampling_args):
    """Display results and optionally push to Prime Hub."""
    logger.info(f"\n--- {env} ---")
    logger.info(f"Rewards: avg={np.mean(results.reward):.3f}, std={np.std(results.reward):.3f}")
    
    for metric_name, metric_values in results.metrics.items():
        logger.info(f"{metric_name}: avg={np.mean(metric_values):.3f}, std={np.std(metric_values):.3f}")
    
    if args.save_to_hub:
        eval_name = args.eval_name or f"{model.replace('/', '-')}-{env}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
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
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        sample_results = []
        for i in range(len(results.reward)):
            result_entry = {
                "example_id": i,
                "reward": float(results.reward[i]),
                "prompt": results.prompt[i] if i < len(results.prompt) else [],
                "completion": results.completion[i] if i < len(results.completion) else [],
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
        
        push_eval_to_prime_hub(
            eval_name=eval_name,
            model_name=model,
            dataset=env,
            metrics=metrics,
            metadata=metadata,
            results=sample_results,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate environment(s) using verifiers. Supports single or multiple environments."
    )
    parser.add_argument(
        "env",
        type=str,
        nargs='+',
        help="Environment module name(s). Pass multiple for parallel evaluation (e.g., 'gsm8k math500')"
    )
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default={},
        help='Environment module arguments as JSON object. For single env: \'{"key": "value"}\'. For multi-env: \'{"env1": {"key": "val"}, "env2": {...}}\'',
    )
    parser.add_argument(
        "--per-env-config",
        type=json.loads,
        default=None,
        help=(
            'Per-environment configuration as JSON. Supports: num_examples, rollouts_per_example, '
            'max_concurrent, env_args, and sampling_args. '
            'Example: \'{"env1": {"num_examples": 10, "sampling_args": {"temperature": 0.8}}, '
            '"env2": {"num_examples": 20, "sampling_args": {"temperature": 0.5}}}\'. '
            'If a parameter is not specified for an environment, it falls back to the global value.'
        ),
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
        default=5,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--rollouts-per-example",
        "-r",
        type=int,
        default=3,
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
        "--save-dataset",
        "-s",
        default=False,
        action="store_true",
        help="Save dataset to disk",
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
        "--save-to-hub",
        "-P",
        default=False,
        action="store_true",
        help=(
            "Save evaluation results to Prime Hub (requires prime-cli). "
            "NOTE: The environment must be pushed to Prime Hub first. "
            "Use 'prime env push' or env.push_to_hub() before running evals."
        ),
    )
    parser.add_argument(
        "--eval-name",
        type=str,
        default=None,
        help="Name for the evaluation run (used when saving to Prime Hub)",
    )
    args = parser.parse_args()
    
    # Normalize to list
    envs = args.env if isinstance(args.env, list) else [args.env]
    
    # Setup client and model
    client, model = setup_client_from_args(args)
    
    # Prepare sampling arguments
    sampling_args = prepare_sampling_args_from_cli(args)
    
    # Prepare per-environment configuration
    env_args_dict = {}
    if args.per_env_config:
        for env in envs:
            if env in args.per_env_config:
                env_args_dict[env] = args.per_env_config[env].get("env_args", {})
    else:
        if any(env in args.env_args for env in envs):
            env_args_dict = args.env_args
        else:
            env_args_dict = {env: args.env_args for env in envs}
    
    # Prepare per-environment parameters
    num_examples_list = []
    rollouts_list = []
    max_concurrent_list = []
    sampling_args_dict = {}
    
    for env in envs:
        if args.per_env_config and env in args.per_env_config:
            cfg = args.per_env_config[env]
            num_examples_list.append(cfg.get("num_examples", args.num_examples))
            rollouts_list.append(cfg.get("rollouts_per_example", args.rollouts_per_example))
            max_concurrent_list.append(cfg.get("max_concurrent", args.max_concurrent))
            
            # Per-environment sampling args
            if "sampling_args" in cfg:
                sampling_args_dict[env] = cfg["sampling_args"]
        else:
            num_examples_list.append(args.num_examples)
            rollouts_list.append(args.rollouts_per_example)
            max_concurrent_list.append(args.max_concurrent)
    
    # Run evaluation (always use async path)
    logger.info(f"Evaluating {len(envs)} environment{'s' if len(envs) > 1 else ''}: {', '.join(envs)}")
    
    results_dict = asyncio.run(
        eval_environments_parallel(
            envs=envs,
            env_args_dict=env_args_dict,
            client=client,
            model=model,
            num_examples=num_examples_list,
            rollouts_per_example=rollouts_list,
            max_concurrent=max_concurrent_list,
            sampling_args=sampling_args,
            sampling_args_dict=sampling_args_dict if sampling_args_dict else None,
        )
    )
    
    # Display results
    if len(envs) > 1:
        logger.info("\n" + "="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
    
    for idx, (env, results) in enumerate(results_dict.items()):
        env_sampling_args = sampling_args_dict.get(env, sampling_args) if sampling_args_dict else sampling_args
        
        display_and_push_results(
            env=env,
            results=results,
            model=model,
            args=args,
            num_examples=num_examples_list[idx],
            rollouts_per_example=rollouts_list[idx],
            max_concurrent=max_concurrent_list[idx],
            sampling_args=env_sampling_args,
        )
    
    if len(envs) > 1:
        logger.info("\n" + "="*80)
        logger.info(f"✓ Completed evaluation of {len(envs)} environments")
        logger.info("="*80)
    else:
        logger.info(f"✓ Evaluation complete")


if __name__ == "__main__":
    main()
