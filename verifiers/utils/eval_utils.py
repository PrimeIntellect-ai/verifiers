import importlib.util
import json
import logging
import time
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
from datasets import Dataset, disable_progress_bar, enable_progress_bar
from datasets.utils import logging as ds_logging

from verifiers.errors import Error
from verifiers.types import (
    Endpoints,
    EvalConfig,
    GenerateMetadata,
    GenerateOutputs,
    RolloutResult,
)
from verifiers.utils.async_utils import EventLoopLagMonitor
from verifiers.utils.client_utils import setup_client
from verifiers.utils.error_utils import ErrorChain
from verifiers.utils.logging_utils import print_prompt_completions_sample, print_time
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls
from verifiers.utils.path_utils import get_eval_results_path
from verifiers.utils.type_utils import build_generate_outputs

logger = logging.getLogger(__name__)


def _coerce_errors(errors: list[str | None]) -> list[Error | None]:
    return [Error(e) if e is not None else None for e in errors]


def load_endpoints(endpoints_path: str):
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
            endpoints = cast(Endpoints, endpoints_module.ENDPOINTS)
            logger.debug(
                f"Successfully loaded {len(endpoints)} endpoints from registry"
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
        endpoints: Endpoints = {}
    return endpoints


def get_output_by_task(output: GenerateOutputs) -> dict[str, GenerateOutputs]:
    """Group output by task name."""
    rollouts = output["rollouts"]
    task_groups: dict[str, list[RolloutResult]] = {}
    for r in rollouts:
        task = r.get("task", "default")
        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append(r)

    return {
        task: GenerateOutputs(rollouts=group, metadata=output["metadata"])
        for task, group in task_groups.items()
    }


def print_rewards(output: GenerateOutputs):
    rollouts = output["rollouts"]
    rewards = [r.get("reward", 0.0) for r in rollouts]
    print("Rewards:")
    print(
        f"reward: avg - {sum(rewards) / len(rewards):.3f}, std - {np.std(rewards):.3f}"
    )

    rpe = output["metadata"]["rollouts_per_example"]
    n = len(rewards) // rpe
    for i in range(rpe):
        trials = [round(rewards[i + (j * rpe)], 3) for j in range(n)]
        print(f"r{i + 1}: {trials}")

    # Aggregate metrics
    metrics: dict[str, list[float]] = {}
    for r in rollouts:
        if r.get("metrics"):
            for k, v in r["metrics"].items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

    for k, v in metrics.items():
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
        for i in range(rpe):
            trials = [round(v[i + (j * rpe)], 3) for j in range(n)]
            print(f"r{i + 1}: {trials}")


def print_info(output: GenerateOutputs):
    rollouts = output["rollouts"]
    is_truncated = [r.get("is_truncated", False) for r in rollouts]
    stop_conditions = [r.get("stop_condition") for r in rollouts]
    errors = _coerce_errors([r.get("error") for r in rollouts])

    print("Info:")
    print(
        f"is_truncated: avg - {np.mean(is_truncated):.3f}, std - {np.std(is_truncated):.3f}"
    )

    counter = Counter(stop_conditions)
    print(
        f"stop_conditions: {', '.join([f'{k}: {v / counter.total():.3f}' for k, v in counter.items()])}"
    )

    has_errors = [e is not None for e in errors]
    if any(has_errors):
        print(
            f"errors: avg - {np.mean(has_errors):.3f}, std - {np.std(has_errors):.3f}"
        )
        error_objs = [e for e in errors if e is not None]
        error_chains = [ErrorChain(e) for e in error_objs]
        counter = Counter(error_chains)
        for error_chain, count in counter.items():
            print(f" - {repr(error_chain)}: {count / counter.total():.3f}")


def print_timing(output: GenerateOutputs):
    rollouts = output["rollouts"]
    generation_ms = [r.get("timing", {}).get("generation_ms", 0.0) for r in rollouts]
    scoring_ms = [r.get("timing", {}).get("scoring_ms", 0.0) for r in rollouts]
    total_ms = [r.get("timing", {}).get("total_ms", 0.0) for r in rollouts]

    generation_arr = np.array(generation_ms) / 1000
    scoring_arr = np.array(scoring_ms) / 1000
    total_arr = np.array(total_ms) / 1000

    print("Timing:")
    print(
        f"generation: min - {print_time(float(np.min(generation_arr)))}, mean - {print_time(float(np.mean(generation_arr)))}, max - {print_time(float(np.max(generation_arr)))}"
    )
    print(
        f"scoring: min - {print_time(float(np.min(scoring_arr)))}, mean - {print_time(float(np.mean(scoring_arr)))}, max - {print_time(float(np.max(scoring_arr)))}"
    )
    print(
        f"total: min - {print_time(float(np.min(total_arr)))}, mean - {print_time(float(np.mean(total_arr)))}, max - {print_time(float(np.max(total_arr)))}"
    )


def print_results(
    output: GenerateOutputs,
    event_loop_lags: list[float] | None = None,
    num_samples: int = 1,
):
    rollouts = output["rollouts"]
    metadata = output["metadata"]

    print("--- Evaluation ---")
    print(f"Environment: {metadata['env_id']}")
    print(f"Model: {metadata['model']}")
    print(f"Provider: {metadata['base_url']}")
    print(f"Examples: {metadata['num_examples']}")
    print(f"Rollouts per example: {metadata['rollouts_per_example']}")
    print("--- Example ---")

    printable_prompts = [messages_to_printable(r["prompt"]) for r in rollouts]
    printable_completions = [
        messages_to_printable(r["completion"]) if r.get("completion") else ""
        for r in rollouts
    ]
    errors = _coerce_errors([r.get("error") for r in rollouts])
    rewards = [r.get("reward", 0.0) for r in rollouts]
    print_prompt_completions_sample(
        printable_prompts,
        printable_completions,
        errors,
        rewards,
        step=0,
        num_samples=num_samples,
    )
    print("--- All ---")
    print_rewards(output)
    print_info(output)
    print_timing(output)

    tasks = [r.get("task", "default") for r in rollouts]
    num_tasks = len(set(tasks))
    if num_tasks > 1:
        task_outputs = get_output_by_task(output)
        for task, task_output in task_outputs.items():
            print(f"\n--- {task} ---")
            print_rewards(task_output)
            print_info(task_output)
            print_timing(task_output)

    if event_loop_lags:
        print("\nPerformance:")
        event_loop_lags_arr = np.array(event_loop_lags)
        med_lag, p90_lag, max_lag = (
            np.median(event_loop_lags_arr),
            np.percentile(event_loop_lags_arr, 90),
            np.max(event_loop_lags_arr),
        )
        print(
            f"event_loop_lag: med - {print_time(float(med_lag))}, p90 - {print_time(float(p90_lag))}, max - {print_time(float(max_lag))}"
        )


async def run_evaluation(config: EvalConfig) -> GenerateOutputs:
    """
    Run evaluation using worker subprocesses.

    Workers call load_environment in their own subprocess, so each worker
    owns its own environment instance. This enables multiprocessing and
    proper isolation.
    """
    from tqdm import tqdm

    from verifiers.workers.client import EnvClient

    # set up AsyncOpenAI client with high limits to prevent timeouts
    client = setup_client(
        config.client_config,
    )
    logger.debug(
        f"Initialized AsyncOpenAI client with base_url: {config.client_config.api_base_url}"
    )

    # load event loop lag monitor
    event_loop_lag_monitor = EventLoopLagMonitor()
    event_loop_lag_monitor.run_in_background()

    results_path = get_eval_results_path(config)
    logger.info(f"Starting evaluation with model: {config.model}")
    logger.info(
        f"Configuration: num_examples={config.num_examples}, rollouts_per_example={config.rollouts_per_example}, "
        f"max_concurrent={config.max_concurrent}, num_workers={config.num_workers}"
    )

    start_time = time.time()

    # Create client - workers will load environment and dataset
    env_client = EnvClient(
        env_id=config.env_id,
        env_args={**config.env_args, **config.extra_env_kwargs},
        client_base_url=str(client.base_url),
        client_api_key=client.api_key or "",
        num_workers=config.num_workers,
        max_concurrent=config.max_concurrent,
        sampling_args=config.sampling_args or {},
        independent_scoring=config.independent_scoring,
        state_columns=config.state_columns,
    )

    try:
        # Start workers - first worker loads env and returns dataset
        env_client.start(num_examples=config.num_examples)

        # Get dataset from client (fetched from first worker)
        dataset = env_client.dataset
        num_examples = env_client.num_examples
        total_rollouts = num_examples * config.rollouts_per_example

        # Use env's sampling args as base, override with config
        gen_sampling_args = deepcopy(env_client.env_sampling_args)
        if config.sampling_args:
            gen_sampling_args.update(config.sampling_args)
        env_client.update_sampling_args(gen_sampling_args)

        pbar = tqdm(
            total=num_examples,
            desc=f"Processing {num_examples} groups ({total_rollouts} total rollouts)",
            postfix=dict(reward="?"),
        )

        all_results: list[RolloutResult] = []
        reward_sum, reward_count = 0, 0

        try:
            # Construct groups explicitly: (group_inputs, example_id)
            # Orchestrator controls duplication via rollouts_per_example
            groups: list[tuple[list, int]] = []
            for example in dataset.to_list():
                example_id = example.get("example_id", 0)
                group_inputs = [example] * config.rollouts_per_example
                groups.append((group_inputs, example_id))

            results_list = await env_client.run_groups(
                groups=groups,
                model_name=config.model,
            )

            for group_results in results_list:
                for result in group_results:
                    all_results.append(result)

                    r = result.get("reward")
                    if r is not None:
                        reward_sum += r
                        reward_count += 1

                pbar.update(1)
                if reward_count > 0:
                    pbar.set_postfix(reward=f"{reward_sum / reward_count:.3f}")
        finally:
            pbar.close()

    finally:
        env_client.stop()

    # sort by example_id to ensure deterministic ordering
    all_results.sort(key=lambda s: s.get("example_id", 0))

    # build output structure
    output = build_generate_outputs(
        rollouts=all_results,
        env_id=config.env_id,
        env_args=config.env_args,
        model=config.model,
        base_url=str(client.base_url) if hasattr(client, "base_url") else "",
        sampling_args=gen_sampling_args,
        start_time=start_time,
        path_to_save=results_path,
        state_columns=config.state_columns,
    )

    end_time = time.time()
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

    event_loop_lags = event_loop_lag_monitor.get_lags()

    if config.print_results:
        print_results(output, event_loop_lags)
    if config.save_results:
        save_rollout_results(
            output,
            path=results_path,
            push_to_hf_hub=config.save_to_hf_hub,
            hf_hub_dataset_name=config.hf_hub_dataset_name,
        )
    return output


def sanitize_metadata(metadata: GenerateMetadata) -> dict:
    metadata_dict = dict(metadata)
    metadata_dict.pop("date", None)
    path_to_save = metadata_dict.get("path_to_save")
    if isinstance(path_to_save, Path):
        metadata_dict["path_to_save"] = str(path_to_save)
    return metadata_dict


def get_hf_hub_dataset_name(results: GenerateOutputs) -> str:
    metadata = results["metadata"]
    dataset_name = (
        metadata["env_id"]
        + "_"
        + metadata["model"].replace("/", "_")
        + "_n"
        + str(metadata["num_examples"])
        + "_r"
        + str(metadata["rollouts_per_example"])
    )
    return dataset_name


def make_dataset(output: GenerateOutputs) -> Dataset:
    """Convert GenerateOutputs to a Dataset for saving."""
    rollouts = output["rollouts"]

    clean_prompts = [messages_to_printable(r["prompt"]) for r in rollouts]
    clean_prompts = [sanitize_tool_calls(p) for p in clean_prompts]
    clean_completions = [
        messages_to_printable(r["completion"]) if r.get("completion") else ""
        for r in rollouts
    ]
    clean_completions = [sanitize_tool_calls(c) for c in clean_completions]

    infos = [r.get("info", {}) for r in rollouts]
    answers = [r.get("answer", "") for r in rollouts]
    stop_conditions = [r.get("stop_condition") for r in rollouts]
    is_truncated = [r.get("is_truncated", False) for r in rollouts]

    results_dict: dict = {
        "example_id": [r.get("example_id", 0) for r in rollouts],
        "prompt": clean_prompts,
        "completion": clean_completions,
        "task": [r.get("task", "default") for r in rollouts],
        "reward": [r.get("reward", 0.0) for r in rollouts],
        "error": [r.get("error") for r in rollouts],
        "stop_condition": stop_conditions,
        "is_truncated": is_truncated,
        "generation_ms": [
            r.get("timing", {}).get("generation_ms", 0.0) for r in rollouts
        ],
        "scoring_ms": [r.get("timing", {}).get("scoring_ms", 0.0) for r in rollouts],
        "total_ms": [r.get("timing", {}).get("total_ms", 0.0) for r in rollouts],
    }

    if any(info != {} for info in infos):
        results_dict["info"] = infos
    if any(answer != "" for answer in answers):
        results_dict["answer"] = answers

    # Aggregate metrics from rollouts
    for r in rollouts:
        if r.get("metrics"):
            for k, v in r["metrics"].items():
                if k not in results_dict:
                    results_dict[k] = []
                results_dict[k].append(v)

    return Dataset.from_dict(results_dict)


@contextmanager
def quiet_datasets():
    prev_level = ds_logging.get_verbosity()
    ds_logging.set_verbosity(ds_logging.WARNING)
    disable_progress_bar()
    try:
        yield
    finally:
        ds_logging.set_verbosity(prev_level)
        enable_progress_bar()


def save_to_disk(dataset: Dataset, metadata_dict: dict, path_to_save: Path):
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    with quiet_datasets():
        dataset.to_json(path_to_save / "results.jsonl")
    with open(path_to_save / "metadata.json", "w") as f:
        json.dump(metadata_dict, f)


def save_rollout_results(
    output: GenerateOutputs,
    path: Path,
    push_to_hf_hub: bool = False,
    hf_hub_dataset_name: str | None = None,
):
    """Save GenerateOutputs to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(output)
    metadata_dict = sanitize_metadata(output["metadata"])
    save_to_disk(dataset, metadata_dict, path)
    logger.info(f"Results saved to {path}")
    if push_to_hf_hub:
        dataset_name = hf_hub_dataset_name or get_hf_hub_dataset_name(output)
        dataset.push_to_hub(dataset_name)
        logger.info(f"Dataset saved to Hugging Face Hub: {dataset_name}")
