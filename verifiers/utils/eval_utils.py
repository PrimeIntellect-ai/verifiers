import asyncio
import importlib.util
import json
import logging
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import cast

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]

import numpy as np
from datasets import Dataset, disable_progress_bar, enable_progress_bar
from datasets.utils import logging as ds_logging

import verifiers as vf
from verifiers.types import (
    Endpoints,
    EvalConfig,
    GenerateMetadata,
    GenerateOutputs,
    LogCallback,
    MultiEvalConfig,
    ProgressCallback,
    StartCallback,
    State,
)
from verifiers.utils.async_utils import EventLoopLagMonitor
from verifiers.utils.client_utils import setup_client
from verifiers.utils.error_utils import ErrorChain
from verifiers.utils.logging_utils import print_prompt_completions_sample, print_time
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls
from verifiers.utils.path_utils import get_eval_results_path

logger = logging.getLogger(__name__)


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


def is_toml_config(path: str) -> bool:
    """Checks if a path is a valid TOML config file."""
    return Path(path).is_file() and Path(path).suffix == ".toml"


def load_toml_config(path: Path) -> list[dict]:
    """Loads and validates a TOML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        raw_config = tomllib.load(f)

    # validate schema
    env_list: list[dict] = raw_config.get("env", [])
    if not env_list:
        raise ValueError(
            f"Config file must contain at least one [[env]] section: {path}"
        )

    if not all("env_id" in e for e in env_list):
        raise ValueError(f"All [[env]] sections must contain an env_id field: {path}")

    valid_fields = set(EvalConfig.model_fields.keys())
    for env in env_list:
        invalid_fields = set(env.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(
                f"Invalid field(s) {invalid_fields} for {env.get('env_id', 'unknown')}. "
                f"Valid fields are: {sorted(valid_fields)}"
            )

    return env_list


def get_results_by_task(results: GenerateOutputs) -> dict[str, GenerateOutputs]:
    """Group results by task name.

    Args:
        results: The GenerateOutputs from an evaluation run.

    Returns:
        A dictionary mapping task names to their corresponding GenerateOutputs.
    """
    task_indices: dict[str, list[int]] = {}
    for i, task in enumerate(results["task"]):
        if task not in task_indices:
            task_indices[task] = []
        task_indices[task].append(i)

    task_results: dict[str, GenerateOutputs] = {}
    for task, indices in task_indices.items():
        task_results[task] = GenerateOutputs(
            prompt=[results["prompt"][i] for i in indices],
            completion=[results["completion"][i] for i in indices],
            answer=[results["answer"][i] for i in indices],
            state=[results["state"][i] for i in indices],
            task=[results["task"][i] for i in indices],
            info=[results["info"][i] for i in indices],
            example_id=[results["example_id"][i] for i in indices],
            reward=[results["reward"][i] for i in indices],
            metrics={k: [v[i] for i in indices] for k, v in results["metrics"].items()},
            stop_conditions=[results["stop_conditions"][i] for i in indices],
            is_truncated=[results["is_truncated"][i] for i in indices],
            metadata=results["metadata"],
        )
    return task_results


def print_rewards(results: GenerateOutputs):
    print("Rewards:")
    print(
        f"reward: avg - {sum(results['reward']) / len(results['reward']):.3f}, std - {np.std(results['reward']):.3f}"
    )
    r = results["metadata"]["rollouts_per_example"]
    n = len(results["reward"]) // r
    # results are sorted by example_id, so rollout i is at indices [i, i+r, i+2r, ...]
    for i in range(r):
        trials = [round(results["reward"][i + (j * r)], 3) for j in range(n)]
        out = f"r{i + 1}: {trials}"
        print(out)
    for k in results["metrics"]:
        v = results["metrics"][k]
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
        for i in range(r):
            trials = [round(v[i + (j * r)], 3) for j in range(n)]
            out = f"r{i + 1}: {trials}"
            print(out)


def print_info(results: GenerateOutputs):
    print("Info:")
    print(
        f"is_truncated: avg - {np.mean(results['is_truncated']):.3f}, std - {np.std(results['is_truncated']):.3f}"
    )
    counter = Counter(results["stop_conditions"])
    print(
        f"stop_conditions: {', '.join([f'{k}: {v / counter.total():.3f}' for k, v in counter.items()])}"
    )
    errors = [s.get("error") for s in results["state"]]
    has_errors = [e is not None for e in errors]
    if any(has_errors):
        print(
            f"errors: avg - {np.mean(has_errors):.3f}, std - {np.std(has_errors):.3f}"
        )
        errors = [e for e in errors if e is not None]
        error_chains = [ErrorChain(e) for e in errors]
        counter = Counter(error_chains)
        for error_chain, count in counter.items():
            print(f" - {repr(error_chain)}: {count / counter.total():.3f}")


def print_timing(results: GenerateOutputs):
    print("Timing:")
    generation_ms_arr = np.array(
        [s["timing"]["generation_ms"] for s in results["state"]]
    )
    scoring_ms_arr = np.array([s["timing"]["scoring_ms"] for s in results["state"]])
    total_ms_arr = np.array([s["timing"]["total_ms"] for s in results["state"]])
    generation_arr = generation_ms_arr / 1000
    scoring_arr = scoring_ms_arr / 1000
    total_arr = total_ms_arr / 1000

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
    results: GenerateOutputs,
    num_samples: int = 1,
):
    assert results["metadata"] is not None
    print("--- Evaluation ---")
    print(f"Environment: {results['metadata']['env_id']}")
    print(f"Model: {results['metadata']['model']}")
    print(f"Provider: {results['metadata']['base_url']}")
    print(f"Examples: {results['metadata']['num_examples']}")
    print(f"Rollouts per example: {results['metadata']['rollouts_per_example']}")
    print("--- Example ---")

    printable_prompts = [messages_to_printable(p) for p in results["prompt"]]
    printable_completions = [messages_to_printable(c) for c in results["completion"]]
    errors = [s.get("error") for s in results["state"]]
    print_prompt_completions_sample(
        printable_prompts,
        printable_completions,
        errors,
        results["reward"],
        step=0,
        num_samples=num_samples,
    )
    print("--- All ---")
    print_rewards(results)
    print_info(results)
    print_timing(results)

    num_tasks = len(set(results["task"]))
    if num_tasks > 1:
        task_results = get_results_by_task(results)
        for task, task_results in task_results.items():
            print(f"\n--- {task} ---")
            print_rewards(task_results)
            print_info(task_results)
            print_timing(task_results)


async def run_evaluation(
    config: EvalConfig,
    on_start: StartCallback | None = None,
    on_progress: ProgressCallback | None = None,
    on_log: LogCallback | None = None,
) -> GenerateOutputs:
    # set up AsyncOpenAI client with high limits to prevent timeouts
    client = setup_client(config.client_config)
    logger.debug(
        f"Initialized AsyncOpenAI client with base_url: {config.client_config.api_base_url}"
    )

    # load environment
    vf_env = vf.load_environment(env_id=config.env_id, **config.env_args)

    # set extra environment kwargs
    if config.extra_env_kwargs:
        logger.info(f"Setting extra environment kwargs: {config.extra_env_kwargs}")
        vf_env.set_kwargs(**config.extra_env_kwargs)

    # run evaluation
    results_path = get_eval_results_path(config)
    logger.info(f"Starting evaluation with model: {config.model}")
    logger.info(
        f"Configuration: num_examples={config.num_examples}, rollouts_per_example={config.rollouts_per_example}, max_concurrent={config.max_concurrent}"
    )
    results = await vf_env.evaluate(
        client=client,
        model=config.model,
        sampling_args=config.sampling_args,
        num_examples=config.num_examples,
        rollouts_per_example=config.rollouts_per_example,
        max_concurrent=config.max_concurrent,
        max_concurrent_generation=config.max_concurrent_generation,
        max_concurrent_scoring=config.max_concurrent_scoring,
        results_path=results_path,
        state_columns=config.state_columns,
        save_results=config.save_results,
        save_every=config.save_every,
        independent_scoring=config.independent_scoring,
        on_start=on_start,
        on_progress=on_progress,
        on_log=on_log,
    )

    if config.save_results:
        save_rollout_results(results, config.save_to_hf_hub, config.hf_hub_dataset_name)
    return results


def compute_metrics_from_states(states: list[State]) -> dict[str, float]:
    """Compute aggregated metrics from a list of states.

    Args:
        states: List of State objects from completed rollouts.

    Returns:
        Dictionary of metric names to their average values.
    """
    if not states:
        return {}

    metrics_sums: dict[str, float] = {}
    metrics_counts: dict[str, int] = {}
    error_count = 0

    for s in states:
        # Track errors
        if s.get("error") is not None:
            error_count += 1
        # Track top-level reward
        reward = s.get("reward")
        if reward is not None:
            metrics_sums["reward"] = metrics_sums.get("reward", 0.0) + reward
            metrics_counts["reward"] = metrics_counts.get("reward", 0) + 1
        # Track all other metrics
        state_metrics = s.get("metrics", {})
        for name, value in state_metrics.items():
            if value is not None:
                metrics_sums[name] = metrics_sums.get(name, 0.0) + value
                metrics_counts[name] = metrics_counts.get(name, 0) + 1

    # Compute averages
    avg_metrics: dict[str, float] = {}
    for name, total in metrics_sums.items():
        count = metrics_counts.get(name, 0)
        if count > 0:
            avg_metrics[name] = total / count

    # Add error rate as a special metric
    if states:
        avg_metrics["error_rate"] = error_count / len(states)

    return avg_metrics


async def run_multi_evaluation(config: MultiEvalConfig) -> None:
    # load event loop lag monitor
    event_loop_lag_monitor = EventLoopLagMonitor()
    event_loop_lag_monitor.run_in_background()

    async def run_with_progress(env_config: EvalConfig) -> GenerateOutputs:
        if env_config.use_tqdm:
            from tqdm import tqdm

            # Delay pbar creation until first callback (when generate actually starts)
            pbar: tqdm | None = None
            reward_accum = 0

            def tqdm_callback(all_states: list[State], new_states: list[State]) -> None:
                nonlocal reward_accum
                nonlocal pbar

                completed = len(all_states)
                if pbar is None:
                    # Create pbar on first callback
                    if env_config.independent_scoring:
                        pbar_total = (
                            env_config.num_examples * env_config.rollouts_per_example
                        )
                        pbar_desc = (
                            f"Processing {pbar_total} {env_config.env_id} rollouts"
                        )
                    else:
                        pbar_total = env_config.num_examples
                        pbar_desc = f"Processing {pbar_total} {env_config.env_id} groups ({env_config.num_examples * env_config.rollouts_per_example} total rollouts)"
                    pbar = tqdm(
                        total=pbar_total, desc=pbar_desc, postfix=dict(reward="?")
                    )
                pbar.update()

                for s in new_states:
                    reward = s.get("reward")
                    if reward is not None:
                        reward_accum += reward

                pbar.set_postfix(reward=f"{reward_accum / completed:.3f}")

            on_progress = tqdm_callback
        else:
            on_progress = None
        result = await run_evaluation(env_config, on_progress=on_progress)
        return result

    start_time = time.time()
    all_results = await asyncio.gather(
        *[run_with_progress(env_config) for env_config in config.env]
    )
    end_time = time.time()
    event_loop_lags = event_loop_lag_monitor.get_lags()
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

    for results in all_results:
        print_results(results)

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


async def run_multi_evaluation_tui(config: MultiEvalConfig) -> list[GenerateOutputs]:
    """Run multi-environment evaluation with a TUI display."""

    from verifiers.utils.eval_tui import EvalTUI, is_tty

    # Fall back to non-TUI mode if not a TTY
    if not is_tty():
        logger.info("Not a TTY, falling back to standard output")
        await run_multi_evaluation(config)
        return []

    # critical level to suppress loggin
    with vf.log_level(logging.CRITICAL):
        # Create modified configs with tqdm disabled (TUI handles progress display)
        tui_configs = [
            env_config.model_copy(update={"use_tqdm": False})
            for env_config in config.env
        ]

        tui = EvalTUI(config.env)

        async def run_with_updates(env_config: EvalConfig) -> GenerateOutputs:
            """Run a single evaluation with TUI progress updates."""
            env_id = env_config.env_id

            reward_accum = 0
            metrics_accum = defaultdict(float)
            error_accum = 0

            def on_start(total: int) -> None:
                tui.update_env_state(env_id, total=total)

            def on_progress(all_states: list[State], new_states: list[State]) -> None:
                nonlocal error_accum, reward_accum, metrics_accum

                # Compute progress as number of groups or rollouts completed
                if env_config.independent_scoring:
                    completed = len(all_states)
                else:
                    # Count unique example_ids to get number of groups
                    completed = len(set(s.get("example_id", 0) for s in all_states))

                for s in new_states:
                    if s.get("error") is not None:
                        error_accum += 1
                    reward = s.get("reward")
                    if reward is not None:
                        reward_accum += reward
                    state_metrics = s.get("metrics", {})
                    for name, value in state_metrics.items():
                        if value is not None:
                            metrics_accum[name] += value

                reward = reward_accum / completed
                metrics = {
                    name: metrics_accum[name] / completed for name in metrics_accum
                }
                error_rate = error_accum / completed

                tui.update_env_state(
                    env_id,
                    progress=completed,
                    reward=reward,
                    metrics=metrics,
                    error_rate=error_rate,
                )

            def on_log(message: str) -> None:
                tui.update_env_state(env_id, log_message=message)

            tui.update_env_state(env_id, status="running")
            try:
                result = await run_evaluation(
                    env_config,
                    on_start=on_start,
                    on_progress=on_progress,
                    on_log=on_log,
                )

                # Update final state from results with all metrics
                rewards = result["reward"]
                final_metrics: dict[str, float] = {}
                if rewards:
                    final_metrics["reward"] = sum(rewards) / len(rewards)
                # Add all other metrics from result
                for name, values in result["metrics"].items():
                    if values:
                        final_metrics[name] = sum(values) / len(values)
                # Compute final error_rate
                errors = [s.get("error") for s in result["state"]]
                has_errors = [e is not None for e in errors]
                if has_errors:
                    final_metrics["error_rate"] = sum(has_errors) / len(has_errors)

                # Get save path if results were saved
                save_path = (
                    result["metadata"]["path_to_save"]
                    if env_config.save_results
                    else None
                )

                # Calculate actual total from result metadata
                actual_total = (
                    result["metadata"]["num_examples"]
                    * result["metadata"]["rollouts_per_example"]
                )

                tui.update_env_state(
                    env_id,
                    status="completed",
                    total=actual_total,
                    progress=actual_total,  # Mark as fully complete
                    metrics=final_metrics,
                    save_path=save_path,
                )

                return result
            except Exception as e:
                tui.update_env_state(env_id, status="failed", error=str(e))
                raise

        # Run evaluations with TUI
        all_results: list[GenerateOutputs] = []
        try:
            async with tui:
                # Run all evaluations concurrently (using tui_configs with tqdm disabled)
                results = await asyncio.gather(
                    *[run_with_updates(env_config) for env_config in tui_configs],
                    return_exceptions=True,
                )

                # Process results (silently, errors shown in TUI)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        # Error already shown in TUI via update_env_state
                        pass
                    else:
                        all_results.append(result)

                # Refresh to show completion state
                tui.refresh()

                # Wait for user to exit
                await tui.wait_for_exit()

        except KeyboardInterrupt:
            pass  # Silent exit on interrupt

        # Print final summary after TUI exits
        tui.print_final_summary()

        return all_results


def sanitize_metadata(metadata: GenerateMetadata) -> dict:
    metadata_dict = dict(metadata)
    metadata_dict.pop("path_to_save")
    metadata_dict.pop("date")

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


def make_dataset(results: GenerateOutputs, **kwargs) -> Dataset:
    clean_prompts = [messages_to_printable(p) for p in results["prompt"]]
    clean_prompts = [sanitize_tool_calls(p) for p in clean_prompts]
    clean_completions = [messages_to_printable(c) for c in results["completion"]]
    clean_completions = [sanitize_tool_calls(c) for c in clean_completions]
    save_info = any(info != {} for info in results["info"])
    save_answer = any(answer != "" for answer in results["answer"])
    errors = [s.get("error") for s in results["state"]]
    results_dict = {
        "example_id": results["example_id"],
        "prompt": clean_prompts,
        "completion": clean_completions,
        "task": results["task"],
        "reward": results["reward"],
        "error": [repr(e) if e is not None else None for e in errors],
        "generation_ms": [s["timing"]["generation_ms"] for s in results["state"]],
        "scoring_ms": [s["timing"]["scoring_ms"] for s in results["state"]],
        "total_ms": [s["timing"]["total_ms"] for s in results["state"]],
    }
    if save_info:
        results_dict["info"] = results["info"]
    if save_answer:
        results_dict["answer"] = results["answer"]
    for k in results["metrics"]:
        v = results["metrics"][k]
        results_dict[k] = v

    # Add selected state columns if specified
    state_columns = results["metadata"]["state_columns"]
    if state_columns:
        for col in state_columns:
            if col == "responses":
                results_dict[col] = [
                    [r.model_dump() for r in s.get(col, [])] for s in results["state"]
                ]
            else:
                results_dict[col] = [s.get(col) for s in results["state"]]

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
    results: GenerateOutputs,
    push_to_hf_hub: bool = False,
    hf_hub_dataset_name: str | None = None,
):
    path_to_save = results["metadata"]["path_to_save"]
    path_to_save.parent.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(results)
    metadata_dict = sanitize_metadata(results["metadata"])
    save_to_disk(dataset, metadata_dict, path_to_save)
    logger.info(f"Results saved to {path_to_save}")
    if push_to_hf_hub:
        dataset_name = hf_hub_dataset_name or get_hf_hub_dataset_name(results)
        dataset.push_to_hub(dataset_name)
        logger.info(f"Dataset saved to Hugging Face Hub: {dataset_name}")
