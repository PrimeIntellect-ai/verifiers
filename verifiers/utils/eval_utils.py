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
    EvalRunConfig,
    GenerateMetadata,
    GenerateOutputs,
    LogCallback,
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


def load_toml_config(path: Path) -> list[dict]:
    """Loads and validates a TOML config file.

    Config format supports global defaults at the top level, with per-eval overrides:

        # Global defaults (optional)
        model = "openai/gpt-4.1-mini"
        num_examples = 10

        [[eval]]
        env_id = "gsm8k"

        [[eval]]
        env_id = "math-python"
        num_examples = 5  # overrides global default

    Minimal config (just a single eval):

        [[eval]]
        env_id = "gsm8k"
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        raw_config = tomllib.load(f)

    # validate schema
    eval_list = raw_config.get("eval", [])
    if not isinstance(eval_list, list):
        raise ValueError(
            f"Config file uses [eval] but should use [[eval]] (double brackets) "
            f"for array of tables: {path}"
        )
    if not eval_list:
        raise ValueError(
            f"Config file must contain at least one [[eval]] section: {path}"
        )

    if not all("env_id" in e for e in eval_list):
        raise ValueError(f"All [[eval]] sections must contain an env_id field: {path}")

    # extract global defaults (everything except 'eval' key)
    global_defaults = {k: v for k, v in raw_config.items() if k != "eval"}

    valid_fields = set(EvalConfig.model_fields.keys())

    # validate global fields
    if global_defaults:
        invalid_global = set(global_defaults.keys()) - valid_fields
        if invalid_global:
            raise ValueError(
                f"Invalid global field(s) {invalid_global}. "
                f"Valid fields are: {sorted(valid_fields)}"
            )

    # merge global defaults with per-eval configs
    merged_eval_list: list[dict] = []
    for eval_config in eval_list:
        invalid_fields = set(eval_config.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(
                f"Invalid field(s) {invalid_fields} for {eval_config.get('env_id', 'unknown')}. "
                f"Valid fields are: {sorted(valid_fields)}"
            )
        # global defaults, then per-eval overrides
        merged = {**global_defaults, **eval_config}
        merged_eval_list.append(merged)

    return merged_eval_list


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
    logger.debug(f"Starting evaluation with model: {config.model}")
    logger.debug(
        f"Configuration: num_examples={config.num_examples}, rollouts_per_example={config.rollouts_per_example}, max_concurrent={config.max_concurrent}"
    )
    # disable tqdm when callbacks are provided (TUI handles progress display)
    use_tqdm = config.use_tqdm and on_progress is None
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
        push_to_hf_hub=config.save_to_hf_hub,
        hf_hub_dataset_name=config.hf_hub_dataset_name,
        use_tqdm=use_tqdm,
        independent_scoring=config.independent_scoring,
        max_retries=config.max_retries,
        on_start=on_start,
        on_progress=on_progress,
        on_log=on_log,
    )

    return results


async def run_evaluations(config: EvalRunConfig) -> None:
    # load event loop lag monitor
    event_loop_lag_monitor = EventLoopLagMonitor()
    event_loop_lag_monitor.run_in_background()

    start_time = time.time()
    all_results = await asyncio.gather(
        *[run_evaluation(eval_config) for eval_config in config.evals]
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


async def run_evaluations_tui(config: EvalRunConfig, tui_mode: bool = True) -> None:
    """Run multi-environment evaluation with a Rich display.

    Args:
        config: Evaluation run configuration.
        tui_mode: If True, use alternate screen (--tui flag). If False, refresh in-place.
    """
    from verifiers.utils.eval_display import EvalDisplay, is_tty

    # fall back to non-display mode if not a tty
    if not is_tty():
        logger.debug("Not a TTY, falling back to standard output")
        await run_evaluations(config)
        return

    display = EvalDisplay(config.evals, screen=tui_mode)

    async def run_with_progress(
        env_config: EvalConfig, env_idx: int
    ) -> GenerateOutputs:
        """Run a single evaluation with display progress updates."""
        reward_accum = 0
        metrics_accum = defaultdict(float)
        error_accum = 0

        def on_start(total: int) -> None:
            # total is num_examples * rollouts_per_example
            # compute actual num_examples (resolves -1 to actual count)
            num_examples = total // env_config.rollouts_per_example
            display.update_env_state(env_idx, total=total, num_examples=num_examples)

        def on_progress(all_states: list[State], new_states: list[State]) -> None:
            nonlocal error_accum, reward_accum, metrics_accum

            # Progress is always rollout-based
            completed = len(all_states)

            for s in new_states:
                if s.get("error") is not None:
                    error_accum += 1
                reward = s.get("reward")
                if reward is not None:
                    reward_accum += reward
                state_metrics = s.get("metrics") or {}
                for name, value in state_metrics.items():
                    if value is not None:
                        metrics_accum[name] += value

            # Compute averages over completed rollouts
            reward = reward_accum / completed
            metrics = {name: metrics_accum[name] / completed for name in metrics_accum}
            error_rate = error_accum / completed

            display.update_env_state(
                env_idx,
                progress=completed,
                reward=reward,
                metrics=metrics,
                error_rate=error_rate,
            )

        def on_log(message: str) -> None:
            display.update_env_state(env_idx, log_message=message)

        display.update_env_state(env_idx, status="running")
        try:
            result = await run_evaluation(
                env_config,
                on_start=on_start,
                on_progress=on_progress,
                on_log=on_log,
            )

            # get save path if results were saved
            save_path = (
                result["metadata"]["path_to_save"] if env_config.save_results else None
            )

            display.update_env_state(
                env_idx,
                status="completed",
                save_path=save_path,
                results=result,
            )

            return result
        except Exception as e:
            display.update_env_state(env_idx, status="failed", error=str(e))
            raise

    try:
        async with display:
            await asyncio.gather(
                *[
                    run_with_progress(env_config, idx)
                    for idx, env_config in enumerate(config.evals)
                ],
                return_exceptions=True,
            )

            display.refresh()
            if tui_mode:
                await display.wait_for_exit()

    except KeyboardInterrupt:
        pass  # exit on interrupt

    # print final summary after exit
    display.print_final_summary()


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
