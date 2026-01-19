import asyncio
import importlib.util
import json
import logging
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import cast

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

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
    EvalEnvConfig,
    EvalRunConfig,
    GenerateMetadata,
    GenerateOutputs,
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


class EvalTomlConfig(TypedDict):
    evals: list[dict]
    env_defaults: dict
    model_defaults: dict
    save_defaults: dict


def load_toml_config(path: Path) -> EvalTomlConfig:
    """Loads and validates a TOML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        raw_config = tomllib.load(f)

    eval_list = raw_config.get("eval")
    env_defaults = raw_config.get("env", {})
    model_defaults = raw_config.get("model", {})
    save_defaults = raw_config.get("save", {})

    if eval_list is None:
        env_list = raw_config.get("env", [])
        if isinstance(env_list, list) and env_list:
            eval_list = [{"env": env} for env in env_list]
            env_defaults = {}
        else:
            raise ValueError(
                f"Config file must contain at least one [[eval]] or [[env]] section: {path}"
            )

    if not isinstance(eval_list, list) or not eval_list:
        raise ValueError(
            f"Config file must contain at least one [[eval]] section: {path}"
        )

    valid_env_fields = set(EvalEnvConfig.model_fields.keys())
    valid_model_fields = {
        "model",
        "sampling_args",
        "max_concurrent",
        "max_concurrent_generation",
        "max_concurrent_scoring",
        "api_key_var",
        "api_base_url",
        "extra_headers",
        "client_config",
    }
    valid_save_fields = {
        "save_results",
        "save_every",
        "save_to_hf_hub",
        "hf_hub_dataset_name",
    }

    if env_defaults and not isinstance(env_defaults, dict):
        raise ValueError("[env] defaults must be a table")
    if model_defaults and not isinstance(model_defaults, dict):
        raise ValueError("[model] defaults must be a table")
    if save_defaults and not isinstance(save_defaults, dict):
        raise ValueError("[save] defaults must be a table")

    invalid_env_defaults = set(env_defaults.keys()) - valid_env_fields
    if invalid_env_defaults:
        raise ValueError(
            f"Invalid [env] default field(s) {invalid_env_defaults}. "
            f"Valid fields are: {sorted(valid_env_fields)}"
        )

    invalid_model_defaults = set(model_defaults.keys()) - valid_model_fields
    if invalid_model_defaults:
        raise ValueError(
            f"Invalid [model] default field(s) {invalid_model_defaults}. "
            f"Valid fields are: {sorted(valid_model_fields)}"
        )

    invalid_save_defaults = set(save_defaults.keys()) - valid_save_fields
    if invalid_save_defaults:
        raise ValueError(
            f"Invalid [save] default field(s) {invalid_save_defaults}. "
            f"Valid fields are: {sorted(valid_save_fields)}"
        )

    for eval_entry in eval_list:
        if not isinstance(eval_entry, dict):
            raise ValueError("Each [[eval]] entry must be a table")
        raw_env = eval_entry.get("env", {})
        raw_model = eval_entry.get("model", {})
        raw_save = eval_entry.get("save", {})
        if "env_id" in eval_entry:
            raw_env = {**raw_env, "env_id": eval_entry["env_id"]}

        if "env" not in eval_entry and "env_id" not in eval_entry:
            if "env_id" not in env_defaults:
                raise ValueError(
                    "Each [[eval]] entry must include an env section, "
                    "or env_id must be set in [env] defaults"
                )

        if "env_id" not in raw_env and "env_id" not in env_defaults:
            raise ValueError(
                "Each [[eval]] entry must include env.env_id, "
                "or env_id must be set in [env] defaults"
            )

        invalid_env_fields = set(raw_env.keys()) - (valid_env_fields | {"env_id"})
        if invalid_env_fields:
            raise ValueError(
                f"Invalid env field(s) {invalid_env_fields} for {raw_env.get('env_id', 'unknown')}. "
                f"Valid fields are: {sorted(valid_env_fields | {'env_id'})}"
            )

        invalid_model_fields = set(raw_model.keys()) - valid_model_fields
        if invalid_model_fields:
            raise ValueError(
                f"Invalid model field(s) {invalid_model_fields} for {raw_env.get('env_id', 'unknown')}. "
                f"Valid fields are: {sorted(valid_model_fields)}"
            )

        invalid_save_fields = set(raw_save.keys()) - valid_save_fields
        if invalid_save_fields:
            raise ValueError(
                f"Invalid save field(s) {invalid_save_fields} for {raw_env.get('env_id', 'unknown')}. "
                f"Valid fields are: {sorted(valid_save_fields)}"
            )

    return {
        "evals": eval_list,
        "env_defaults": env_defaults,
        "model_defaults": model_defaults,
        "save_defaults": save_defaults,
    }


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
    eval_config: EvalConfig, run_config: EvalRunConfig
) -> GenerateOutputs:
    # set up AsyncOpenAI client with high limits to prevent timeouts
    client = setup_client(eval_config.model.client_config)
    logger.debug(
        f"Initialized AsyncOpenAI client with base_url: {eval_config.model.client_config.api_base_url}"
    )

    # load environment
    vf_env = vf.load_environment(
        env_id=eval_config.env.env_id, **eval_config.env.env_args
    )

    # set extra environment kwargs
    if eval_config.env.extra_env_kwargs:
        logger.info(
            f"Setting extra environment kwargs: {eval_config.env.extra_env_kwargs}"
        )
        vf_env.set_kwargs(**eval_config.env.extra_env_kwargs)

    # run evaluation
    results_path = get_eval_results_path(run_config, eval_config)
    logger.info(f"Starting evaluation with model: {eval_config.model.model}")
    logger.info(
        "Configuration: num_examples=%s, rollouts_per_example=%s, max_concurrent=%s",
        eval_config.env.num_examples,
        eval_config.env.rollouts_per_example,
        eval_config.model.max_concurrent,
    )
    results = await vf_env.evaluate(
        client=client,
        model=eval_config.model.model,
        sampling_args=eval_config.model.sampling_args,
        num_examples=eval_config.env.num_examples,
        rollouts_per_example=eval_config.env.rollouts_per_example,
        max_concurrent=eval_config.model.max_concurrent,
        max_concurrent_generation=eval_config.model.max_concurrent_generation,
        max_concurrent_scoring=eval_config.model.max_concurrent_scoring,
        results_path=results_path,
        state_columns=eval_config.env.state_columns,
        save_results=eval_config.save.save_results,
        save_every=eval_config.save.save_every,
        independent_scoring=not eval_config.env.interleave_scoring,
    )

    if eval_config.save.save_results:
        save_rollout_results(
            results,
            eval_config.save.save_to_hf_hub,
            eval_config.save.hf_hub_dataset_name,
        )
    return results


async def run_evaluations(config: EvalRunConfig) -> None:
    # load event loop lag monitor
    event_loop_lag_monitor = EventLoopLagMonitor()
    event_loop_lag_monitor.run_in_background()

    start_time = time.time()
    all_results = await asyncio.gather(
        *[run_evaluation(eval_config, config) for eval_config in config.evals]
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
