import importlib.util
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import numpy as np
from datasets import Dataset, disable_progress_bar, enable_progress_bar
from datasets.utils import logging as ds_logging

import verifiers as vf
from verifiers.types import Endpoints, EvalConfig, GenerateMetadata, GenerateOutputs
from verifiers.utils.client_utils import setup_client
from verifiers.utils.logging_utils import print_prompt_completions_sample
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


def print_results(results: GenerateOutputs, num_samples: int = 1):
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
    print_prompt_completions_sample(
        printable_prompts,
        printable_completions,
        results["reward"],
        step=0,
        num_samples=num_samples,
    )
    print("--- All ---")
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


async def run_evaluation(config: EvalConfig) -> GenerateOutputs:
    # set up AsyncOpenAI client with high limits to prevent timeouts
    client = setup_client(
        config.client_config,
    )
    logger.debug(
        f"Initialized AsyncOpenAI client with base_url: {config.client_config.api_base_url}"
    )

    # load environment
    vf_env = vf.load_environment(env_id=config.env_id, **config.env_args)

    # run evaluation
    results_path = get_eval_results_path(config)
    logger.info(f"Starting evaluation with model: {config.model}")
    logger.info(
        f"Configuration: num_examples={config.num_examples}, rollouts_per_example={config.rollouts_per_example}, max_concurrent={config.max_concurrent}"
    )
    start_time = time.time()
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
    )
    end_time = time.time()
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    if config.print_results:
        print_results(results)
    if config.save_results:
        save_rollout_results(results, config.save_to_hf_hub, config.hf_hub_dataset_name)
    return results


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
    results_dict = {
        "example_id": results["example_id"],
        "prompt": clean_prompts,
        "completion": clean_completions,
        "task": results["task"],
        "reward": results["reward"],
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
