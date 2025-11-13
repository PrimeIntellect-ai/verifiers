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
from verifiers.types import (
    Endpoints,
    EvalConfig,
    GenerateMetadata,
    GenerateOutputs,
    State,
)
from verifiers.utils.client_utils import setup_client
from verifiers.utils.logging_utils import print_prompt_completions_sample
from verifiers.utils.message_utils import messages_to_printable
from verifiers.utils.path_utils import get_eval_results_path
from verifiers.utils.rollout_utils import serialize_rollout
from verifiers.utils.streaming_utils import StreamingHandler

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
    for i in range(r):
        # rounded to 3 decimal places
        trials = [round(results["reward"][(i * n) + j], 3) for j in range(n)]
        out = f"r{i + 1}: {trials}"
        print(out)
    for k in results["metrics"]:
        v = results["metrics"][k]
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
        for i in range(r):
            # rounded to 3 decimal places
            trials = [round(v[(i * n) + j], 3) for j in range(n)]
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

    # Setup streaming handler for incremental saving
    streaming_handler = None
    rollout_callback = None

    if config.save_results:
        total_rollouts = config.num_examples * config.rollouts_per_example
        results_jsonl_path = results_path / "results.jsonl"
        streaming_handler = StreamingHandler(
            results_path=results_jsonl_path,
            total_rollouts=total_rollouts,
            state_columns=config.state_columns,
        )

        async def rollout_callback(state: State) -> None:
            streaming_handler.log_rollout(state)
            await streaming_handler.write_rollout_jsonl(state)

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
        on_rollout_complete=rollout_callback,
    )
    end_time = time.time()
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

    if config.print_results:
        print_results(results)

    if config.save_results:
        metadata_dict = sanitize_metadata(results["metadata"])
        with open(results_path / "metadata.json", "w") as f:
            json.dump(metadata_dict, f)
        logger.info(f"Metadata saved to {results_path / 'metadata.json'}")

        if config.save_to_hf_hub:
            dataset = Dataset.from_json(str(results_path / "results.jsonl"))
            dataset_name = config.hf_hub_dataset_name or get_hf_hub_dataset_name(
                results
            )
            dataset.push_to_hub(dataset_name)
            logger.info(f"Dataset pushed to Hugging Face Hub: {dataset_name}")

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
    state_columns = results["metadata"]["state_columns"]

    serialized_rollouts = [
        serialize_rollout(state, state_columns=state_columns, include_timestamp=False)
        for state in results["state"]
    ]

    if not serialized_rollouts:
        return Dataset.from_dict({})

    all_keys = {key for rollout in serialized_rollouts for key in rollout.keys()}

    results_dict = {
        key: [rollout.get(key) for rollout in serialized_rollouts] for key in all_keys
    }

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
