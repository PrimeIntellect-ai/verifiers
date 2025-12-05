import importlib.util
import json
import time

import structlog
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import numpy as np
from datasets import Dataset, disable_progress_bar, enable_progress_bar
from datasets.utils import logging as ds_logging

import verifiers as vf
from verifiers.types import Endpoints, EvalConfig, GenerateMetadata, GenerateOutputs
from verifiers.utils.client_utils import setup_client
from verifiers.utils.logging_utils import print_prompt_completions_sample, log_context
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls
from verifiers.utils.path_utils import get_eval_results_path

logger = structlog.stdlib.get_logger(component=__name__)


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

    # Use rich console for structured output
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Config table
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="dim")
    config_table.add_column("Value", style="bold")
    config_table.add_row("Environment", results['metadata']['env_id'])
    config_table.add_row("Model", results['metadata']['model'])
    config_table.add_row("Provider", results['metadata']['base_url'])
    config_table.add_row("Examples", str(results['metadata']['num_examples']))
    config_table.add_row("Rollouts", str(results['metadata']['rollouts_per_example']))

    console.print()
    console.print("[bold]Evaluation Config[/bold]")
    console.print(config_table)
    console.print()

    # Sample output
    printable_prompts = [messages_to_printable(p) for p in results["prompt"]]
    printable_completions = [messages_to_printable(c) for c in results["completion"]]
    print_prompt_completions_sample(
        printable_prompts,
        printable_completions,
        results["reward"],
        step=0,
        num_samples=num_samples,
    )

    # Results table
    r = results["metadata"]["rollouts_per_example"]
    n = len(results["reward"]) // r

    results_table = Table(show_header=True, header_style="bold")
    results_table.add_column("Metric")
    results_table.add_column("Avg", justify="right")
    results_table.add_column("Std", justify="right")
    for i in range(r):
        results_table.add_column(f"R{i+1}", justify="right")

    # Reward row
    avg_reward = sum(results['reward']) / len(results['reward'])
    std_reward = float(np.std(results['reward']))
    reward_row = ["reward", f"{avg_reward:.3f}", f"{std_reward:.3f}"]
    for i in range(r):
        trials = [results["reward"][i + (j * r)] for j in range(n)]
        reward_row.append(f"{sum(trials)/len(trials):.3f}")
    results_table.add_row(*reward_row, style="bold cyan")

    # Metric rows
    for k in results["metrics"]:
        v = results["metrics"][k]
        avg_v = sum(v) / len(v)
        std_v = float(np.std(v))
        metric_row = [k, f"{avg_v:.3f}", f"{std_v:.3f}"]
        for i in range(r):
            trials = [v[i + (j * r)] for j in range(n)]
            metric_row.append(f"{sum(trials)/len(trials):.3f}")
        results_table.add_row(*metric_row)

    console.print()
    console.print("[bold]Results[/bold]")
    console.print(results_table)


def print_detailed_stats(results: GenerateOutputs):
    """Print detailed statistics table for evaluation results."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    states = results["state"]

    # Extract timing data (convert ms to seconds)
    total_times = [
        s["timing"]["total_ms"] / 1000
        for s in states
        if s.get("timing") and "total_ms" in s["timing"]
    ]
    gen_times = [
        s["timing"]["generation_ms"] / 1000
        for s in states
        if s.get("timing") and "generation_ms" in s["timing"]
    ]
    score_times = [
        s["timing"]["scoring_ms"] / 1000
        for s in states
        if s.get("timing") and "scoring_ms" in s["timing"]
    ]

    # Extract turn data from trajectory
    turns_used = [len(s.get("trajectory", [])) for s in states]

    # Count exceptions based on stop_condition
    n_exceptions = sum(
        1 for s in states
        if s.get("stop_condition") and "error" in s.get("stop_condition", "").lower()
    )

    detailed_table = Table(show_header=True, header_style="bold")
    detailed_table.add_column("Stat")
    detailed_table.add_column("Avg", justify="right")
    detailed_table.add_column("Min", justify="right")
    detailed_table.add_column("Max", justify="right")

    if total_times:
        detailed_table.add_row(
            "Total time (s)",
            f"{np.mean(total_times):.2f}",
            f"{min(total_times):.2f}",
            f"{max(total_times):.2f}",
        )
    if gen_times:
        detailed_table.add_row(
            "Generation time (s)",
            f"{np.mean(gen_times):.2f}",
            f"{min(gen_times):.2f}",
            f"{max(gen_times):.2f}",
        )
    if score_times:
        detailed_table.add_row(
            "Scoring time (s)",
            f"{np.mean(score_times):.2f}",
            f"{min(score_times):.2f}",
            f"{max(score_times):.2f}",
        )
    if turns_used:
        detailed_table.add_row(
            "Turns used",
            f"{np.mean(turns_used):.1f}",
            f"{min(turns_used)}",
            f"{max(turns_used)}",
        )

    detailed_table.add_row("Exceptions", str(n_exceptions), "-", "-")

    console.print()
    console.print("[bold]Detailed Stats[/bold]")
    console.print(detailed_table)


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
    with log_context(env_id=config.env_id, model=config.model):
        logger.info(
            "Starting evaluation",
            examples=config.num_examples,
            rollouts=config.rollouts_per_example,
            _print=True
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
        # Calculate scoring stats
        score_times = [
            s["timing"]["scoring_ms"] / 1000
            for s in results["state"]
            if s.get("timing") and "scoring_ms" in s["timing"]
        ]
        avg_score_time = round(sum(score_times) / len(score_times), 2) if score_times else 0
        logger.info(
            "Evaluation complete",
            avg_reward=round(results["metadata"]["avg_reward"], 3),
            avg_score_time_s=avg_score_time,
            _print=True
        )
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
