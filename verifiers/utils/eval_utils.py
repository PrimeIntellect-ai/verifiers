from __future__ import annotations

import asyncio
import importlib.util
import logging
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

from datasets import disable_progress_bar, enable_progress_bar
from datasets.utils import logging as ds_logging

import numpy as np

import verifiers as vf
from verifiers.utils.import_utils import load_toml

if TYPE_CHECKING:
    pass
from verifiers.types import (
    Endpoint,
    Endpoints,
    EvalConfig,
    EvalRunConfig,
    GenerateMetadata,
    GenerateOutputs,
    LogCallback,
    ProgressCallback,
    RolloutInput,
    RolloutOutput,
    StartCallback,
)
from verifiers.utils.async_utils import EventLoopLagMonitor
from verifiers.utils.logging_utils import print_prompt_completions_sample, print_time
from verifiers.utils.path_utils import get_eval_results_path

logger = logging.getLogger(__name__)


def _coerce_endpoint(raw_endpoint: object, source: str) -> Endpoint:
    if not isinstance(raw_endpoint, dict):
        raise ValueError(f"Endpoint entry must be a table/dict in {source}")

    raw_endpoint_dict = cast(dict[str, object], raw_endpoint)
    model = raw_endpoint_dict.get("model")
    url = raw_endpoint_dict.get("url")
    key = raw_endpoint_dict.get("key")

    missing = [
        field
        for field, value in (("model", model), ("url", url), ("key", key))
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required field(s) {missing} in {source}")

    if (
        not isinstance(model, str)
        or not isinstance(url, str)
        or not isinstance(key, str)
    ):
        raise ValueError(
            f"Fields 'model', 'url', and 'key' must all be strings in {source}"
        )

    return Endpoint(model=model, url=url, key=key)


def _normalize_python_endpoints(raw_endpoints: object, source: Path) -> Endpoints:
    if not isinstance(raw_endpoints, dict):
        raise ValueError(f"ENDPOINTS must be a dict in {source}")

    raw_endpoints_dict = cast(dict[str, object], raw_endpoints)
    normalized: Endpoints = {}
    for endpoint_id, raw_endpoint_group in raw_endpoints_dict.items():
        if not isinstance(endpoint_id, str):
            raise ValueError(f"Endpoint ids must be strings in {source}")

        if isinstance(raw_endpoint_group, list):
            if not raw_endpoint_group:
                raise ValueError(
                    f"Endpoint '{endpoint_id}' has an empty endpoint list in {source}"
                )
            normalized[endpoint_id] = [
                _coerce_endpoint(
                    raw_endpoint,
                    source=f"{source} (ENDPOINTS['{endpoint_id}'])",
                )
                for raw_endpoint in raw_endpoint_group
            ]
        else:
            normalized[endpoint_id] = [
                _coerce_endpoint(
                    raw_endpoint_group,
                    source=f"{source} (ENDPOINTS['{endpoint_id}'])",
                )
            ]

    return normalized


def _normalize_toml_endpoints(raw_toml: object, source: Path) -> Endpoints:
    if not isinstance(raw_toml, dict):
        raise ValueError(f"Expected top-level TOML table in {source}")

    raw_toml_dict = cast(dict[str, object], raw_toml)
    raw_endpoint_entries = raw_toml_dict.get("endpoint", [])
    if not isinstance(raw_endpoint_entries, list):
        raise ValueError(
            f"Expected [[endpoint]] array-of-tables in {source}, got {type(raw_endpoint_entries)}"
        )

    normalized: Endpoints = {}
    for idx, raw_entry in enumerate(raw_endpoint_entries):
        entry_source = f"{source} ([[endpoint]] index {idx})"
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"Each [[endpoint]] entry must be a table in {entry_source}"
            )

        raw_entry_dict = cast(dict[str, object], raw_entry)
        endpoint_id = raw_entry_dict.get("endpoint_id")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            raise ValueError(
                f"Each [[endpoint]] entry must include non-empty string 'endpoint_id' in {entry_source}"
            )

        url = raw_entry_dict.get("url")
        api_base_url = raw_entry_dict.get("api_base_url")
        if url is not None and api_base_url is not None and url != api_base_url:
            raise ValueError(
                f"Conflicting values for 'url' and 'api_base_url' in {entry_source}"
            )

        key = raw_entry_dict.get("key")
        api_key_var = raw_entry_dict.get("api_key_var")
        if key is not None and api_key_var is not None and key != api_key_var:
            raise ValueError(
                f"Conflicting values for 'key' and 'api_key_var' in {entry_source}"
            )

        endpoint_payload = {
            k: v for k, v in raw_entry_dict.items() if k != "endpoint_id"
        }
        endpoint_payload["url"] = url if url is not None else api_base_url
        endpoint_payload["key"] = key if key is not None else api_key_var
        endpoint = _coerce_endpoint(
            endpoint_payload,
            source=f"{entry_source} (endpoint_id={endpoint_id!r})",
        )
        normalized.setdefault(endpoint_id, []).append(endpoint)

    return normalized


def resolve_endpoints_file(endpoints_path: str) -> Path | None:
    endpoints_path_obj = Path(endpoints_path)
    if endpoints_path_obj.is_dir():
        toml_file = endpoints_path_obj / "endpoints.toml"
        python_file = endpoints_path_obj / "endpoints.py"
        if toml_file.exists():
            return toml_file
        if python_file.exists():
            return python_file
        return None
    return endpoints_path_obj


def load_endpoints(endpoints_path: str):
    try:
        endpoints_file = resolve_endpoints_file(endpoints_path)
        if endpoints_file is None:
            raise ImportError(
                f"Neither endpoints.py nor endpoints.toml found at {endpoints_path}"
            )

        if endpoints_file.exists():
            logger.debug(f"Loading endpoint registry from {endpoints_file}")
            if endpoints_file.suffix == ".py":
                spec = importlib.util.spec_from_file_location(
                    "endpoints", endpoints_file
                )
                assert spec and spec.loader
                endpoints_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(endpoints_module)
                # check that module exposes ENDPOINTS
                if not hasattr(endpoints_module, "ENDPOINTS"):
                    raise AttributeError(
                        f"Module '{endpoints_file}' does not have a 'ENDPOINTS' attribute"
                    )
                endpoints = _normalize_python_endpoints(
                    cast(object, endpoints_module.ENDPOINTS),
                    source=endpoints_file,
                )
            elif endpoints_file.suffix == ".toml":
                with open(endpoints_file, "rb") as f:
                    raw_toml = load_toml(f)
                endpoints = _normalize_toml_endpoints(raw_toml, source=endpoints_file)
            else:
                raise ImportError(
                    f"Unsupported endpoints file extension '{endpoints_file.suffix}' at {endpoints_file}"
                )
            num_endpoint_variants = sum(len(group) for group in endpoints.values())
            logger.debug(
                "Successfully loaded %d endpoint ids (%d endpoint variant(s)) from registry",
                len(endpoints),
                num_endpoint_variants,
            )
        else:
            raise ImportError(f"Endpoint registry file not found at {endpoints_file}")
    except (ImportError, AttributeError, ValueError) as e:
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
        raw_config = load_toml(f)

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

    # valid fields mirror cli args, not evalconfig
    # TODO: properly tie EvalConfig to CLI
    valid_fields = {
        # environment
        "env_id",
        "env_args",
        "env_dir_path",
        "endpoints_path",
        "extra_env_kwargs",
        # model/client
        "endpoint_id",
        "model",
        "api_key_var",
        "api_base_url",
        "header",
        # sampling
        "sampling_args",
        "max_tokens",
        "temperature",
        # evaluation
        "num_examples",
        "rollouts_per_example",
        "max_concurrent",
        "independent_scoring",
        "max_retries",
        # logging
        "verbose",
        # saving
        "state_columns",
        "save_results",
        "resume",
        "resume_path",
        "save_to_hf_hub",
        "hf_hub_dataset_name",
    }

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
        # Resolve endpoints_path relative to the config file location.
        endpoints_path = merged.get("endpoints_path")
        if isinstance(endpoints_path, str):
            endpoints_path_obj = Path(endpoints_path)
            if not endpoints_path_obj.is_absolute():
                merged["endpoints_path"] = str(
                    (path.parent / endpoints_path_obj).resolve()
                )
        merged_eval_list.append(merged)

    return merged_eval_list


def filter_inputs(
    inputs: list[RolloutInput], outputs: list[RolloutOutput], rollouts_per_example: int
) -> list[RolloutInput]:
    """Filter inputs based on the number of rollouts per example."""
    inputs_by_example_id, outputs_by_example_id = defaultdict(list), defaultdict(list)
    for input in inputs:
        inputs_by_example_id[input["example_id"]].append(input)
    for output in outputs:
        outputs_by_example_id[output["example_id"]].append(output)

    filtered_inputs: list[RolloutInput] = []
    for example_id in inputs_by_example_id.keys():
        example_inputs = inputs_by_example_id[example_id]
        example_outputs = outputs_by_example_id[example_id]
        rollouts_left = rollouts_per_example - len(example_outputs)
        if rollouts_left > 0:
            filtered_inputs.extend(example_inputs[:rollouts_left])

    return filtered_inputs


def to_col_order(list_of_dicts: list[Mapping[str, float]]) -> dict[str, list[float]]:
    """Convert a list of mappings to a dictionary of lists, ordered by the keys of the first mapping."""
    if not list_of_dicts:
        return {}
    return {k: [m[k] for m in list_of_dicts] for k in list_of_dicts[0].keys()}


def get_task_outputs(results: GenerateOutputs, task: str) -> GenerateOutputs:
    """Get only the rollouts for a given task."""
    outputs = [o for o in results["outputs"] if o["task"] == task]
    return GenerateOutputs(
        outputs=outputs,
        metadata=results["metadata"],  # duplicate metadata
    )


def print_rewards(results: GenerateOutputs):
    rewards = [o["reward"] for o in results["outputs"]]
    print("Rewards:")
    print(
        f"reward: avg - {sum(rewards) / len(rewards):.3f}, std - {np.std(rewards):.3f}"
    )
    r = results["metadata"]["rollouts_per_example"]
    n = len(rewards) // r
    # results are sorted by example_id, so rollout i is at indices [i, i+r, i+2r, ...]
    for i in range(r):
        trials = [round(rewards[i + (j * r)], 3) for j in range(n)]
        out = f"r{i + 1}: {trials}"
        print(out)

    metrics = [o["metrics"] for o in results["outputs"]]
    metrics_col = to_col_order(metrics)
    for k in metrics_col.keys():
        v = metrics_col[k]
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
        for i in range(r):
            trials = [round(v[i + (j * r)], 3) for j in range(n)]
            out = f"r{i + 1}: {trials}"
            print(out)


def print_info(results: GenerateOutputs):
    is_truncated = [o["is_truncated"] for o in results["outputs"]]
    print("Info:")
    print(
        f"is_truncated: avg - {np.mean(is_truncated):.3f}, std - {np.std(is_truncated):.3f}"
    )
    stop_conditions = [o["stop_condition"] for o in results["outputs"]]
    counter = Counter(stop_conditions)
    print(
        f"stop_conditions: {', '.join([f'{k}: {v / counter.total():.3f}' for k, v in counter.items()])}"
    )
    errors = [o.get("error") for o in results["outputs"]]
    has_errors = [e is not None for e in errors]
    if any(has_errors):
        print(
            f"errors: avg - {np.mean(has_errors):.3f}, std - {np.std(has_errors):.3f}"
        )
        error_chains = [e["error_chain_str"] for e in errors if e is not None]
        # Errors are serialized as strings, count unique error types
        counter = Counter(error_chains)
        for error_str, count in counter.items():
            print(f" - {error_str}: {count / counter.total():.3f}")


def print_timing(results: GenerateOutputs):
    print("Timing:")
    timing = [o["timing"] for o in results["outputs"]]
    timing_col = to_col_order(timing)
    generation_ms_arr = np.array(timing_col["generation_ms"])
    scoring_ms_arr = np.array(timing_col["scoring_ms"])
    total_ms_arr = np.array(timing_col["total_ms"])
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


def print_usage(results: GenerateOutputs):
    usage_count = 0
    input_tokens_total = 0.0
    output_tokens_total = 0.0
    for output in results["outputs"]:
        token_usage = output.get("token_usage")
        if not isinstance(token_usage, Mapping):
            continue
        usage_count += 1
        input_tokens_total += float(token_usage.get("input_tokens", 0.0))
        output_tokens_total += float(token_usage.get("output_tokens", 0.0))

    usage = None
    if usage_count > 0:
        usage = {
            "input_tokens": input_tokens_total / usage_count,
            "output_tokens": output_tokens_total / usage_count,
        }
    elif results["metadata"].get("usage") is not None:
        usage = results["metadata"]["usage"]

    if usage is None:
        return

    print("Usage:")
    print(f"input_tokens (avg): {usage['input_tokens']:.3f}")
    print(f"output_tokens (avg): {usage['output_tokens']:.3f}")


def print_results(results: GenerateOutputs, num_samples: int = 1):
    assert results["metadata"] is not None
    print("--- Evaluation ---")
    print(f"Environment: {results['metadata']['env_id']}")
    print(f"Model: {results['metadata']['model']}")
    print(f"Provider: {results['metadata']['base_url']}")
    print(f"Examples: {results['metadata']['num_examples']}")
    print(f"Rollouts per example: {results['metadata']['rollouts_per_example']}")
    print("--- Example ---")

    # prompt/completion are already in printable format from state_to_output
    printable_prompts = [o["prompt"] if o["prompt"] else [] for o in results["outputs"]]
    printable_completions = [
        o["completion"] if o["completion"] else [] for o in results["outputs"]
    ]
    rewards = [o["reward"] for o in results["outputs"]]
    errors = [o.get("error") for o in results["outputs"]]
    print_prompt_completions_sample(
        printable_prompts,
        printable_completions,
        errors,
        rewards,
        step=0,
        num_samples=num_samples,
    )
    print("--- All ---")
    print_rewards(results)
    print_info(results)
    print_timing(results)
    print_usage(results)

    tasks = set([o["task"] for o in results["outputs"]])
    if len(tasks) > 1:
        for task in tasks:
            task_results = get_task_outputs(results, task)
            print(f"\n--- {task} ---")
            print_rewards(task_results)
            print_info(task_results)
            print_timing(task_results)
            print_usage(task_results)


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


async def run_evaluation(
    config: EvalConfig,
    on_start: StartCallback | None = None,
    on_progress: ProgressCallback | None = None,
    on_log: LogCallback | None = None,
) -> GenerateOutputs:
    # load environment
    vf_env = vf.load_environment(env_id=config.env_id, **config.env_args)

    # set extra environment kwargs
    if config.extra_env_kwargs:
        logger.info(f"Setting extra environment kwargs: {config.extra_env_kwargs}")
        vf_env.set_kwargs(**config.extra_env_kwargs)

    # start env server as sidecar process
    try:
        await vf_env.start_server(extra_env_kwargs=config.extra_env_kwargs)

        # run evaluation
        results_path = config.resume_path or get_eval_results_path(config)
        logger.debug(f"Starting evaluation with model: {config.model}")
        logger.debug(
            f"Configuration: num_examples={config.num_examples}, rollouts_per_example={config.rollouts_per_example}, max_concurrent={config.max_concurrent}"
        )
        effective_group_max_concurrent = config.max_concurrent
        if (
            not config.independent_scoring
            and config.max_concurrent > 0
            and config.rollouts_per_example > 1
        ):
            # Grouped scoring applies the semaphore at group level. Convert
            # rollout-level concurrency to group-level slots.
            effective_group_max_concurrent = math.ceil(
                config.max_concurrent / config.rollouts_per_example
            )
            if config.num_examples > 0:
                effective_group_max_concurrent = min(
                    effective_group_max_concurrent, config.num_examples
                )

        outputs = await vf_env.evaluate(
            client=config.client_config,
            model=config.model,
            sampling_args=config.sampling_args,
            num_examples=config.num_examples,
            rollouts_per_example=config.rollouts_per_example,
            max_concurrent=effective_group_max_concurrent,
            results_path=results_path,
            state_columns=config.state_columns,
            save_results=config.save_results,
            push_to_hf_hub=config.save_to_hf_hub,
            hf_hub_dataset_name=config.hf_hub_dataset_name,
            independent_scoring=config.independent_scoring,
            max_retries=config.max_retries,
            on_start=on_start,
            on_progress=on_progress,
            on_log=on_log,
        )
    finally:
        await vf_env.stop_server()

    return outputs


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

        def on_start(raw_inputs: list[RolloutInput], filtered_inputs) -> None:
            total = len(raw_inputs)
            if (
                isinstance(filtered_inputs, list)
                and filtered_inputs
                and isinstance(filtered_inputs[0], list)
            ):
                remaining = sum(len(g) for g in filtered_inputs)
            else:
                remaining = len(filtered_inputs) if filtered_inputs else 0
            resumed = total - remaining
            num_examples = total // env_config.rollouts_per_example
            display.update_env_state(
                env_idx, total=total, num_examples=num_examples, progress=resumed
            )

        def on_progress(
            all_outputs: list[RolloutOutput],
            new_outputs: list[RolloutOutput],
            metadata: GenerateMetadata,
        ) -> None:
            display.update_env_state(
                env_idx,
                progress=len(all_outputs),
                reward=metadata.get("avg_reward"),
                metrics=metadata.get("avg_metrics"),
                error_rate=metadata.get("avg_error"),
                usage=metadata.get("usage"),
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

    async def refresh_loop() -> None:
        while not display.state.all_completed:
            display.refresh()
            await asyncio.sleep(1)

    try:
        async with display:
            refresh_task = asyncio.create_task(refresh_loop())
            try:
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
            finally:
                refresh_task.cancel()
                with suppress(asyncio.CancelledError):
                    await refresh_task

    except KeyboardInterrupt:
        pass  # exit on interrupt

    # print final summary after exit
    display.print_final_summary()
