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

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]

import numpy as np

import verifiers as vf

if TYPE_CHECKING:
    pass
from verifiers.types import (
    Endpoints,
    EvalConfig,
    EvalRunConfig,
    GenerateOutputs,
    LogCallback,
    ProgressCallback,
    RolloutOutput,
    StartCallback,
)
from verifiers.utils.async_utils import EventLoopLagMonitor
from verifiers.utils.logging_utils import print_prompt_completions_sample, print_time
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
        "save_every",
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
        merged_eval_list.append(merged)

    return merged_eval_list


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

    tasks = set([o["task"] for o in results["outputs"]])
    if len(tasks) > 1:
        for task in tasks:
            task_results = get_task_outputs(results, task)
            print(f"\n--- {task} ---")
            print_rewards(task_results)
            print_info(task_results)
            print_timing(task_results)


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
        results_path = get_eval_results_path(config)
        logger.debug(f"Starting evaluation with model: {config.model}")
        logger.debug(
            f"Configuration: num_examples={config.num_examples}, rollouts_per_example={config.rollouts_per_example}, max_concurrent={config.max_concurrent}"
        )
        # disable tqdm when callbacks are provided (TUI handles progress display)
        use_tqdm = config.use_tqdm and on_progress is None
        effective_max_concurrent = config.max_concurrent
        if (
            not config.independent_scoring
            and config.max_concurrent > 0
            and config.rollouts_per_example > 1
        ):
            effective_max_concurrent = math.ceil(
                config.max_concurrent / config.rollouts_per_example
            )

        outputs = await vf_env.evaluate(
            client=config.client_config,
            model=config.model,
            sampling_args=config.sampling_args,
            num_examples=config.num_examples,
            rollouts_per_example=config.rollouts_per_example,
            max_concurrent=effective_max_concurrent,
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
        reward_accum = 0
        metrics_accum = defaultdict(float)
        error_accum = 0
        input_tokens_accum = 0.0
        output_tokens_accum = 0.0
        usage_count = 0
        usage_seen = False

        def on_start(total: int) -> None:
            # total is num_examples * rollouts_per_example
            # compute actual num_examples (resolves -1 to actual count)
            num_examples = total // env_config.rollouts_per_example
            display.update_env_state(env_idx, total=total, num_examples=num_examples)

        def on_progress(
            all_outputs: list[RolloutOutput], new_outputs: list[RolloutOutput]
        ) -> None:
            nonlocal error_accum, reward_accum, metrics_accum
            nonlocal input_tokens_accum, output_tokens_accum, usage_seen, usage_count

            # Progress is always rollout-based
            completed = len(all_outputs)

            for o in new_outputs:
                if o.get("error") is not None:
                    error_accum += 1
                reward = o.get("reward")
                if reward is not None:
                    reward_accum += reward
                output_metrics = o.get("metrics") or {}
                for name, value in output_metrics.items():
                    if value is not None:
                        metrics_accum[name] += value
                token_usage = o.get("token_usage")
                if isinstance(token_usage, dict):
                    usage_seen = True
                    usage_count += 1
                    input_tokens_accum += float(token_usage.get("input_tokens", 0.0))
                    output_tokens_accum += float(token_usage.get("output_tokens", 0.0))

            # Compute averages over completed rollouts
            reward = reward_accum / completed
            metrics = {name: metrics_accum[name] / completed for name in metrics_accum}
            error_rate = error_accum / completed
            usage = None
            if usage_seen and usage_count > 0:
                usage = {
                    "input_tokens": input_tokens_accum / usage_count,
                    "output_tokens": output_tokens_accum / usage_count,
                }

            display.update_env_state(
                env_idx,
                progress=completed,
                reward=reward,
                metrics=metrics,
                usage=usage,
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
