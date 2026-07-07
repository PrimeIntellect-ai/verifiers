"""The legacy (v0) eval's live display: the classic vf-eval TUI, fed by the bridge.

`run_legacy_eval_with_display` wraps `run_legacy_eval` in the old `EvalDisplay`
(`verifiers/utils/eval_display.py`) — one env panel with live progress, running
reward/metrics/timing averages, and tailing of the run's `eval.log`. The bridge stays
display-free; it reports through its `on_start`/`on_output` callbacks."""

import asyncio
from pathlib import Path

from verifiers.v1.cli.output import output_path
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.legacy import run_legacy_eval
from verifiers.v1.trace import Trace

# The timing spans the classic display averages (mirrors the old on_display_progress).
_TIMING_KEYS = ("setup", "generation", "scoring", "overhead", "total", "model", "env")


def _display_config(config: EvalConfig):
    """The v0 config shape the classic display renders (title, settings, client target)."""
    from verifiers.types import EvalConfig as V0EvalConfig

    return V0EvalConfig(
        env_id=config.id or "",
        env_args=config.args,
        env_dir_path="",
        model=config.model,
        client_config={
            "api_base_url": config.client.base_url,
            "api_key_var": config.client.api_key_var,
        },
        sampling_args=config.sampling.model_dump(exclude_none=True),
        num_examples=config.num_tasks or 0,
        rollouts_per_example=config.num_rollouts,
        max_concurrent=config.max_concurrent or 0,
        extra_env_kwargs=config.extra_env_kwargs,
        verbose=config.verbose,
    )


async def run_legacy_eval_with_display(
    config: EvalConfig, log_file: str
) -> list[Trace]:
    """Run the bridge under the classic display; returns the traces like `run_legacy_eval`."""
    from verifiers.utils.eval_display import EvalDisplay

    display = EvalDisplay([_display_config(config)])
    display.add_log_file_for_env(0, Path(log_file))

    # Running aggregates over raw v0 outputs — the same numbers the classic evaluator's
    # progress callbacks carried (avg reward / metrics / timing, error rate).
    count = 0
    reward_sum = 0.0
    error_count = 0
    metric_sums: dict[str, float] = {}
    timing_sums: dict[str, float] = {}
    timing_counts: dict[str, int] = {}

    def on_start(num_tasks: int, num_rollouts: int) -> None:
        display.update_env_state(
            0, status="running", num_examples=num_tasks, total=num_rollouts
        )

    def on_output(out: dict) -> None:
        nonlocal count, reward_sum, error_count
        count += 1
        reward = out.get("reward")
        if isinstance(reward, (int, float)):
            reward_sum += float(reward)
        if out.get("error") is not None:
            error_count += 1
        for key, value in (out.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        timing = out.get("timing") or {}
        for key in _TIMING_KEYS:
            value = timing.get(key)
            if isinstance(value, dict):
                value = value.get("duration", 0.0)
            if isinstance(value, (int, float)):
                timing_sums[key] = timing_sums.get(key, 0.0) + float(value)
                timing_counts[key] = timing_counts.get(key, 0) + 1
        display.update_env_state(
            0,
            progress=count,
            reward=reward_sum / count,
            metrics={k: v / count for k, v in metric_sums.items()},
            error_rate=error_count / count,
            avg_timing={k: timing_sums[k] / timing_counts[k] for k in timing_sums},
        )

    async def tick() -> None:  # keep the elapsed-time counters moving between outputs
        while True:
            await asyncio.sleep(1)
            display.refresh()

    async with display:
        ticker = asyncio.create_task(tick())
        try:
            traces = await run_legacy_eval(
                config, on_start=on_start, on_output=on_output
            )
            display.update_env_state(
                0, status="completed", save_path=output_path(config)
            )
        except Exception as exc:
            display.update_env_state(0, status="failed", error=str(exc))
            raise
        finally:
            ticker.cancel()
    display.print_final_summary()
    return traces
