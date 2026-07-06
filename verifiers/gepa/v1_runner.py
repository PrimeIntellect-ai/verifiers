import asyncio
import logging
from pathlib import Path

from gepa.api import optimize

from verifiers.gepa.adapter import make_reflection_lm
from verifiers.gepa.config import GEPAV1Config
from verifiers.gepa.display import GEPADisplay
from verifiers.gepa.gepa_utils import save_gepa_results
from verifiers.gepa.v1_adapter import (
    VerifiersV1GEPAAdapter,
    ensure_v1_gepa_supported,
    make_v1_gepa_dataset,
    shared_initial_prompt_v1,
)
from verifiers.types import ClientConfig
from verifiers.v1.clients import BaseClientConfig, resolve_client
from verifiers.v1.env import Environment

logger = logging.getLogger(__name__)


def run_gepa_v1_optimization(
    config: GEPAV1Config,
    model: str,
    reflection_model: str,
    client_config: BaseClientConfig,
    reflection_client_config: ClientConfig,
    run_dir: Path | None,
):
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    display = GEPADisplay(
        env_id=config.environment_label,
        model=model,
        reflection_model=reflection_model,
        max_metric_calls=config.gepa.max_calls,
        num_train=config.gepa.num_train,
        num_val=config.gepa.num_val,
        log_file=run_dir / "gepa.log" if run_dir else None,
        perfect_score=config.gepa.perfect_score,
        screen=config.tui,
    )

    with display:
        env = Environment(config)
        ensure_v1_gepa_supported(env)
        tasks = env.taskset.load_tasks()
        if not tasks:
            raise ValueError(f"No tasks available for v1 GEPA env {config.env_id!r}.")

        initial_prompt = shared_initial_prompt_v1(tasks)
        if not initial_prompt:
            logger.warning("No system prompt attached to v1 tasks.")
            logger.warning(
                "GEPA will add a system prompt to each task during candidate evaluation."
            )

        trainset = make_v1_gepa_dataset(tasks, config.gepa.num_train, config.gepa.seed)
        valset = make_v1_gepa_dataset(tasks, config.gepa.num_val, config.gepa.seed + 1)
        if not trainset:
            raise ValueError("No train examples available for v1 GEPA.")
        if not valset:
            raise ValueError("No validation examples available for v1 GEPA.")

        valset_example_ids = [
            item.get("example_id", i) for i, item in enumerate(valset)
        ]
        display.set_valset_info(len(valset), valset_example_ids)
        display.num_train = len(trainset)
        display.num_val = len(valset)

        client = resolve_client(client_config)
        runner = asyncio.Runner()
        serving = env.serving([row["task"] for row in [*trainset, *valset]])
        serving_entered = False
        try:
            runner.run(serving.__aenter__())
            serving_entered = True
            adapter = VerifiersV1GEPAAdapter(
                env=env,
                client=client,
                model=model,
                sampling=config.sampling,
                runner=runner,
                max_concurrent=config.gepa.max_concurrent,
                state_columns=config.gepa.state_columns,
                display=display,
            )
            reflection_lm = make_reflection_lm(
                client_config=reflection_client_config, model=reflection_model
            )
            optimize_kwargs: dict = {
                "seed_candidate": {"system_prompt": initial_prompt},
                "trainset": trainset,
                "valset": valset,
                "adapter": adapter,
                "reflection_lm": reflection_lm,
                "max_metric_calls": config.gepa.max_calls,
                "reflection_minibatch_size": config.gepa.minibatch_size,
                "run_dir": str(run_dir) if run_dir else None,
                "seed": config.gepa.seed,
                "display_progress_bar": False,
                "skip_perfect_score": config.gepa.perfect_score is not None,
                "logger": display,
            }
            if config.gepa.perfect_score is not None:
                optimize_kwargs["perfect_score"] = config.gepa.perfect_score
            result = optimize(**optimize_kwargs)
        finally:
            if serving_entered:
                runner.run(serving.__aexit__(None, None, None))
            runner.run(client.close())
            runner.close()

        save_path = None
        if run_dir and config.save_results:
            save_gepa_results(
                run_dir, result, config=_run_config(config, model, reflection_model)
            )
            save_path = str(run_dir)
            logger.debug(f"Results saved to {run_dir}")

        best_prompt = result.best_candidate.get("system_prompt", "")  # type: ignore[unresolved-attribute]
        display.set_result(best_prompt=best_prompt, save_path=save_path)

    return result


def _run_config(config: GEPAV1Config, model: str, reflection_model: str) -> dict:
    return {
        "env_id": config.environment_label,
        "v1": True,
        "env_config": config.model_dump(mode="json"),
        "model": model,
        "reflection_model": reflection_model,
        "num_train": config.gepa.num_train,
        "num_val": config.gepa.num_val,
        "max_metric_calls": config.gepa.max_calls,
        "minibatch_size": config.gepa.minibatch_size,
        "perfect_score": config.gepa.perfect_score,
        "state_columns": config.gepa.state_columns,
        "seed": config.gepa.seed,
    }
