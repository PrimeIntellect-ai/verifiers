"""Legacy (v0) eval: run a classic `load_environment(...)` env through the v0 evaluator.

There is no separate v0 parse: every invocation flows through the single v1 `EvalConfig`
parse, and a taskset id that resolves to a classic v0 env auto-converts to the legacy `id`
at validation time (`EnvConfig._resolve_plugins` via `is_legacy_env`). This module is the
one v1 -> v0 mapping: the typed `EvalConfig` becomes a v0 `EvalConfig` and runs through
`run_evaluations`, writing the v0 artifact shape (`metadata.json` + v0 `results.jsonl`).
"""

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

from verifiers import setup_logging
from verifiers.types import (
    ClientConfig as V0ClientConfig,
    EvalConfig as V0EvalConfig,
    EvalRunConfig,
)
from verifiers.utils.eval_utils import get_log_level, run_evaluations
from verifiers.utils.import_utils import load_toml
from verifiers.v1.configs.eval import EvalConfig


def run_legacy_eval(config: EvalConfig) -> None:
    """Run the parsed v1 config through the v0 evaluator."""
    setup_logging(get_log_level(config.verbose))
    asyncio.run(run_evaluations(EvalRunConfig(evals=[v0_eval_config(config)])))


def v0_eval_config(config: EvalConfig) -> V0EvalConfig:
    """Map the typed v1 `EvalConfig` onto the v0 one — shape changes only.

    Counts the CLI/config didn't set fall back to the env's own `[tool.verifiers.eval]`
    defaults (its pyproject), then to the v1 semantics (all tasks, 1 rollout)."""
    assert config.id is not None
    defaults = _env_eval_defaults(config.id)
    num_tasks = (
        config.num_tasks
        if "num_tasks" in config.model_fields_set
        else defaults.get("num_examples", config.num_tasks)
    )
    num_rollouts = (
        config.num_rollouts
        if "num_rollouts" in config.model_fields_set
        else defaults.get("rollouts_per_example", config.num_rollouts)
    )
    extra_env_kwargs = dict(config.extra_env_kwargs)
    if config.timeout.rollout is not None:
        extra_env_kwargs["timeout_seconds"] = config.timeout.rollout
    return V0EvalConfig(
        env_id=config.id,
        env_args=dict(config.args),
        env_dir_path="./environments",
        extra_env_kwargs=extra_env_kwargs,
        model=config.model,
        client_config=V0ClientConfig(
            # the train client has no v0 dialect field; fall back to chat completions
            client_type=getattr(
                config.client, "v0_client_type", "openai_chat_completions"
            ),
            api_key_var=config.client.api_key_var,
            api_base_url=config.client.base_url,
            extra_headers=dict(config.client.headers),
            extra_headers_from_state={"X-Session-ID": "trajectory_id"},
        ),
        sampling_args=config.sampling.model_dump(exclude_none=True),
        num_examples=-1 if num_tasks is None else num_tasks,  # v0: -1 = all
        rollouts_per_example=num_rollouts,
        shuffle=config.shuffle,
        shuffle_seed=0 if config.shuffle else None,  # v0 shuffling needs a seed
        max_concurrent=-1 if config.max_concurrent is None else config.max_concurrent,
        max_retries=config.retries.rollout.max_retries,
        verbose=config.verbose,
        disable_tui=not config.rich,
        output_dir=str(config.output_dir) if config.output_dir is not None else None,
        state_columns=[],
        save_results=True,
    )


def _env_eval_defaults(env_id: str) -> dict[str, Any]:
    """The env's eval counts from its installed pyproject (`[tool.verifiers.eval]`)."""
    module_name = env_id.replace("-", "_").split("/")[-1]
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {}
    if spec.submodule_search_locations:
        base_dir = Path(next(iter(spec.submodule_search_locations)))
    elif spec.origin:
        base_dir = Path(spec.origin).parent
    else:
        return {}
    pyproject = base_dir / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    with pyproject.open("rb") as handle:
        raw = load_toml(handle)
    eval_config = raw.get("tool", {}).get("verifiers", {}).get("eval", {})
    return {
        key: eval_config[key]
        for key in ("num_examples", "rollouts_per_example")
        if key in eval_config
    }
