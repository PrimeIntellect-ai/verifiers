"""Typed CLI for V0 GEPA prompt optimization."""

from collections.abc import Mapping
import json
import logging
import os
from pathlib import Path
import sys
import tomllib
from typing import Any, cast

from gepa.api import optimize
from pydantic_config import cli

import verifiers as vf
from verifiers import setup_logging
from verifiers.clients import resolve_client
from verifiers.envs.env_group import ENV_GROUP_INFO_KEY
from verifiers.gepa.adapter import VerifiersGEPAAdapter, make_reflection_lm
from verifiers.gepa.config import GEPAConfig, GEPAEnvConfig, GEPAOptimizationConfig
from verifiers.gepa.display import GEPADisplay
from verifiers.gepa.gepa_utils import save_gepa_results
from verifiers.types import ClientConfig, EndpointConfig
from verifiers.utils.eval_utils import load_endpoints
from verifiers.utils.path_utils import get_gepa_results_path

logger = logging.getLogger(__name__)


def _gepa_extra_headers_from_group(
    endpoint_group: list[EndpointConfig], alias: str
) -> dict[str, str]:
    maps = [dict(entry.extra_headers) for entry in endpoint_group]
    unique = {tuple(sorted(m.items())) for m in maps}
    if len(unique) > 1:
        raise ValueError(
            f"Endpoint alias {alias!r} has different headers across endpoint variants; "
            "GEPA requires a single consistent header set."
        )
    return maps[0] if maps else {}


DEFAULT_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"
USAGE = "usage: vf-gepa <env-id> [options] | vf-gepa @ config.toml"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    target_idx = None
    bool_flags = {
        "--verbose",
        "--tui",
        "--no-tui",
        "--save-results",
        "--no-save-results",
    }
    for idx, arg in enumerate(argv):
        if arg.startswith(("-", "@")):
            continue
        previous = argv[idx - 1] if idx else ""
        next_arg = argv[idx + 1] if idx + 1 < len(argv) else ""
        follows_value = bool(previous and not previous.startswith(("-", "@")))
        follows_bool = previous in bool_flags
        followed_by_flag = not next_arg or next_arg.startswith("-")
        if idx == 0 or (followed_by_flag and (follows_value or follows_bool)):
            target_idx = idx
            break
    if target_idx is not None:
        target = argv.pop(target_idx)
        argv = (
            ["@", target] if Path(target).suffix == ".toml" else ["--id", target]
        ) + argv
    if not argv or any(arg in ("-h", "--help") for arg in argv):
        print(USAGE)
        cli(GEPAConfig, args=argv or ["--help"])
        return

    config_path = None
    for idx, arg in enumerate(argv):
        if arg == "@" and idx + 1 < len(argv):
            config_path = Path(argv[idx + 1])
            break
        if arg.startswith("@") and len(arg) > 1:
            config_path = Path(arg[1:])
            break
    endpoints_overridden = any(
        arg in ("--endpoints-path", "--endpoints_path")
        or arg.startswith(("--endpoints-path=", "--endpoints_path="))
        for arg in argv
    )

    # Rewrite bare top-level GEPA budget flags (e.g. `--max-calls`, `--num-train`)
    # to their nested `--gepa.*` form so documented commands keep working without
    # spelling the nested path.
    gepa_fields = {
        name.replace("_", "-") for name in GEPAOptimizationConfig.model_fields
    }
    argv = [
        f"--gepa.{arg[2:]}"
        if arg.startswith("--")
        and arg[2:].split("=", 1)[0].replace("_", "-") in gepa_fields
        else arg
        for arg in argv
    ]

    config = cli(GEPAConfig, args=argv)
    if (
        config_path is not None
        and config_path.is_file()
        and not endpoints_overridden
        and not config.endpoints_path.is_absolute()
    ):
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        if "endpoints_path" in raw:
            config.endpoints_path = (
                config_path.parent / config.endpoints_path
            ).resolve()
    setup_logging("DEBUG" if config.verbose else os.getenv("VF_LOG_LEVEL", "INFO"))
    env_configs = config.environments
    env_id = config.environment_label

    endpoints = load_endpoints(str(config.endpoints_path))
    main_extra_headers: dict[str, str] = {}
    if config.model in endpoints:
        endpoint_group = endpoints[config.model]
        endpoint = endpoint_group[0]
        main_extra_headers = _gepa_extra_headers_from_group(
            endpoint_group, config.model
        )

        endpoint_keys = {entry.api_key_var for entry in endpoint_group}
        if config.api_key_var is None and len(endpoint_keys) > 1:
            raise ValueError(
                f"Endpoint alias {config.model!r} maps to multiple API key variables; "
                "set --api-key-var."
            )
        api_key_var = config.api_key_var or endpoint.api_key_var

        endpoint_urls = {entry.base_url for entry in endpoint_group}
        if config.api_base_url is None and len(endpoint_urls) > 1:
            raise ValueError(
                f"Endpoint alias {config.model!r} maps to multiple URLs; set --api-base-url."
            )
        api_base_url = config.api_base_url or endpoint.base_url

        endpoint_models = {entry.model for entry in endpoint_group}
        if len(endpoint_models) > 1:
            raise ValueError(
                f"Endpoint alias {config.model!r} maps to multiple model ids: "
                f"{sorted(endpoint_models)}"
            )
        model = endpoint.model
    else:
        api_key_var = config.api_key_var or DEFAULT_API_KEY_VAR
        api_base_url = config.api_base_url or DEFAULT_API_BASE_URL
        model = config.model

    reflection_extra_headers = main_extra_headers
    if config.reflection_model and config.reflection_model in endpoints:
        endpoint_group = endpoints[config.reflection_model]
        endpoint = endpoint_group[0]
        reflection_extra_headers = _gepa_extra_headers_from_group(
            endpoint_group, config.reflection_model
        )
        endpoint_models = {entry.model for entry in endpoint_group}
        endpoint_keys = {entry.api_key_var for entry in endpoint_group}
        endpoint_urls = {entry.base_url for entry in endpoint_group}
        if len(endpoint_models) > 1 or len(endpoint_keys) > 1 or len(endpoint_urls) > 1:
            raise ValueError(
                f"Reflection endpoint alias {config.reflection_model!r} must resolve to "
                "one model, API key variable, and URL."
            )
        reflection_model = endpoint.model
        reflection_api_key_var = endpoint.api_key_var
        reflection_api_base_url = endpoint.base_url
    else:
        reflection_model = config.reflection_model or model
        reflection_api_key_var = api_key_var
        reflection_api_base_url = api_base_url

    run_dir = config.run_dir
    if (
        run_dir is not None
        and config_path is not None
        and config_path.is_file()
        and not run_dir.is_absolute()
    ):
        run_dir = (config_path.parent / run_dir).resolve()
    if run_dir is None and config.save_results:
        run_dir = get_gepa_results_path(env_id, model, str(config.env_dir_path))

    run_gepa_optimization(
        env_id=env_id,
        env_configs=env_configs,
        model=model,
        reflection_model=reflection_model,
        client_config=ClientConfig(
            api_key_var=api_key_var,
            api_base_url=api_base_url,
            extra_headers=main_extra_headers,
        ),
        reflection_client_config=ClientConfig(
            api_key_var=reflection_api_key_var,
            api_base_url=reflection_api_base_url,
            extra_headers=reflection_extra_headers,
        ),
        max_metric_calls=config.gepa.max_calls,
        minibatch_size=config.gepa.minibatch_size,
        perfect_score=config.gepa.perfect_score,
        state_columns=config.gepa.state_columns,
        num_train=config.gepa.num_train,
        num_val=config.gepa.num_val,
        max_concurrent=config.gepa.max_concurrent,
        sampling_args=config.sampling,
        seed=config.gepa.seed,
        run_dir=run_dir,
        save_results=config.save_results,
        tui_mode=config.tui,
    )


def _unique_env_names(env_configs: list[GEPAEnvConfig]) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for config in env_configs:
        base = config.id
        count = seen.get(base, 0) + 1
        seen[base] = count
        names.append(base if count == 1 else f"{base}:{count}")
    return names


def _load_gepa_environment(
    env_configs: list[GEPAEnvConfig],
) -> tuple[vf.Environment, list[vf.Environment], list[str]]:
    envs = []
    for config in env_configs:
        env_id = config.id
        env_args = config.args
        logger.debug(f"Loading environment: {env_id}")
        env = vf.load_environment(env_id=env_id, **env_args)

        extra_env_kwargs = config.extra_env_kwargs
        if extra_env_kwargs:
            logger.info(
                f"Setting extra environment kwargs for {env_id}: {extra_env_kwargs}"
            )
            env.set_kwargs(**extra_env_kwargs)
        envs.append(env)

    env_names = _unique_env_names(env_configs)
    if len(envs) == 1:
        return envs[0], envs, env_names
    return vf.EnvGroup(envs=envs, env_names=env_names), envs, env_names


def _shared_initial_prompt(envs: list[vf.Environment]) -> str:
    prompts = [env.system_prompt or "" for env in envs]
    initial_prompt = prompts[0] if prompts else ""
    if len(set(prompts)) > 1:
        logger.warning(
            "Multiple environment system prompts detected; GEPA will optimize one "
            "shared prompt initialized from the first environment."
        )
    return initial_prompt


def _balanced_counts(n: int, num_envs: int) -> list[int]:
    if n < 0:
        return [-1] * num_envs
    base, remainder = divmod(n, num_envs)
    return [base + (idx < remainder) for idx in range(num_envs)]


def _repeat_to_count(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n < 0:
        return rows
    if not rows:
        return []
    if len(rows) >= n:
        return rows[:n]
    return [rows[idx % len(rows)] for idx in range(n)]


def _gepa_info_dict(info: object) -> dict[str, Any]:
    if info is None:
        return {}
    if isinstance(info, str):
        parsed = json.loads(info)
        if isinstance(parsed, dict):
            return dict(cast(dict[str, Any], parsed))
        raise ValueError("GEPA dataset row info must decode to a dict.")
    if isinstance(info, Mapping):
        return dict(cast(Mapping[str, Any], info))
    raise ValueError("GEPA dataset row info must be a dict.")


def _gepa_route(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    if value is None:
        return ()
    raise ValueError("GEPA dataset row info['env_id'] must be a string or list.")


def _gepa_route_value(route: tuple[str, ...]) -> str | list[str] | None:
    if not route:
        return None
    if len(route) == 1:
        return route[0]
    return list(route)


def _add_env_group_route(row: dict[str, Any], env_name: str) -> dict[str, Any]:
    routed = dict(row)
    info = _gepa_info_dict(routed.get("info"))
    child_route = _gepa_route(info.get(ENV_GROUP_INFO_KEY))
    info[ENV_GROUP_INFO_KEY] = _gepa_route_value((env_name, *child_route))
    routed["info"] = info
    return routed


def _load_gepa_dataset(
    env: vf.Environment,
    envs: list[vf.Environment],
    env_names: list[str],
    split: str,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(envs) == 1:
        dataset = (
            env.get_dataset(n=n, seed=seed)
            if split == "train"
            else env.get_eval_dataset(n=n, seed=seed)
        )
        return dataset.to_list()

    rows: list[dict[str, Any]] = []
    counts = _balanced_counts(n, len(envs))
    for idx, (sub_env, env_name, count) in enumerate(zip(envs, env_names, counts)):
        dataset_seed = seed + idx
        dataset = (
            sub_env.get_dataset(n=-1, seed=dataset_seed)
            if split == "train"
            else sub_env.get_eval_dataset(n=-1, seed=dataset_seed)
        )
        selected_rows = _repeat_to_count(dataset.to_list(), count)
        for selected_row in selected_rows:
            row = _add_env_group_route(selected_row, env_name)
            if "example_id" in row:
                row.setdefault("source_example_id", row["example_id"])
            row["example_id"] = len(rows)
            row["task"] = env_name
            rows.append(row)

    if not rows:
        raise ValueError(f"No {split} examples available for GEPA.")
    return rows


def run_gepa_optimization(
    env_id: str,
    env_configs: list[GEPAEnvConfig],
    model: str,
    reflection_model: str,
    client_config: ClientConfig,
    reflection_client_config: ClientConfig,
    max_metric_calls: int,
    minibatch_size: int,
    perfect_score: float | None,
    state_columns: list[str],
    num_train: int,
    num_val: int,
    max_concurrent: int,
    sampling_args: dict,
    seed: int,
    run_dir: Path | None,
    save_results: bool,
    tui_mode: bool = False,
):
    # Create run_dir early
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    # Create display with config (valset info updated after env loads)
    display = GEPADisplay(
        env_id=env_id,
        model=model,
        reflection_model=reflection_model,
        max_metric_calls=max_metric_calls,
        num_train=num_train,  # requested count, actual may differ
        num_val=num_val,  # requested count, actual may differ
        log_file=run_dir / "gepa.log" if run_dir else None,
        perfect_score=perfect_score,
        screen=tui_mode,
    )

    with display:
        env, envs, env_names = _load_gepa_environment(env_configs)

        # Check system prompt
        initial_prompt = _shared_initial_prompt(envs)
        if not initial_prompt:
            logger.warning("No system prompt attached to environment.")
            logger.warning(
                "Will add a system message block to the start of chat messages."
            )

        # Get datasets
        logger.debug(f"Loading trainset ({num_train} examples)")
        trainset = _load_gepa_dataset(
            env=env,
            envs=envs,
            env_names=env_names,
            split="train",
            n=num_train,
            seed=seed,
        )

        logger.debug(f"Loading valset ({num_val} examples)")
        valset = _load_gepa_dataset(
            env=env,
            envs=envs,
            env_names=env_names,
            split="eval",
            n=num_val,
            seed=seed,
        )

        # Update display with actual valset info
        valset_example_ids = [
            item.get("example_id", i) for i, item in enumerate(valset)
        ]
        display.set_valset_info(len(valset), valset_example_ids)
        # Update actual counts (may differ from requested if dataset is smaller)
        display.num_train = len(trainset)
        display.num_val = len(valset)

        # Set up client
        client = resolve_client(client_config)

        logger.debug(f"Results will be saved to: {run_dir}")

        # Create adapter
        adapter = VerifiersGEPAAdapter(
            env=env,
            client=client,
            model=model,
            sampling_args=sampling_args,
            max_concurrent=max_concurrent,
            state_columns=state_columns,
            display=display,
        )

        # Create reflection LM
        reflection_lm = make_reflection_lm(
            client_config=reflection_client_config, model=reflection_model
        )

        # Configure perfect score handling
        skip_perfect_score = perfect_score is not None

        # Run GEPA
        logger.debug(
            f"Starting GEPA optimization (budget={max_metric_calls}, minibatch={minibatch_size})"
        )
        logger.debug(f"Eval model: {model}")
        logger.debug(f"Reflection model: {reflection_model}")
        if perfect_score is not None:
            logger.debug(
                f"Perfect score: {perfect_score} (will not reflect on minibatches with perfect score)"
            )

        seed_candidate = {"system_prompt": initial_prompt}
        optimize_kwargs: dict = {
            "seed_candidate": seed_candidate,
            "trainset": trainset,
            "valset": valset,
            "adapter": adapter,
            "reflection_lm": reflection_lm,
            "max_metric_calls": max_metric_calls,
            "reflection_minibatch_size": minibatch_size,
            "run_dir": str(run_dir) if run_dir else None,
            "seed": seed,
            "display_progress_bar": False,
            "skip_perfect_score": skip_perfect_score,
            "logger": display,
        }
        if perfect_score is not None:
            optimize_kwargs["perfect_score"] = perfect_score
        result = optimize(**optimize_kwargs)

        # Save results
        save_path = None
        if run_dir and save_results:
            run_config = {
                "env_id": env_id,
                "envs": [config.model_dump(mode="json") for config in env_configs],
                "env_args": env_configs[0].args if len(env_configs) == 1 else {},
                "model": model,
                "reflection_model": reflection_model,
                "num_train": num_train,
                "num_val": num_val,
                "max_metric_calls": max_metric_calls,
                "minibatch_size": minibatch_size,
                "perfect_score": perfect_score,
                "state_columns": state_columns,
                "seed": seed,
            }
            save_gepa_results(run_dir, result, config=run_config)
            save_path = str(run_dir)
            logger.debug(f"Results saved to {run_dir}")

        # Set result info for final summary
        best_prompt = result.best_candidate.get("system_prompt", "")  # type: ignore[unresolved-attribute]
        display.set_result(best_prompt=best_prompt, save_path=save_path)

    return result


if __name__ == "__main__":
    main()
