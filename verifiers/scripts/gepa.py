"""
GEPA prompt optimization CLI.

Aligned with vf-eval patterns for consistency.
"""

import argparse
from collections.abc import Mapping
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

from gepa.api import optimize

import verifiers as vf
from verifiers import setup_logging
from verifiers.clients import resolve_client
from verifiers.envs.env_group import ENV_GROUP_INFO_KEY
from verifiers.gepa.adapter import VerifiersGEPAAdapter, make_reflection_lm
from verifiers.gepa.display import GEPADisplay
from verifiers.gepa.gepa_utils import save_gepa_results
from verifiers.types import ClientConfig
from verifiers.utils.eval_utils import load_endpoints
from verifiers.utils.import_utils import load_toml
from verifiers.utils.path_utils import get_gepa_results_path

logger = logging.getLogger(__name__)


def _gepa_extra_headers_from_group(endpoint_group: list, alias: str) -> dict[str, str]:
    maps = [dict(entry.get("extra_headers", {})) for entry in endpoint_group]
    unique = {tuple(sorted(m.items())) for m in maps}
    if len(unique) > 1:
        raise ValueError(
            f"Endpoint alias {alias!r} has different headers across endpoint variants; "
            "GEPA requires a single consistent header set."
        )
    return maps[0] if maps else {}


DEFAULT_API_KEY_VAR = "PRIME_API_KEY"
DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_ENDPOINTS_PATH = "./configs/endpoints.toml"
DEFAULT_ENV_DIR_PATH = "./environments"
DEFAULT_NUM_TRAIN = 100
DEFAULT_NUM_VAL = 50
DEFAULT_MAX_METRIC_CALLS = 500
DEFAULT_MINIBATCH_SIZE = 3


def _ensure_table(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be a TOML table.")
    return cast(dict[str, Any], value)


def _load_env_configs(raw_config: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    raw_env = raw_config.get("env")
    if isinstance(raw_env, dict):
        raw_envs = [raw_env]
    elif isinstance(raw_env, list):
        raw_envs = raw_env
    else:
        raise ValueError(f"Config file must contain an [env] or [[env]] table: {path}")

    if not raw_envs:
        raise ValueError(f"Config file must contain at least one env table: {path}")

    env_configs = []
    for idx, env_table in enumerate(raw_envs):
        if not isinstance(env_table, dict):
            raise ValueError(f"Each [[env]] entry must be a TOML table: {path}")
        env_id = env_table.get("env_id")
        if not isinstance(env_id, str) or not env_id:
            raise ValueError(
                f"env entry {idx} must contain a non-empty env_id string: {path}"
            )
        env_configs.append(
            {
                "env_id": env_id,
                "env_args": _ensure_table(
                    env_table.get("env_args", {}), "env.env_args"
                ),
                "extra_env_kwargs": _ensure_table(
                    env_table.get("extra_env_kwargs", {}),
                    "env.extra_env_kwargs",
                ),
            }
        )
    return env_configs


def _env_group_label(env_configs: list[dict[str, Any]]) -> str:
    if len(env_configs) == 1:
        return cast(str, env_configs[0]["env_id"])
    names = [str(config["env_id"]).rstrip("/").split("/")[-1] for config in env_configs]
    return "+".join(names)


def _apply_execution_compat(
    config: dict[str, Any],
    execution_table: dict[str, Any],
    warnings: list[str],
) -> None:
    if not execution_table:
        return

    warnings.append(
        "[execution] in GEPA TOML configs is deprecated and undocumented. "
        "Move max_concurrent and seed under [gepa], and move sampling_args values "
        "under [sampling]."
    )

    for key in ("max_concurrent", "seed"):
        if key not in execution_table:
            continue
        if key in config:
            raise ValueError(
                f"GEPA TOML config sets {key!r} in both [gepa] and [execution]."
            )
        config[key] = execution_table[key]

    if "sampling_args" in execution_table:
        if "sampling_args" in config:
            raise ValueError(
                "GEPA TOML config sets sampling parameters in both [sampling] "
                "and [execution].sampling_args."
            )
        config["sampling_args"] = _ensure_table(
            execution_table["sampling_args"], "execution.sampling_args"
        )


def load_gepa_toml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"TOML config file not found: {path}")

    with path.open("rb") as f:
        raw_config = load_toml(f)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Expected top-level TOML table in {path}")

    env_configs = _load_env_configs(raw_config, path)
    warnings: list[str] = []

    config: dict[str, Any] = {
        "env_id": _env_group_label(env_configs),
        "envs": env_configs,
    }
    if len(env_configs) == 1:
        config["env_args"] = env_configs[0]["env_args"]
        config["extra_env_kwargs"] = env_configs[0]["extra_env_kwargs"]

    for key in (
        "model",
        "reflection_model",
        "endpoints_path",
        "api_key_var",
        "api_base_url",
        "env_dir_path",
        "run_dir",
        "save_results",
        "verbose",
        "tui",
    ):
        if key in raw_config:
            config[key] = raw_config[key]

    gepa_table = _ensure_table(raw_config.get("gepa", {}), "gepa")
    for key in (
        "max_calls",
        "minibatch_size",
        "perfect_score",
        "state_columns",
        "num_train",
        "num_val",
        "max_concurrent",
        "seed",
    ):
        if key in gepa_table:
            config[key] = gepa_table[key]

    sampling_table = _ensure_table(raw_config.get("sampling", {}), "sampling")
    if sampling_table:
        config["sampling_args"] = sampling_table

    execution_table = _ensure_table(raw_config.get("execution", {}), "execution")
    _apply_execution_compat(config, execution_table, warnings)
    if warnings:
        config["_warnings"] = warnings

    # Resolve config-relative paths for consistency with vf-eval.
    endpoints_path = config.get("endpoints_path")
    if isinstance(endpoints_path, str):
        endpoints_path_obj = Path(endpoints_path)
        if not endpoints_path_obj.is_absolute():
            config["endpoints_path"] = str((path.parent / endpoints_path_obj).resolve())

    run_dir = config.get("run_dir")
    if isinstance(run_dir, str):
        run_dir_obj = Path(run_dir)
        if not run_dir_obj.is_absolute():
            config["run_dir"] = str((path.parent / run_dir_obj).resolve())

    return config


def resolve_gepa_config_args(args: argparse.Namespace) -> argparse.Namespace:
    raw = args.env_id_or_config
    config_path = Path(raw)
    if config_path.suffix == ".toml":
        config = load_gepa_toml_config(config_path)
    else:
        args.env_id = raw
        return args

    for key in (
        "env_id",
        "envs",
        "env_args",
        "extra_env_kwargs",
        "model",
        "reflection_model",
        "endpoints_path",
        "api_key_var",
        "api_base_url",
        "env_dir_path",
        "run_dir",
        "verbose",
        "tui",
        "max_calls",
        "minibatch_size",
        "perfect_score",
        "state_columns",
        "num_train",
        "num_val",
        "max_concurrent",
        "sampling_args",
        "seed",
    ):
        if key in config:
            setattr(args, key, config[key])

    if "save_results" in config:
        save_results = config["save_results"]
        if not isinstance(save_results, bool):
            raise ValueError("'save_results' must be a boolean.")
        args.no_save = not save_results
    args.config_warnings = config.get("_warnings", [])

    return args


def main():
    parser = argparse.ArgumentParser(description="Run GEPA prompt optimization")

    # Environment (aligned with vf-eval config entrypoint)
    parser.add_argument(
        "env_id_or_config",
        type=str,
        help="Environment module name or path to TOML config file.",
    )
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default={},
        help='Environment module arguments as JSON object (e.g., \'{"key": "value"}\')',
    )
    parser.add_argument(
        "--env-dir-path",
        "-p",
        type=str,
        default=DEFAULT_ENV_DIR_PATH,
        help="Path to environments directory",
    )
    parser.add_argument(
        "--extra-env-kwargs",
        "-x",
        type=json.loads,
        default={},
        help="Extra environment kwargs as JSON object. Passed to environment constructor.",
    )

    # Model configuration (aligned with vf-eval)
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openai/gpt-4.1-mini",
        help="Model for evaluation rollouts",
    )
    parser.add_argument(
        "--reflection-model",
        "-M",
        type=str,
        default=None,
        help="Model for reflection/teacher LLM (defaults to --model)",
    )
    parser.add_argument(
        "--endpoints-path",
        "-e",
        type=str,
        default=DEFAULT_ENDPOINTS_PATH,
        help="Path to API endpoints TOML registry",
    )
    parser.add_argument("--api-key-var", "-k", type=str, default=None)
    parser.add_argument("--api-base-url", "-b", type=str, default=None)

    # GEPA optimization (the key params)
    parser.add_argument(
        "--max-calls",
        "-B",
        type=int,
        default=DEFAULT_MAX_METRIC_CALLS,
        help="Maximum metric calls (evaluation budget)",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=DEFAULT_MINIBATCH_SIZE,
        help="Minibatch size for reflection",
    )
    parser.add_argument(
        "--perfect-score",
        type=float,
        default=None,
        help="If set, dont reflect on minibatches with this score",
    )
    parser.add_argument(
        "--state-columns",
        type=str,
        nargs="*",
        default=[],
        help="Additional state columns to copy to reflection dataset",
    )

    # Dataset sizes
    parser.add_argument(
        "--num-train",
        "-n",
        type=int,
        default=DEFAULT_NUM_TRAIN,
        help="Training examples",
    )
    parser.add_argument(
        "--num-val", "-N", type=int, default=DEFAULT_NUM_VAL, help="Validation examples"
    )

    # Execution (aligned with vf-eval)
    parser.add_argument("--max-concurrent", "-c", type=int, default=32)
    parser.add_argument("--sampling-args", "-S", type=json.loads, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")

    # Output
    parser.add_argument(
        "--run-dir", "-o", type=str, default=None, help="Override output directory"
    )
    parser.add_argument("--no-save", action="store_true", help="Disable saving results")
    parser.add_argument(
        "--tui",
        "-u",
        action="store_true",
        default=False,
        help="Use alternate screen mode (TUI) for display",
    )

    args = parser.parse_args()
    try:
        args = resolve_gepa_config_args(args)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    setup_logging("DEBUG" if args.verbose else os.getenv("VF_LOG_LEVEL", "INFO"))
    for warning in getattr(args, "config_warnings", []):
        logger.warning(warning)

    env_configs = getattr(args, "envs", None)
    if env_configs is None:
        env_configs = [
            {
                "env_id": args.env_id,
                "env_args": args.env_args,
                "extra_env_kwargs": args.extra_env_kwargs,
            }
        ]

    # Load endpoints and resolve model config
    endpoints = load_endpoints(args.endpoints_path)
    api_key_override = args.api_key_var is not None
    api_base_url_override = args.api_base_url is not None

    main_extra_headers: dict[str, str] = {}
    if args.model in endpoints:
        endpoint_group = endpoints[args.model]
        endpoint = endpoint_group[0]
        main_extra_headers = _gepa_extra_headers_from_group(endpoint_group, args.model)

        if api_key_override:
            api_key_var = args.api_key_var
        else:
            endpoint_keys = {entry["key"] for entry in endpoint_group}
            if len(endpoint_keys) > 1:
                raise ValueError(
                    f"Endpoint alias '{args.model}' maps to multiple API key vars {sorted(endpoint_keys)}, "
                    "which is not yet supported by GEPA config. Please set --api-key-var explicitly."
                )
            api_key_var = endpoint["key"]

        if api_base_url_override:
            api_base_url = args.api_base_url
        else:
            endpoint_urls = {entry["url"] for entry in endpoint_group}
            if len(endpoint_urls) > 1:
                raise ValueError(
                    f"Endpoint alias '{args.model}' maps to multiple URLs {sorted(endpoint_urls)} and GEPA currently requires a single --api-base-url."
                )
            api_base_url = endpoint["url"]

        endpoint_models = {entry["model"] for entry in endpoint_group}
        if len(endpoint_models) > 1:
            raise ValueError(
                f"Endpoint alias '{args.model}' maps to multiple model ids {sorted(endpoint_models)}, "
                "which is not yet supported by GEPA config."
            )
        model = endpoint["model"]
    else:
        api_key_var = args.api_key_var if api_key_override else DEFAULT_API_KEY_VAR
        api_base_url = (
            args.api_base_url if api_base_url_override else DEFAULT_API_BASE_URL
        )
        model = args.model

    reflection_extra_headers: dict[str, str] = main_extra_headers
    # Resolve reflection model and its client config
    if args.reflection_model and args.reflection_model in endpoints:
        reflection_endpoint_group = endpoints[args.reflection_model]
        reflection_endpoint = reflection_endpoint_group[0]
        reflection_extra_headers = _gepa_extra_headers_from_group(
            reflection_endpoint_group, args.reflection_model
        )

        reflection_endpoint_models = {
            entry["model"] for entry in reflection_endpoint_group
        }
        if len(reflection_endpoint_models) > 1:
            raise ValueError(
                f"Endpoint alias '{args.reflection_model}' maps to multiple model ids {sorted(reflection_endpoint_models)}, "
                "which is not yet supported by GEPA reflection config."
            )
        reflection_endpoint_keys = {entry["key"] for entry in reflection_endpoint_group}
        if len(reflection_endpoint_keys) > 1:
            raise ValueError(
                f"Endpoint alias '{args.reflection_model}' maps to multiple API key vars {sorted(reflection_endpoint_keys)}, "
                "which is not yet supported by GEPA reflection config."
            )
        reflection_endpoint_urls = {entry["url"] for entry in reflection_endpoint_group}
        if len(reflection_endpoint_urls) > 1:
            raise ValueError(
                f"Endpoint alias '{args.reflection_model}' maps to multiple URLs {sorted(reflection_endpoint_urls)} and GEPA currently requires a single --api-base-url."
            )

        reflection_model = reflection_endpoint["model"]
        reflection_api_key_var = reflection_endpoint["key"]
        reflection_api_base_url = reflection_endpoint["url"]
    elif args.reflection_model:
        # Reflection model specified but not in endpoints - use defaults
        reflection_model = args.reflection_model
        reflection_api_key_var = api_key_var
        reflection_api_base_url = api_base_url
        reflection_extra_headers = main_extra_headers
    else:
        # No reflection model specified - use main model config
        reflection_model = model
        reflection_api_key_var = api_key_var
        reflection_api_base_url = api_base_url
        reflection_extra_headers = main_extra_headers

    # Generate run_dir (save to environment folder like vf-eval)
    save_results = not args.no_save
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif save_results:
        run_dir = get_gepa_results_path(args.env_id, model, args.env_dir_path)
    else:
        run_dir = None

    # Run optimization
    client_config = ClientConfig(
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        extra_headers=main_extra_headers,
    )
    reflection_client_config = ClientConfig(
        api_key_var=reflection_api_key_var,
        api_base_url=reflection_api_base_url,
        extra_headers=reflection_extra_headers,
    )

    run_gepa_optimization(
        env_id=args.env_id,
        env_configs=env_configs,
        model=model,
        reflection_model=reflection_model,
        client_config=client_config,
        reflection_client_config=reflection_client_config,
        max_metric_calls=args.max_calls,
        minibatch_size=args.minibatch_size,
        perfect_score=args.perfect_score,
        state_columns=args.state_columns,
        num_train=args.num_train,
        num_val=args.num_val,
        max_concurrent=args.max_concurrent,
        sampling_args=args.sampling_args or {},
        seed=args.seed,
        run_dir=run_dir,
        save_results=save_results,
        tui_mode=args.tui,
    )


def _unique_env_names(env_configs: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for config in env_configs:
        base = str(config["env_id"])
        count = seen.get(base, 0) + 1
        seen[base] = count
        names.append(base if count == 1 else f"{base}:{count}")
    return names


def _load_gepa_environment(
    env_configs: list[dict[str, Any]],
) -> tuple[vf.Environment, list[vf.Environment], list[str]]:
    envs = []
    for config in env_configs:
        env_id = config["env_id"]
        env_args = config["env_args"]
        logger.debug(f"Loading environment: {env_id}")
        env = vf.load_environment(env_id=env_id, **env_args)

        extra_env_kwargs = config["extra_env_kwargs"]
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
    env_configs: list[dict[str, Any]],
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
                "envs": env_configs,
                "env_args": env_configs[0]["env_args"] if len(env_configs) == 1 else {},
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
