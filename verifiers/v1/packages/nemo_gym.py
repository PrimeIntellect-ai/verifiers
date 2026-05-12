from __future__ import annotations

from pathlib import Path

DEFAULT_NEMO_GYM_DATA_NAME = "example.jsonl"


def nemo_gym_package_root() -> Path:
    try:
        from nemo_gym import PARENT_DIR as nemo_gym_root
    except ImportError as exc:
        raise ImportError(
            "NeMo Gym integration requires nemo-gym. Install with: uv add nemo-gym"
        ) from exc
    return Path(nemo_gym_root)


def resolve_nemo_gym_config_path(
    nemo_env: str,
    config_name: str | None = None,
) -> Path:
    config_dir = nemo_gym_package_root() / "resources_servers" / nemo_env / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"NeMo Gym config directory not found: {config_dir}")
    if config_name:
        path = config_dir / config_name
        if path.suffix != ".yaml":
            path = path.with_suffix(".yaml")
        if not path.exists():
            raise FileNotFoundError(f"NeMo Gym config not found: {path}")
        return path
    preferred = config_dir / f"{nemo_env}.yaml"
    if preferred.exists():
        return preferred
    configs = sorted(config_dir.glob("*.yaml"))
    if len(configs) == 1:
        return configs[0]
    raise ValueError(
        f"Multiple NeMo Gym configs found for {nemo_env!r}; pass config_name. "
        f"Options: {[path.name for path in configs]}"
    )


def resolve_nemo_gym_data_path(
    nemo_env: str,
    data_name: str = DEFAULT_NEMO_GYM_DATA_NAME,
) -> Path:
    path = nemo_gym_package_root() / "resources_servers" / nemo_env / "data" / data_name
    if not path.exists():
        raise FileNotFoundError(f"NeMo Gym data file not found: {path}")
    return path


def infer_nemo_gym_agent_from_config(config_path: str | Path) -> tuple[str, str]:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError(
            "NeMo Gym config inference requires omegaconf. "
            "Install with: uv add nemo-gym"
        ) from exc

    path = Path(config_path)
    raw_config = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(raw_config, dict):
        raise ValueError(f"NeMo Gym config must be a mapping: {path}")
    for top_level_name, top_level_value in raw_config.items():
        if not isinstance(top_level_value, dict):
            continue
        agents = top_level_value.get("responses_api_agents")
        if not isinstance(agents, dict) or not agents:
            continue
        agent_name = next(iter(agents))
        return str(top_level_name), str(agent_name)
    raise ValueError(f"No responses_api_agents entry found in {path}")


def first_nemo_gym_agent(
    config_paths: list[str] | tuple[str, ...],
) -> tuple[str, str] | None:
    for config_path in config_paths:
        try:
            return infer_nemo_gym_agent_from_config(config_path)
        except FileNotFoundError:
            continue
        except ValueError:
            continue
    return None
