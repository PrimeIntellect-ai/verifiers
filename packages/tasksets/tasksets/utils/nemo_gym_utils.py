from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, cast

ConfigMap: TypeAlias = dict[str, object]
DEFAULT_NEMO_GYM_DATA_NAME = "example.jsonl"


def nemo_gym_package_root() -> Path:
    try:
        from nemo_gym import PARENT_DIR as nemo_gym_root  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "NeMo Gym integration requires nemo-gym. Install as `verifiers[nemogym]`."
        ) from exc
    return Path(nemo_gym_root)


def resolve_nemo_gym_data_path(
    nemo_env: str,
    data_name: str = DEFAULT_NEMO_GYM_DATA_NAME,
) -> Path:
    path = nemo_gym_package_root() / "resources_servers" / nemo_env / "data" / data_name
    if not path.exists():
        raise FileNotFoundError(f"NeMo Gym data file not found: {path}")
    return path


def agent_ref_name(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    name = cast(ConfigMap, value).get("name")
    return name if isinstance(name, str) and name else None
