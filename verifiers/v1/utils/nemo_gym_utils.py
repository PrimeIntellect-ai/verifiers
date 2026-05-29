from pathlib import Path
from typing import cast

from ..types import ConfigData


def nemo_gym_package_root() -> Path:
    try:
        from nemo_gym import PARENT_DIR as nemo_gym_root  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "NeMo Gym integration requires nemo-gym. Install as `verifiers[nemogym]`."
        ) from exc
    return Path(nemo_gym_root)


def agent_ref_name(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    name = cast(ConfigData, value).get("name")
    return name if isinstance(name, str) and name else None
