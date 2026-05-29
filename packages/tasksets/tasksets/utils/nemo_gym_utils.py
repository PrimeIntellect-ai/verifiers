from pathlib import Path

from verifiers.v1.utils.nemo_gym_utils import nemo_gym_package_root

DEFAULT_NEMO_GYM_DATA_NAME = "example.jsonl"


def resolve_nemo_gym_data_path(
    nemo_env: str,
    data_name: str = DEFAULT_NEMO_GYM_DATA_NAME,
) -> Path:
    path = nemo_gym_package_root() / "resources_servers" / nemo_env / "data" / data_name
    if not path.exists():
        raise FileNotFoundError(f"NeMo Gym data file not found: {path}")
    return path
