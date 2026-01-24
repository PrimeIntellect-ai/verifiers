import uuid
from pathlib import Path

from verifiers.types import EvalConfig


def get_results_path(
    env_id: str,
    model: str,
    base_path: Path = Path("./outputs"),
    subdir: str = "evals",
) -> Path:
    uuid_str = str(uuid.uuid4())[:8]
    env_model_str = f"{env_id}--{model.replace('/', '--')}"
    return base_path / subdir / env_model_str / uuid_str


def get_eval_results_path(config: EvalConfig) -> Path:
    module_name = config.env_id.replace("-", "_")
    local_env_dir = Path(config.env_dir_path) / module_name

    if local_env_dir.exists():
        base_path = local_env_dir / "outputs"
        results_path = get_results_path(config.env_id, config.model, base_path)
    else:
        base_path = Path("./outputs")
        results_path = get_results_path(config.env_id, config.model, base_path)
    return results_path


def get_gepa_results_path(
    env_id: str,
    model: str,
    env_dir_path: str = "./environments",
) -> Path:
    """Generate path for GEPA optimization run.

    If environment directory exists locally, saves to:
        {env_dir}/{module_name}/outputs/gepa/{env_id}--{model}/{uuid8}/
    Otherwise saves to:
        ./outputs/gepa/{env_id}--{model}/{uuid8}/
    """
    module_name = env_id.replace("-", "_")
    local_env_dir = Path(env_dir_path) / module_name

    if local_env_dir.exists():
        base_path = local_env_dir / "outputs"
    else:
        base_path = Path("./outputs")

    return get_results_path(env_id, model, base_path, subdir="gepa")
