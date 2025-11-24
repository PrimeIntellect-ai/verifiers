import uuid
from pathlib import Path

from verifiers.types import EvalConfig, GEPAConfig


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
        results_path = get_results_path(config.env_id, config.model, base_path, "evals")
    else:
        base_path = Path("./outputs")
        results_path = get_results_path(config.env_id, config.model, base_path, "evals")
    return results_path


def get_gepa_results_path(config: GEPAConfig) -> Path:
    module_name = config.env_id.replace("-", "_")
    local_env_dir = Path(config.env_dir_path) / module_name

    if local_env_dir.exists():
        base_path = local_env_dir / "outputs"
        results_path = get_results_path(config.env_id, config.model, base_path, "gepa")
    else:
        base_path = Path("./outputs")
        results_path = get_results_path(config.env_id, config.model, base_path, "gepa")
    return results_path
