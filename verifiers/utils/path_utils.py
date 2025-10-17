import uuid
from pathlib import Path

from verifiers.types import EvalConfig


def get_results_path(
    env_id: str,
    model: str,
    base_path: Path = Path("./outputs"),
) -> Path:
    uuid_str = str(uuid.uuid4())[:8]
    env_model_str = f"{env_id}--{model.replace('/', '--')}"
    return base_path / "evals" / env_model_str / uuid_str


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
