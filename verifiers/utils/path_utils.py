import json
import logging
import tempfile
import uuid
from pathlib import Path

from verifiers.types import EvalConfig

logger = logging.getLogger(__name__)


def write_temp_file(content: str, suffix: str = ".txt") -> str:
    """Write content to a named temporary file and return its path.

    Intended to be called via ``await asyncio.to_thread(write_temp_file, ...)``
    so that file I/O does not block the event loop.
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=suffix) as f:
        f.write(content)
        return f.name


def _get_outputs_base_path(
    env_id: str,
    env_dir_path: str = "./environments",
    output_dir: str | None = None,
) -> Path:
    """Resolve where outputs should be stored for an environment."""
    if output_dir is not None:
        return Path(output_dir)
    module_name = env_id.replace("-", "_")
    local_env_dir = Path(env_dir_path) / module_name

    if local_env_dir.exists():
        return local_env_dir / "outputs"
    return Path("./outputs")


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
    base_path = _get_outputs_base_path(
        config.env_id, config.env_dir_path, config.output_dir
    )
    return get_results_path(config.name or config.env_id, config.model, base_path)


def get_eval_runs_dir(
    env_id: str,
    model: str,
    env_dir_path: str = "./environments",
    output_dir: str | None = None,
    name: str | None = None,
) -> Path:
    """Return directory containing all eval run directories for env/model."""
    base_path = _get_outputs_base_path(env_id, env_dir_path, output_dir)
    env_model_str = f"{name or env_id}--{model.replace('/', '--')}"
    return base_path / "evals" / env_model_str


def is_valid_eval_results_path(path: Path) -> bool:
    """Checks if a path is a valid evaluation results path."""
    results_file = path / "results.jsonl"
    metadata_file = path / "metadata.json"
    return (
        path.exists()
        and path.is_dir()
        and results_file.exists()
        and results_file.is_file()
        and metadata_file.exists()
        and metadata_file.is_file()
    )


def _count_saved_rollouts(results_path: Path, rollouts_per_example: int) -> int:
    """Count completed rollout rows in results.jsonl."""
    outputs_path = results_path / "results.jsonl"
    counts: dict[object, int] = {}
    with open(outputs_path, "rb") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                output = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                logger.warning(
                    "Ignoring malformed JSONL row in %s at line %s",
                    outputs_path,
                    line_idx,
                )
                continue
            if not isinstance(output, dict) or "example_id" not in output:
                continue
            example_id = output["example_id"]
            counts[example_id] = min(
                counts.get(example_id, 0) + 1,
                rollouts_per_example,
            )
    return sum(counts.values())


def find_latest_incomplete_eval_results_path(
    env_id: str,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    shuffle: bool = False,
    shuffle_seed: int | None = None,
    env_dir_path: str = "./environments",
    output_dir: str | None = None,
    name: str | None = None,
) -> Path | None:
    """Find the newest resumable, incomplete eval run for the provided config."""
    runs_dir = get_eval_runs_dir(
        env_id=env_id,
        model=model,
        env_dir_path=env_dir_path,
        output_dir=output_dir,
        name=name,
    )
    if not runs_dir.exists():
        return None

    total_rollouts = num_examples * rollouts_per_example
    candidates: list[Path] = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and is_valid_eval_results_path(run_dir):
            candidates.append(run_dir)

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for candidate in candidates:
        metadata_path = candidate / "metadata.json"
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(metadata, dict):
            continue

        if metadata.get("env_id") != env_id:
            continue
        if metadata.get("model") != model:
            continue
        if metadata.get("rollouts_per_example") != rollouts_per_example:
            continue
        if bool(metadata.get("shuffle", False)) != shuffle:
            continue
        if shuffle:
            expected_shuffle_seed = 0 if shuffle_seed is None else shuffle_seed
            if metadata.get("shuffle_seed") != expected_shuffle_seed:
                continue

        saved_num_examples = metadata.get("num_examples")
        if not isinstance(saved_num_examples, int) or saved_num_examples > num_examples:
            continue

        saved_rollouts = _count_saved_rollouts(candidate, rollouts_per_example)
        if saved_rollouts < total_rollouts:
            return candidate

    return None


def get_gepa_results_path(
    env_id: str,
    model: str,
    env_dir_path: str = "./environments",
    output_dir: str | None = None,
) -> Path:
    """Generate path for GEPA optimization run.

    If environment directory exists locally, saves to:
        {env_dir}/{module_name}/outputs/gepa/{env_id}--{model}/{uuid8}/
    Otherwise saves to:
        ./outputs/gepa/{env_id}--{model}/{uuid8}/
    """
    base_path = _get_outputs_base_path(env_id, env_dir_path, output_dir)
    return get_results_path(env_id.replace("/", "--"), model, base_path, subdir="gepa")
