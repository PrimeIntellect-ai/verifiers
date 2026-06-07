import json
from pathlib import Path

from pydantic import Field

import verifiers.v1 as vf

from .command import CommandHarness, CommandHarnessConfig, shell_command

DEFAULT_NEMO_GYM_COMMAND = "python -m nemo_gym.cli run-one"
DEFAULT_NEMO_GYM_DATA_NAME = "example.jsonl"


class NeMoGymHarnessConfig(CommandHarnessConfig):
    command: list[str] = Field(default_factory=list)
    nemo_env: str | None = None
    config_name: str | None = None
    config_paths: list[str] = Field(default_factory=list)
    server_name: str | None = None
    agent_name: str | None = None
    timeout_seconds: float | None = None
    global_config: vf.JsonData = Field(default_factory=dict)


class NeMoGymHarness(CommandHarness[NeMoGymHarnessConfig]):
    config: NeMoGymHarnessConfig

    def command(self, task: vf.Task, state: vf.State) -> list[str]:
        _ = state
        if self.config.command:
            return list(self.config.command)
        row = getattr(task, "nemo_gym_row", None)
        if not isinstance(row, dict):
            raise ValueError("NeMoGymHarness tasks must contain nemo_gym_row.")
        payload = json.dumps(
            {
                "row": row,
                "config_paths": self.config_paths(),
                "server_name": self.config.server_name,
                "agent_name": self.config.agent_name,
                "global_config": self.config.global_config,
            },
            ensure_ascii=False,
        )
        script = f"""
set -eo pipefail
export NEMO_GYM_ROW_JSON={payload!r}
{DEFAULT_NEMO_GYM_COMMAND}
"""
        return shell_command(script)

    def config_paths(self) -> list[str]:
        if self.config.config_paths:
            return list(self.config.config_paths)
        if self.config.nemo_env is None:
            raise ValueError("NeMoGymHarness requires config_paths or nemo_env.")
        return [
            str(
                resolve_nemo_gym_config_path(
                    self.config.nemo_env,
                    self.config.config_name,
                )
            )
        ]


def nemo_gym_package_root() -> Path:
    try:
        from nemo_gym import PARENT_DIR as nemo_gym_root  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "NeMo Gym integration requires nemo-gym. Install as `verifiers[nemogym]`."
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
    if not configs:
        raise FileNotFoundError(f"No NeMo Gym configs found in: {config_dir}")
    if len(configs) == 1:
        return configs[0]
    raise ValueError(
        f"Multiple NeMo Gym configs found for {nemo_env!r}; pass config_name. "
        f"Options: {[path.name for path in configs]}"
    )


def load_harness(config: NeMoGymHarnessConfig) -> NeMoGymHarness:
    return NeMoGymHarness(config=config)
