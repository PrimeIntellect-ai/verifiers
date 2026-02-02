import logging
from pathlib import Path

from verifiers.envs.experimental.local_harbor_env import LocalHarborEnv

logger = logging.getLogger(__name__)


class TerminusLocalEnv(LocalHarborEnv):
    def __init__(
        self,
        dataset_path: str | Path = Path(__file__).parent / "tasks",
        tasks: list[str] | None = None,
        agent_name: str = "terminus-2",
        model_name: str = "anthropic/claude-sonnet-4",
        environment_type: str = "prime",
        timeout_seconds: float = 3600.0,
        max_turns: int | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset_path=dataset_path,
            tasks=tasks,
            agent_name=agent_name,
            model_name=model_name,
            environment_type=environment_type,
            timeout_seconds=timeout_seconds,
            max_turns=max_turns,
            **kwargs,
        )


def load_environment(
    dataset_path: str | Path = Path(__file__).parent / "tasks",
    tasks: list[str] | None = None,
    agent_name: str = "terminus-2",
    model_name: str = "anthropic/claude-sonnet-4",
    environment_type: str = "prime",
    timeout_seconds: float = 3600.0,
    timeout_multiplier: float = 1.0,
    delete_sandbox: bool = True,
    disable_verifier: bool = False,
    trials_dir: Path | None = None,
    max_turns: int | None = None,
    interception_port: int = 8765,
) -> TerminusLocalEnv:
    return TerminusLocalEnv(
        dataset_path=dataset_path,
        tasks=tasks,
        agent_name=agent_name,
        model_name=model_name,
        environment_type=environment_type,
        timeout_seconds=timeout_seconds,
        timeout_multiplier=timeout_multiplier,
        delete_sandbox=delete_sandbox,
        disable_verifier=disable_verifier,
        trials_dir=trials_dir,
        max_turns=max_turns,
        interception_port=interception_port,
    )
