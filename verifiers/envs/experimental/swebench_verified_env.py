from __future__ import annotations

from typing import Any

from verifiers.envs.experimental.harnesses.base import Harness
from verifiers.envs.experimental.task_agent_env import TaskAgentEnv
from verifiers.envs.experimental.tasksets.swebench_verified import (
    DEFAULT_AGENT_WORKDIR,
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    SWEBenchVerifiedTaskSet,
)


class SWEBenchVerifiedEnv(TaskAgentEnv):
    """Composable SWE-bench Verified environment with a pluggable agent harness."""

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        dataset_split: str = DEFAULT_DATASET_SPLIT,
        instance_ids: list[str] | None = None,
        repos: list[str] | None = None,
        max_examples: int = -1,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        docker_namespace: str = "swebench",
        docker_arch: str = "x86_64",
        docker_tag: str = "latest",
        harness: Harness | None = None,
        harness_config: dict[str, Any] | None = None,
        max_turns: int = -1,
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 4,
        memory_gb: int = 8,
        disk_size_gb: int = 20,
        gpu_count: int = 0,
        timeout_minutes: int = 180,
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: Any | None = None,
        labels: list[str] | None = None,
        **kwargs,
    ):
        taskset = SWEBenchVerifiedTaskSet(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            instance_ids=instance_ids,
            repos=repos,
            max_examples=max_examples,
            agent_workdir=agent_workdir,
            docker_namespace=docker_namespace,
            docker_arch=docker_arch,
            docker_tag=docker_tag,
            start_command=start_command,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=gpu_count,
            timeout_minutes=timeout_minutes,
            team_id=team_id,
            advanced_configs=advanced_configs,
            labels=labels,
            harness_config=harness_config,
        )

        kwargs.setdefault("env_id", "swebench_verified")
        super().__init__(
            harness=harness,
            taskset=taskset,
            max_turns=max_turns,
            environment_vars=environment_vars,
            **kwargs,
        )
