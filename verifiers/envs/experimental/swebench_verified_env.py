from __future__ import annotations

from typing import Any

from verifiers.envs.experimental.harnesses.base import Harness
from verifiers.envs.experimental.harnesses.opencode import (
    DEFAULT_INSTALL_COMMAND,
    DEFAULT_RUN_COMMAND_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT,
    OpenCodeHarness,
)
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
        asset_dir: str = OpenCodeHarness.DEFAULT_ASSET_DIR,
        disabled_tools: list[str] | None = None,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        install_command: str = DEFAULT_INSTALL_COMMAND,
        run_command_template: str = DEFAULT_RUN_COMMAND_TEMPLATE,
        disable_compaction: bool = OpenCodeHarness.DEFAULT_DISABLE_COMPACTION,
        enable_interleaved: bool = OpenCodeHarness.DEFAULT_ENABLE_INTERLEAVED,
        provider_timeout_ms: int = OpenCodeHarness.DEFAULT_PROVIDER_TIMEOUT_MS,
        max_turns: int = -1,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 1.0,
        interception_port: int | None = None,
        interception_url: str | None = None,
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
        include_hints: bool = True,
        **kwargs,
    ):
        if harness is None:
            harness = OpenCodeHarness(
                asset_dir=asset_dir,
                agent_workdir=agent_workdir,
                disabled_tools=disabled_tools,
                system_prompt=system_prompt,
                install_command=install_command,
                run_command_template=run_command_template,
                disable_compaction=disable_compaction,
                enable_interleaved=enable_interleaved,
                provider_timeout_ms=provider_timeout_ms,
                interception_port=interception_port,
                interception_url=interception_url,
                poll_interval=poll_interval,
                timeout_seconds=timeout_seconds,
            )
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
            include_hints=include_hints,
        )

        kwargs.setdefault("env_id", "swebench_verified")
        super().__init__(
            harness=harness,
            taskset=taskset,
            max_turns=max_turns,
            environment_vars=environment_vars,
            **kwargs,
        )
