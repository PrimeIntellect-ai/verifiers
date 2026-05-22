from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import Field, field_validator
from verifiers.decorators import cleanup, setup
from verifiers.envs.experimental.composable.task import SandboxSpec
from verifiers.envs.experimental.composable.tasksets.swe import make_swe_taskset

from ...config import TasksetConfig
from ...state import State
from ...task import Task
from ...taskset import Taskset
from ...types import ConfigData, ConfigMap, Handler


SPLIT_BACKENDS = {
    "multiswe",
    "swelego-real",
    "swerebench-v2",
    "swesmith-py",
    "swesmith-go",
    "swesmith-java",
    "swesmith-js",
    "swesmith-ts",
    "swesmith-rs",
    "swesmith-cpp",
    "swesmith-php",
}


class SWETasksetConfig(TasksetConfig):
    task_type: str = "r2e"
    dataset_name: str | None = None
    split: str | None = None
    filter_repos: list[str] | None = None
    filter_fn: str | None = None
    ds_keep_in_memory: bool | None = None
    ds_num_proc: int | None = None
    timeout_minutes: int | None = None
    repo_path: str | None = None
    alt_path: str | None = None
    hide_tests_from_agent: bool | None = None
    skip_install: bool | None = None
    openswe_config: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("env", mode="before")
    @classmethod
    def validate_env(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): str(item) for key, item in value.items()}
        return value


class SWETaskset(Taskset[SWETasksetConfig]):
    config: SWETasksetConfig

    def __init__(self, config: SWETasksetConfig | None = None):
        self.config = cast(SWETasksetConfig, self._coerce_config(config))
        self.legacy_taskset = make_swe_taskset(
            backend=self.config.task_type,
            **self._factory_kwargs(),
        )
        self.legacy_rubric = self.legacy_taskset.get_rubric()
        super().__init__(config=self.config)
        self.source = self.load_rows
        self.taskset_id = self.config.taskset_id or self.legacy_taskset.name
        self._add_legacy_reward_signals()

    def _factory_kwargs(self) -> ConfigData:
        kwargs: ConfigData = {}
        for name in (
            "dataset_name",
            "filter_fn",
            "ds_num_proc",
            "timeout_minutes",
        ):
            value = getattr(self.config, name)
            if value is not None:
                kwargs[name] = value
        if self.config.ds_keep_in_memory is not None:
            kwargs["ds_keep_in_memory"] = self.config.ds_keep_in_memory
        if self.config.split is not None:
            if self.config.task_type not in SPLIT_BACKENDS:
                raise ValueError(
                    f"task_type={self.config.task_type!r} does not accept split."
                )
            kwargs["split"] = self.config.split
        if self.config.task_type == "r2e":
            for name in ("repo_path", "alt_path", "hide_tests_from_agent"):
                value = getattr(self.config, name)
                if value is not None:
                    kwargs[name] = value
        if self.config.task_type == "swebench" and self.config.skip_install is not None:
            kwargs["skip_install"] = self.config.skip_install
        if (
            self.config.task_type == "openswe"
            and self.config.openswe_config is not None
        ):
            kwargs["config"] = self.config.openswe_config
        return kwargs

    def _add_legacy_reward_signals(self) -> None:
        group_rewards = self.legacy_rubric._get_group_reward_funcs()
        if group_rewards:
            raise ValueError("SWETaskset does not yet adapt group reward rubrics.")
        for func, weight in zip(
            self.legacy_rubric._get_individual_reward_funcs(),
            self.legacy_rubric._get_individual_reward_weights(),
        ):
            self.rewards.append(self._legacy_reward_signal(func, weight))

    def _legacy_reward_signal(self, func: Handler, weight: float) -> Handler:
        async def run(task: Task, state: State) -> float:
            if "answer" in task:
                state.setdefault("answer", task["answer"])
            return float(
                await self.legacy_rubric._call_individual_reward_func(func, state)
            )

        run.__name__ = getattr(func, "__name__", "legacy_reward")
        setattr(run, "reward", True)
        setattr(run, "reward_weight", float(weight))
        setattr(run, "reward_stage", "rollout")
        setattr(run, "reward_priority", 0)
        return run

    def load_rows(self) -> list[ConfigData]:
        rows: list[ConfigData] = []
        dataset = self.legacy_taskset.get_dataset()
        for index in range(len(dataset)):
            row = dict(dataset[index])
            info = dict(cast(ConfigMap, row.get("info") or {}))
            if not self._keeps_repo(info):
                continue
            instruction = str(self.legacy_taskset.get_instruction(info))
            workdir = str(self.legacy_taskset.get_workdir(info))
            task_row: ConfigData = {
                "example_id": row.get("example_id", index),
                "task_id": task_id(info, index),
                "question": row.get("question", instruction),
                "instruction": instruction,
                "prompt": [{"role": "user", "content": instruction}],
                "answer": row.get("answer", ""),
                "info": info,
                "program": {"env": self._program_env(info, workdir)},
            }
            sandbox = self._sandbox_config(info, workdir)
            if sandbox is not None:
                task_row["sandbox"] = sandbox
            rows.append(task_row)
        return rows

    def _keeps_repo(self, info: ConfigMap) -> bool:
        if not self.config.filter_repos:
            return True
        excluded = set(self.config.filter_repos)
        names = {
            str(value)
            for value in (
                info.get("repo"),
                info.get("repo_name"),
                info.get("repo_full_name"),
            )
            if value is not None
        }
        return not names.intersection(excluded)

    def _program_env(self, info: ConfigMap, workdir: str) -> dict[str, str]:
        _ = info
        env = {
            str(key): str(value)
            for key, value in {
                **self.legacy_taskset.get_env_vars(),
                **self.config.env,
            }.items()
        }
        agent_path = env.pop("PATH", None)
        agent_workdir = env.pop("AGENT_WORKDIR", workdir)
        if agent_path is not None:
            env.setdefault("AGENT_PATH", agent_path)
        env.setdefault("AGENT_WORKDIR", str(agent_workdir))
        return env

    def _sandbox_config(self, info: ConfigMap, workdir: str) -> ConfigData | None:
        spec = self.legacy_taskset.get_sandbox_spec(dict(info))
        if spec is None:
            return None
        return sandbox_spec_config(spec, workdir)

    @setup(priority=250)
    async def setup_swe_sandbox(
        self, task: Task, state: State, sandbox: object | None = None
    ) -> None:
        if sandbox is None:
            raise RuntimeError("SWE setup requires the active program sandbox.")
        lease = getattr(sandbox, "lease", None)
        client = getattr(lease, "client", None)
        sandbox_id = getattr(sandbox, "id", None)
        if client is None or sandbox_id is None:
            raise RuntimeError("SWE setup received an invalid sandbox handle.")
        state["sandbox_client"] = client
        state["sandbox_id"] = str(sandbox_id)
        if "answer" in task:
            state.setdefault("answer", task["answer"])
        sandbox_config = task.get("sandbox")
        if isinstance(sandbox_config, Mapping):
            sandbox_data = dict(sandbox_config)
            timeout_minutes = int(sandbox_data.get("timeout_minutes") or 60)
            state.setdefault("test_timeout", timeout_minutes * 60)
        await self.legacy_taskset.setup(state)

    @cleanup(priority=100)
    async def cleanup_swe_state(self, task: Task, state: State) -> None:
        _ = task
        archive = state.pop("r2e_tests_archive_local_path", None)
        if isinstance(archive, str):
            Path(archive).unlink(missing_ok=True)
        state.pop("sandbox_client", None)

    async def validate_instance(self, state: State) -> bool:
        return bool(await self.legacy_taskset.validate_instance(state))


def sandbox_spec_config(spec: SandboxSpec, workdir: str) -> ConfigData:
    config: ConfigData = {
        "image": spec.image,
        "cpu_cores": spec.cpu_cores,
        "memory_gb": spec.memory_gb,
        "disk_size_gb": spec.disk_size_gb,
        "gpu_count": spec.gpu_count,
        "workdir": workdir,
        "scope": "rollout",
    }
    if spec.timeout_minutes is not None:
        config["timeout_minutes"] = spec.timeout_minutes
    return config


def task_id(info: ConfigMap, index: int) -> str:
    for key in ("instance_id", "commit_hash", "id"):
        value = info.get(key)
        if value is not None:
            return str(value)
    return str(index)


def load_taskset(config: SWETasksetConfig) -> SWETaskset:
    return SWETaskset(config=config)
