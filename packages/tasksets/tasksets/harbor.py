from pathlib import Path
from typing import cast

import verifiers as vf
from verifiers.utils.import_utils import load_toml
from verifiers.v1.utils.sandbox_utils import SandboxClient

from tasksets.utils.harbor_utils import (
    TASKS_SUBDIR,
    bundle_tasks_root,
    download_harbor_dataset,
    harbor_sandbox,
    harbor_task_dirs,
    parse_gb,
    parse_number,
    parse_reward_text,
    upload_harbor_tests,
)

HARBOR_DEFAULT_SANDBOX = vf.SandboxConfig(
    image="python:3.11-slim",
    cpu_cores=2.0,
    memory_gb=4.0,
    disk_size_gb=10.0,
    timeout_minutes=120,
    workdir="/app",
    command_timeout=900,
)


class HarborTasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "harbor"
    dataset: str | None = None
    bundle_package: str | None = None
    task_names: list[str] | None = None
    cache_dir: str | None = None
    refresh: bool = False
    sandbox: vf.SandboxConfig = HARBOR_DEFAULT_SANDBOX
    verifier_timeout_seconds: float = 900.0
    task_dir: str = "/task"
    env: dict[str, str] = {}


class HarborTaskset(vf.Taskset[HarborTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        config = self.config
        if config.dataset is not None:
            cache_dir_path = (
                Path(str(config.cache_dir)).expanduser() if config.cache_dir else None
            )
            root = download_harbor_dataset(
                config.dataset,
                cache_dir=cache_dir_path,
                refresh=config.refresh,
            )
        else:
            bundle_package = config.bundle_package
            if bundle_package is None:
                raise RuntimeError(
                    "HarborTaskset() without a dataset requires bundle_package. "
                    "Pass dataset='...' to fetch from Harbor Hub, or set "
                    "bundle_package=__name__ from the package that owns tasks/."
                )
            root = bundle_tasks_root(bundle_package)
            if not root.exists():
                raise FileNotFoundError(
                    "HarborTaskset() without a dataset requires "
                    f"{bundle_package}/{TASKS_SUBDIR}/ to contain Harbor task "
                    f"directories. Not found: {root}"
                )
        task_dirs = harbor_task_dirs(root, list(config.task_names or []))
        tasks: list[vf.JsonData] = []
        for task_dir in task_dirs:
            task_toml_path = task_dir / "task.toml"
            instruction_path = task_dir / "instruction.md"
            with task_toml_path.open("rb") as f:
                task_config = load_toml(f)
            environment = task_config.get("environment", {}) or {}
            assert isinstance(environment, dict)
            agent_config = task_config.get("agent", {}) or {}
            verifier_config = task_config.get("verifier", {}) or {}
            if not isinstance(agent_config, dict):
                raise TypeError(f"{task_toml_path} [agent] must be a mapping.")
            if not isinstance(verifier_config, dict):
                raise TypeError(f"{task_toml_path} [verifier] must be a mapping.")
            instruction = instruction_path.read_text().strip()
            task_remote_dir = config.task_dir.rstrip("/") or "/task"
            sandbox = harbor_sandbox(HARBOR_DEFAULT_SANDBOX, config.sandbox)
            sandbox = sandbox.model_copy(
                update={
                    "image": environment.get("docker_image") or sandbox.image,
                    "cpu_cores": parse_number(
                        environment.get("cpus"), sandbox.cpu_cores
                    ),
                    "memory_gb": parse_gb(environment.get("memory"), sandbox.memory_gb),
                    "disk_size_gb": parse_gb(
                        environment.get("storage"), sandbox.disk_size_gb
                    ),
                    "command_timeout": int(
                        parse_number(
                            agent_config.get("timeout_sec"),
                            sandbox.command_timeout or 900,
                        )
                    ),
                    **(
                        {"network_access": bool(environment["allow_internet"])}
                        if "allow_internet" in environment
                        else {}
                    ),
                }
            )
            sandbox_data = sandbox.data(fill_defaults=False)
            workdir = sandbox.workdir or "/app"
            tasks.append(
                {
                    "task_name": task_dir.name,
                    "instruction": instruction,
                    "task_toml": task_toml_path.read_text(),
                    "task_dir": str(task_dir),
                    "prompt": [{"role": "user", "content": instruction}],
                    "sandbox": sandbox_data,
                    "program": {
                        "files": {
                            f"{task_remote_dir}/instruction.md": {
                                "task": "instruction"
                            },
                            f"{task_remote_dir}/task.toml": {"task": "task_toml"},
                        },
                        "env": {
                            "HARBOR_TASK_NAME": task_dir.name,
                            "HARBOR_TASK_DIR": task_remote_dir,
                            "HARBOR_INSTRUCTION_PATH": f"{task_remote_dir}/instruction.md",
                            "AGENT_WORKDIR": workdir,
                            **config.env,
                        },
                    },
                    "harbor": {
                        "task_dir": str(task_dir),
                        "task_name": task_dir.name,
                        "config": task_config,
                        "docker_image": environment.get("docker_image"),
                        "test_timeout": parse_number(
                            verifier_config.get("timeout_sec"),
                            config.verifier_timeout_seconds,
                        ),
                    },
                    "info": {
                        "harbor": {
                            "task_name": task_dir.name,
                            "docker_image": environment.get("docker_image"),
                        }
                    },
                }
            )
        assert tasks, f"No valid Harbor tasks found in {root}."
        return tasks

    @vf.reward(weight=1.0)
    async def harbor_reward(self, task: vf.Task, state: vf.State) -> float:
        if state.get("error") is not None:
            return 0.0
        sandbox_id = state["sandbox_id"]
        assert isinstance(sandbox_id, str)
        harbor = task["harbor"]
        assert isinstance(harbor, dict)
        task_dir = Path(str(harbor["task_dir"]))
        from prime_sandboxes import AsyncSandboxClient

        client = cast(SandboxClient, AsyncSandboxClient())
        try:
            await upload_harbor_tests(client, sandbox_id, task_dir)
            test_timeout = int(parse_number(harbor.get("test_timeout"), 900))
            result = await client.run_background_job(
                sandbox_id=sandbox_id,
                command="bash test.sh",
                working_dir="/tests",
                timeout=test_timeout,
            )
            state["harbor_tests"] = {
                "returncode": result.exit_code,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            }
            reward_result = await client.execute_command(
                sandbox_id=sandbox_id,
                command=(
                    "if [ -s /logs/verifier/reward.txt ]; then "
                    "cat /logs/verifier/reward.txt; "
                    "elif [ -s /logs/verifier/reward.json ]; then "
                    "cat /logs/verifier/reward.json; fi"
                ),
            )
        except Exception as e:
            state["harbor_error"] = str(e)
            return 0.0
        finally:
            await client.aclose()
        return parse_reward_text(str(reward_result.stdout or "").strip())


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)
