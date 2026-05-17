import importlib
import json
import sys
import tarfile
import types
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

import verifiers as root_vf
import verifiers.v1 as vf
from verifiers.v1.packages.harnesses.pi import pi_mcp_json, pi_models_json
from verifiers.v1.packages.harnesses.terminus_2 import (
    DEFAULT_API_BASE_URL,
    DEFAULT_HARBOR_PACKAGE,
    DEFAULT_MODEL_NAME,
    Terminus2,
    terminus_2_agent_script,
)
from verifiers.v1.packages.tasksets.harbor import harbor_reward
from verifiers.v1.utils.program_utils import merge_task_program, merge_task_sandbox


def write_harbor_task(root: Path, name: str = "task-a") -> Path:
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "instruction.md").write_text("Write hello to /app/hello.txt\n")
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[environment]
docker_image = "ubuntu:24.04"
cpus = 1
memory = "2G"
storage = "8G"

[agent]
timeout_sec = 600

[verifier]
timeout_sec = 300
""".strip()
    )
    (task_dir / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt")
    (task_dir / "solution" / "solve.sh").write_text("echo hello > /app/hello.txt")
    return task_dir


def write_harbor_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    package_name = f"harbor_pkg_{uuid4().hex}"
    package_dir = tmp_path / package_name
    tasks_root = package_dir / "tasks"
    tasks_root.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        """
import verifiers.v1 as vf


def load_taskset(**kwargs):
    return vf.HarborTaskset(**kwargs)


def load_env():
    return vf.Env(taskset=vf.HarborTaskset(), harness=vf.OpenCode())
""".lstrip()
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    module = importlib.import_module(package_name)
    setattr(module, "tasks_root", tasks_root)
    return module


def test_harbor_taskset_loads_package_tasks_with_program_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = write_harbor_package(tmp_path, monkeypatch)
    write_harbor_task(cast(Path, getattr(package, "tasks_root")))

    taskset = getattr(package, "load_taskset")()
    task = next(iter(taskset))

    assert task["taskset_id"] == "harbor"
    assert task["task_name"] == "task-a"
    assert task["prompt"] == [
        {"role": "user", "content": "Write hello to /app/hello.txt"}
    ]
    assert task["sandbox"]["image"] == "ubuntu:24.04"
    assert task["sandbox"]["memory_gb"] == 2.0
    assert task["sandbox"]["disk_size_gb"] == 8.0
    assert task["sandbox"]["command_timeout"] == 600
    assert "network_access" not in task["sandbox"]
    assert (
        merge_task_sandbox({"network_access": False, "scope": "rollout"}, task)[
            "network_access"
        ]
        is False
    )
    assert task["harbor"]["test_timeout"] == 300.0
    assert task["program"]["files"] == {
        "/task/instruction.md": {"task": "instruction"},
        "/task/task.toml": {"task": "task_toml"},
    }
    assert task["program"]["env"]["HARBOR_TASK_NAME"] == "task-a"
    assert task["program"]["env"]["AGENT_WORKDIR"] == "/app"


def test_harbor_taskset_rejects_malformed_package_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = write_harbor_package(tmp_path, monkeypatch)
    bad_task = cast(Path, getattr(package, "tasks_root")) / "bad-task"
    bad_task.mkdir()
    (bad_task / "task.toml").write_text('version = "1.0"')

    taskset = getattr(package, "load_taskset")()

    with pytest.raises(ValueError, match="Malformed Harbor task"):
        list(taskset)


def test_harbor_taskset_constructs_env_with_opencode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = write_harbor_package(tmp_path, monkeypatch)
    write_harbor_task(cast(Path, getattr(package, "tasks_root")))

    env = getattr(package, "load_env")()

    row = env.get_dataset()[0]
    task = env.taskset.to_task(row)
    assert task["task_name"] == "task-a"
    assert isinstance(env.harness, vf.OpenCode)
    assert "task_dir" not in cast(dict[str, object], env.harness.program)


def test_harbor_taskset_resolves_verifier_environment_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = write_harbor_package(tmp_path, monkeypatch)
    tasks_root = cast(Path, getattr(package, "tasks_root"))
    write_harbor_task(tasks_root, "shared-default")
    implicit = write_harbor_task(tasks_root, "separate-implicit")
    explicit = write_harbor_task(tasks_root, "separate-reuse-env")
    implicit.joinpath("task.toml").write_text(
        """
schema_version = "1.2"
artifacts = ["/logs/agent/trajectory.json"]

[environment]
docker_image = "agent:latest"
cpus = 1
memory_mb = 2048
storage_mb = 4096

[verifier]
timeout_sec = 30

[verifier.environment]
docker_image = "verifier:latest"
cpus = 2
memory_mb = 1024
storage_mb = 2048
allow_internet = false
""".strip()
    )
    explicit.joinpath("task.toml").write_text(
        """
schema_version = "1.2"

[environment]
docker_image = "agent-reused:latest"
cpus = 3
memory_mb = 3072
storage_mb = 6144

[verifier]
environment_mode = "separate"
timeout_sec = 45
""".strip()
    )

    rows = {task["task_name"]: task for task in getattr(package, "load_taskset")()}

    assert rows["shared-default"]["harbor"]["verifier_mode"] == "shared"
    assert rows["shared-default"]["harbor"]["verifier_sandbox"] is None
    assert rows["separate-implicit"]["harbor"]["verifier_mode"] == "separate"
    assert rows["separate-implicit"]["harbor"]["verifier_sandbox"] == {
        "image": "verifier:latest",
        "cpu_cores": 2.0,
        "memory_gb": 1.0,
        "disk_size_gb": 2.0,
        "timeout_minutes": 120,
        "command_timeout": 30,
        "workdir": "/app",
        "scope": "rollout",
        "network_access": False,
    }
    assert rows["separate-implicit"]["harbor"]["artifacts"] == [
        "/logs/agent/trajectory.json"
    ]
    assert rows["separate-reuse-env"]["harbor"]["verifier_mode"] == "separate"
    assert rows["separate-reuse-env"]["harbor"]["verifier_sandbox"]["image"] == (
        "agent-reused:latest"
    )
    assert rows["separate-reuse-env"]["harbor"]["verifier_sandbox"]["memory_gb"] == 3.0


def test_harbor_taskset_rejects_shared_mode_with_verifier_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = write_harbor_package(tmp_path, monkeypatch)
    task_dir = write_harbor_task(
        cast(Path, getattr(package, "tasks_root")),
        "bad-shared-env",
    )
    task_dir.joinpath("task.toml").write_text(
        """
[environment]
docker_image = "agent:latest"

[verifier]
environment_mode = "shared"

[verifier.environment]
docker_image = "verifier:latest"
""".strip()
    )

    with pytest.raises(ValueError, match="incompatible"):
        list(getattr(package, "load_taskset")())


class FakeHarborCommandResult:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        stdout: str = "",
        stderr: str = "",
    ):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class FakeHarborSandboxClient:
    instances: list["FakeHarborSandboxClient"] = []

    def __init__(self):
        self.create_requests: list[object] = []
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.upload_files: list[tuple[str, str, str]] = []
        self.download_files: list[tuple[str, str, str]] = []
        self.execute_commands: list[tuple[str, int | None, str | None]] = []
        self.background_jobs: list[tuple[str, str, int | None, str | None]] = []
        type(self).instances.append(self)

    async def create(self, request: object) -> object:
        self.create_requests.append(request)
        sandbox_id = f"verifier-sbx-{len(type(self).instances)}"
        self.created.append(sandbox_id)
        return SimpleNamespace(id=sandbox_id)

    async def wait_for_creation(self, sandbox_id: str) -> None:
        assert sandbox_id

    async def delete(self, sandbox_id: str) -> None:
        self.deleted.append(sandbox_id)

    async def upload_file(self, *args: object, **kwargs: object) -> None:
        sandbox_id = str(kwargs.get("sandbox_id") or args[0])
        file_path = str(kwargs.get("file_path") or args[1])
        local_file_path = str(kwargs.get("local_file_path") or args[2])
        self.upload_files.append((sandbox_id, file_path, local_file_path))

    async def upload_bytes(self, *args: object, **kwargs: object) -> None:
        sandbox_id = str(kwargs.get("sandbox_id") or args[0])
        file_path = str(kwargs.get("file_path") or args[1])
        self.upload_files.append((sandbox_id, file_path, "<bytes>"))

    async def download_file(self, *args: object, **kwargs: object) -> None:
        sandbox_id = str(kwargs.get("sandbox_id") or args[0])
        file_path = str(kwargs.get("file_path") or args[1])
        local_file_path = str(kwargs.get("local_file_path") or args[2])
        self.download_files.append((sandbox_id, file_path, local_file_path))
        local_path = Path(local_file_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.endswith(".tar.gz"):
            with tarfile.open(local_path, "w:gz") as tar:
                marker = local_path.parent / "marker.txt"
                marker.write_text(file_path)
                tar.add(marker, arcname="marker.txt")
            return
        local_path.write_text(file_path)

    async def read_file(self, *args: object, **kwargs: object) -> str:
        _ = args, kwargs
        return ""

    async def execute_command(
        self, *args: object, **kwargs: object
    ) -> FakeHarborCommandResult:
        command = str(kwargs.get("command") or args[1])
        timeout = cast(int | None, kwargs.get("timeout"))
        working_dir = cast(str | None, kwargs.get("working_dir"))
        self.execute_commands.append((command, timeout, working_dir))
        if "reward.txt" in command:
            return FakeHarborCommandResult(stdout="1\n")
        return FakeHarborCommandResult()

    async def run_background_job(
        self, *args: object, **kwargs: object
    ) -> FakeHarborCommandResult:
        sandbox_id = str(kwargs.get("sandbox_id") or args[0])
        command = str(kwargs.get("command") or args[1])
        timeout = cast(int | None, kwargs.get("timeout"))
        working_dir = cast(str | None, kwargs.get("working_dir"))
        self.background_jobs.append((sandbox_id, command, timeout, working_dir))
        return FakeHarborCommandResult(stdout="tests passed")

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_harbor_reward_uses_background_job_for_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir = write_harbor_task(tmp_path)
    fake_module = types.ModuleType("prime_sandboxes")
    fake_module.AsyncSandboxClient = FakeHarborSandboxClient
    monkeypatch.setitem(sys.modules, "prime_sandboxes", fake_module)
    FakeHarborSandboxClient.instances = []

    reward = await harbor_reward(
        {"harbor": {"task_dir": str(task_dir), "test_timeout": 120}},
        {"sandbox_id": "sbx-1"},
    )

    client = FakeHarborSandboxClient.instances[0]
    assert reward == 1.0
    assert client.created == []
    assert client.background_jobs == [("sbx-1", "bash test.sh", 120, "/tests")]
    assert ("bash test.sh", 120, "/tests") not in client.execute_commands


@pytest.mark.asyncio
async def test_harbor_reward_uses_fresh_separate_verifier_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir = write_harbor_task(tmp_path)
    fake_module = types.ModuleType("prime_sandboxes")
    fake_module.AsyncSandboxClient = FakeHarborSandboxClient

    class FakeCreateSandboxRequest:
        def __init__(self, **kwargs: object):
            self.__dict__.update(kwargs)

    fake_module.CreateSandboxRequest = FakeCreateSandboxRequest
    monkeypatch.setitem(sys.modules, "prime_sandboxes", fake_module)
    FakeHarborSandboxClient.instances = []

    reward = await harbor_reward(
        {
            "harbor": {
                "task_dir": str(task_dir),
                "test_timeout": 120,
                "verifier_mode": "separate",
                "verifier_sandbox": {
                    "image": "verifier:latest",
                    "scope": "rollout",
                },
                "verifier_env": {"MODEL": "judge"},
                "artifacts": [
                    "/logs/agent/trajectory.json",
                    "/a/data",
                    "/b/data",
                ],
            }
        },
        {"sandbox_id": "agent-sbx"},
    )

    agent_client, verifier_client = FakeHarborSandboxClient.instances
    assert reward == 1.0
    assert agent_client.created == []
    assert verifier_client.create_requests[0].docker_image == "verifier:latest"
    assert verifier_client.deleted == ["verifier-sbx-2"]
    assert verifier_client.background_jobs == [
        ("verifier-sbx-2", "bash /tests/test.sh", 120, None)
    ]
    assert not any(
        path == "/tmp/harbor_tests.tar.gz" for _, path, _ in agent_client.upload_files
    )
    transfer_command = agent_client.execute_commands[0][0]
    assert "/logs/artifacts" in transfer_command
    assert "/logs/agent/trajectory.json" in transfer_command
    assert "/a/data" in transfer_command
    assert "/b/data" in transfer_command
    assert agent_client.download_files[0][1] == "/tmp/_vf_harbor_inputs.tar.gz"
    assert verifier_client.upload_files[0][1] == "/tmp/_vf_harbor_inputs.tar.gz"
    assert any(
        "tar -xzf" in command for command, _, _ in verifier_client.execute_commands
    )
    assert verifier_client.background_jobs[0][1] != "bash test.sh"


def test_packaged_harbor_and_opencode_imports_are_reexported() -> None:
    from verifiers.v1.packages.harnesses import OpenCode, OpenCodeConfig, Pi
    from verifiers.v1.packages.tasksets import HarborTaskset

    assert vf.OpenCode is OpenCode
    assert vf.OpenCodeConfig is OpenCodeConfig
    assert vf.Pi is Pi
    assert vf.Terminus2 is Terminus2
    assert root_vf.Terminus2 is Terminus2
    assert vf.HarborTaskset is HarborTaskset


def test_opencode_config_owns_opencode_harness_fields() -> None:
    harness = vf.OpenCode(
        config=vf.OpenCodeConfig(
            agent_workdir="/workspace",
            disabled_tools=["webfetch"],
            system_prompt=None,
            max_turns=2,
        )
    )
    program = cast(dict[str, object], harness.program)
    command = cast(list[object], program["command"])
    mcp_setup = cast(dict[str, object], program["channels"])["mcp"]
    setup = cast(str, program["setup"])

    assert harness.config.agent_workdir == "/workspace"
    assert harness.config.disabled_tools == ["webfetch"]
    assert harness.config.system_prompt is None
    assert harness.config.max_turns == 2
    assert "apt-get -o Acquire::Retries=3 update" in setup
    assert "apt-get -o Acquire::Retries=3 install" in setup
    assert "/workspace" in cast(str, command[2])
    assert '"webfetch": false' in cast(str, mcp_setup)
    assert "/opencode/system.txt" not in cast(dict[str, object], program["files"])


def test_pi_harness_writes_intercepted_model_and_mcp_config() -> None:
    harness = vf.Pi()
    program = cast(dict[str, object], harness.program)
    setup = cast(str, program["setup"])
    models = json.loads(
        pi_models_json(
            {
                "base_url": "http://127.0.0.1:1/rollout/key/v1",
                "api_key": "secret",
                "api_client_type": "openai_chat_completions",
                "model": "openai/gpt-5.4-mini",
            }
        )
    )
    mcp = json.loads(pi_mcp_json())

    assert "apt-get -o Acquire::Retries=3 update" in setup
    assert "apt-get -o Acquire::Retries=3 install" in setup
    provider = models["providers"]["verifiers"]
    assert provider["baseUrl"] == "http://127.0.0.1:1/rollout/key/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "secret"
    assert provider["models"] == [{"id": "model", "name": "openai/gpt-5.4-mini"}]
    assert mcp["mcpServers"]["verifiers-tools"]["command"] == "python3"


def test_terminus_2_harness_builds_sandbox_program() -> None:
    harness = vf.Terminus2(
        system_prompt="extra system prompt",
        agent_workdir="/workspace",
        max_turns=7,
        python_version="3.12",
    )
    program = cast(dict[str, object], harness.program)
    command = cast(list[object], program["command"])
    setup = cast(str, program["setup"])
    files = cast(dict[str, object], program["files"])
    artifacts = cast(dict[str, object], program["artifacts"])
    env = cast(dict[str, object], program.get("env", {}))

    assert isinstance(harness, vf.Harness)
    assert "/terminus_2/instruction.md" in files
    assert "/terminus_2/system_prompt.txt" in files
    assert "apt-get -o Acquire::Retries=3 update" in setup
    assert "apt-get -o Acquire::Retries=3 install" in setup
    assert "git" not in setup
    assert "terminus_2_log" in artifacts
    assert "OPENAI_MODEL" not in env

    run_script = cast(str, command[2])
    assert "TERMINUS_2_WORKDIR=/workspace" in run_script
    assert f"--with {DEFAULT_HARBOR_PACKAGE}" in run_script
    assert "git+https://github.com" not in run_script
    assert "max_turns=7" in run_script

    script = terminus_2_agent_script(max_turns=7)
    compile(script, "terminus_2_agent.py", "exec")
    assert DEFAULT_MODEL_NAME in script
    assert DEFAULT_API_BASE_URL in script
    assert "OPENAI_MODEL" not in script
    assert "PRIME_API_KEY" in script
    assert "async def prepare_logs_for_host(self) -> None" in script
    assert "max_turns=7" in script


def test_task_program_merges_into_command_program_without_collisions() -> None:
    harness = vf.Harness(
        program={
            "command": ["tool"],
            "sandbox": True,
            "files": {"/harness.txt": "harness"},
            "setup": "echo harness",
            "channels": {"mcp": "echo harness tools"},
            "env": {"HARNESS": "1"},
            "artifacts": {"log": {"path": "/logs/harness.log", "format": "text"}},
            "args": ["--base"],
        },
        sandbox={"image": "python:3.11-slim"},
    )
    task = vf.Task(
        {
            "prompt": [],
            "program": {
                "files": {"/task/instruction.md": "task"},
                "setup": "echo task",
                "env": {"TASK": "1"},
                "artifacts": {"task_log": {"path": "/logs/task.log", "format": "text"}},
                "args": ["--task"],
            },
        }
    ).freeze()

    program = merge_task_program(
        cast(dict[str, object], harness.program), task, kind="command"
    )

    assert program["files"] == {
        "/harness.txt": "harness",
        "/task/instruction.md": "task",
    }
    assert program["setup"] == ["echo harness", "echo task"]
    assert program["channels"] == {"mcp": "echo harness tools"}
    assert program["env"] == {"HARNESS": "1", "TASK": "1"}
    assert program["args"] == ["--base", "--task"]
    assert program["artifacts"] == {
        "log": {"path": "/logs/harness.log", "format": "text"},
        "task_log": {"path": "/logs/task.log", "format": "text"},
    }


def test_task_program_rejects_harness_owned_keys() -> None:
    harness = vf.Harness(
        program={"command": ["tool"], "sandbox": True},
        sandbox={"image": "python:3.11-slim"},
    )
    task = vf.Task({"prompt": [], "program": {"command": ["other"]}}).freeze()

    with pytest.raises(ValueError, match="task.program can only define"):
        merge_task_program(
            cast(dict[str, object], harness.program), task, kind="command"
        )


def test_task_program_rejects_colliding_upload_paths() -> None:
    harness = vf.Harness(
        program={
            "command": ["tool"],
            "sandbox": True,
            "files": {"/task/instruction.md": "harness"},
        },
        sandbox={"image": "python:3.11-slim"},
    )
    task = vf.Task(
        {"prompt": [], "program": {"files": {"/task/instruction.md": "task"}}}
    ).freeze()

    with pytest.raises(ValueError, match="define the same keys"):
        merge_task_program(
            cast(dict[str, object], harness.program), task, kind="command"
        )
