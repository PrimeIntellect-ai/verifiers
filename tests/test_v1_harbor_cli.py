import importlib
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from uuid import uuid4

import pytest

import verifiers as vf
from harnesses import (
    MiniSWEAgent,
    MiniSWEAgentConfig,
    MiniSWEAgentProgramConfig,
    OpenCode,
    OpenCodeConfig,
    OpenCodeProgramConfig,
    Pi,
    PiConfig,
    PiProgramConfig,
    RLM,
    RLMConfig,
    RLMProgramConfig,
    Terminus2Config,
    Terminus2ProgramConfig,
)
from harnesses.pi import PI_DEFAULT_VERSION
from harnesses.terminus_2 import (
    TERMINUS_2_DEFAULT_API_BASE_URL,
    TERMINUS_2_DEFAULT_VERSION,
    TERMINUS_2_DEFAULT_MODEL_NAME,
    Terminus2,
)
from tasksets import HarborTaskset, HarborTasksetConfig
from verifiers.v1.utils.program_utils import merge_task_program, merge_task_sandbox
from verifiers.v1.utils.sandbox_python_utils import SANDBOX_PYTHON


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
import verifiers as vf
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig


def load_taskset(config: HarborTasksetConfig):
    if config.bundle_package is None:
        config = config.model_copy(update={"bundle_package": __name__})
    return HarborTaskset(config=config)


def load_env():
    return vf.Env(taskset=HarborTaskset(config=HarborTasksetConfig(bundle_package=__name__)), harness=OpenCode(config=OpenCodeConfig()))
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

    taskset = getattr(package, "load_taskset")(config=HarborTasksetConfig())
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
        merge_task_sandbox(
            vf.SandboxConfig(network_access=False, scope="rollout"), task
        ).network_access
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

    taskset = getattr(package, "load_taskset")(config=HarborTasksetConfig())

    with pytest.raises(ValueError, match="Malformed Harbor task"):
        list(taskset)


@pytest.mark.parametrize("section", ["agent", "verifier"])
def test_harbor_task_rejects_non_mapping_agent_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: str
) -> None:
    package = write_harbor_package(tmp_path, monkeypatch)
    task_dir = write_harbor_task(cast(Path, getattr(package, "tasks_root")))
    (task_dir / "task.toml").write_text(
        f"""
version = "1.0"
{section} = "invalid"

[environment]
docker_image = "ubuntu:24.04"
""".strip()
    )
    taskset = getattr(package, "load_taskset")(config=HarborTasksetConfig())

    with pytest.raises(TypeError, match=rf"\[{section}\] must be a mapping"):
        list(taskset)


def test_harbor_taskset_constructs_env_with_opencode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = write_harbor_package(tmp_path, monkeypatch)
    write_harbor_task(cast(Path, getattr(package, "tasks_root")))

    env = getattr(package, "load_env")()

    task = next(iter(env.taskset))
    assert task["task_name"] == "task-a"
    assert isinstance(env.harness, OpenCode)
    assert "task_dir" not in cast(dict[str, object], env.harness.config.program.data())


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
        self.execute_commands: list[tuple[str, int | None, str | None]] = []
        self.background_jobs: list[tuple[str, str, int | None, str | None]] = []
        type(self).instances.append(self)

    async def upload_file(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs

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
    fake_module = cast(Any, types.ModuleType("prime_sandboxes"))
    fake_module.AsyncSandboxClient = FakeHarborSandboxClient
    monkeypatch.setitem(sys.modules, "prime_sandboxes", fake_module)
    FakeHarborSandboxClient.instances = []

    taskset = HarborTaskset(config=HarborTasksetConfig(bundle_package=__name__))
    reward = await taskset.harbor_reward(
        vf.Task(
            {"prompt": [], "harbor": {"task_dir": str(task_dir), "test_timeout": 120}}
        ).freeze(),
        vf.State({"sandbox_id": "sbx-1"}),
    )

    client = FakeHarborSandboxClient.instances[0]
    assert reward == 1.0
    assert client.background_jobs == [("sbx-1", "bash test.sh", 120, "/tests")]
    assert ("bash test.sh", 120, "/tests") not in client.execute_commands


def test_packaged_harbor_and_opencode_imports_are_available_from_packages() -> None:
    assert OpenCode
    assert OpenCodeConfig
    assert Pi
    assert Terminus2
    assert HarborTaskset


def test_opencode_config_owns_opencode_harness_fields() -> None:
    harness = OpenCode(
        config=OpenCodeConfig(
            system_prompt=None,
            program=OpenCodeProgramConfig(
                agent_workdir="/workspace",
                disabled_tools=["webfetch"],
            ),
            max_turns=2,
        )
    )
    program = cast(dict[str, object], harness.program_config.data())
    command = cast(list[object], program["command"])
    mcp_setup = cast(dict[str, object], program["channels"])["mcp"]
    setup = cast(str, program["setup"])

    assert harness.config.program.agent_workdir == "/workspace"
    assert harness.config.program.disabled_tools == ["webfetch"]
    assert harness.config.system_prompt is None
    assert harness.config.max_turns == 2
    assert "apt-get -o Acquire::Retries=3 update" in setup
    assert "apt-get -o Acquire::Retries=3 install" in setup
    assert "OPENCODE_RELEASE_REPO=PrimeIntellect-ai/opencode" in setup
    assert "OPENCODE_RELEASE_PATH=releases/download/v1.1.63-rl2" in setup
    assert "/workspace" in cast(str, command[2])
    assert '"webfetch": false' in cast(str, mcp_setup)
    assert "/opencode/system.txt" in cast(dict[str, object], program["files"])


@pytest.mark.parametrize(
    "version",
    ["PrimeIntellect-ai/opencode@latest", "  PrimeIntellect-ai/opencode  "],
)
def test_opencode_latest_version_uses_latest_download_url(
    version: str,
) -> None:
    harness = OpenCode(
        config=OpenCodeConfig(
            version=version,
            program=OpenCodeProgramConfig(
                install_ripgrep=False,
            ),
        )
    )
    program = cast(dict[str, object], harness.program_config.data())
    setup = cast(str, program["setup"])

    assert "OPENCODE_RELEASE_REPO=PrimeIntellect-ai/opencode" in setup
    assert "OPENCODE_RELEASE_PATH=releases/latest/download" in setup


def test_opencode_custom_version_uses_versioned_release() -> None:
    harness = OpenCode(
        config=OpenCodeConfig(
            version="Example/open-code@v2.0.0",
        )
    )
    program = cast(dict[str, object], harness.program_config.data())
    setup = cast(str, program["setup"])

    assert "OPENCODE_RELEASE_REPO=Example/open-code" in setup
    assert "OPENCODE_RELEASE_PATH=releases/download/v2.0.0" in setup


@pytest.mark.parametrize(
    ("harness_cls", "config_cls", "program_cls"),
    [
        (OpenCode, OpenCodeConfig, OpenCodeProgramConfig),
        (MiniSWEAgent, MiniSWEAgentConfig, MiniSWEAgentProgramConfig),
        (Pi, PiConfig, PiProgramConfig),
        (RLM, RLMConfig, RLMProgramConfig),
        (Terminus2, Terminus2Config, Terminus2ProgramConfig),
    ],
)
def test_packaged_command_harnesses_defer_partial_program_overrides(
    harness_cls, config_cls, program_cls
) -> None:
    override = {
        "setup": "echo caller",
        "env": {"CALLER": "1"},
        "args": ["--caller"],
    }
    harness = harness_cls(config=config_cls(program=override))
    program = cast(dict[str, object], harness.program_config.data())
    env = cast(dict[str, object], program["env"])
    setup = cast(list[object], program["setup"])
    args = cast(list[object], program["args"])

    assert program["command"]
    assert env["CALLER"] == "1"
    assert setup[-1] == "echo caller"
    assert args[-1] == "--caller"
    assert isinstance(harness.config.program, program_cls)
    assert isinstance(harness.program_config, vf.ProgramConfig)
    config_args = cast(list[object], harness.config.program.args)
    assert harness.program_config.command == program["command"]
    assert harness.config.program.env["CALLER"] == "1"
    assert harness.config.program.setup == override["setup"]
    assert config_args[-1] == "--caller"


def test_packaged_command_harness_config_program_patch_precedence() -> None:
    harness = MiniSWEAgent(
        config=MiniSWEAgentConfig(
            program=MiniSWEAgentProgramConfig(env={"OPENAI_MODEL": "caller-model"})
        )
    )
    program = cast(dict[str, object], harness.program_config.data())
    env = cast(dict[str, object], program["env"])

    assert env["OPENAI_MODEL"] == "caller-model"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("command", ["other"]),
        ("channels", "mcp"),
    ],
)
def test_packaged_command_harness_config_program_rejects_owned_keys(
    key: str, value: object
) -> None:
    with pytest.raises(ValueError, match="Command ProgramConfig can only"):
        OpenCode(config=OpenCodeConfig.model_validate({"program": {key: value}}))


def test_pi_harness_writes_intercepted_model_and_mcp_config() -> None:
    harness = Pi()
    program = cast(dict[str, object], harness.program_config.data())
    setup = cast(str, program["setup"])
    channels = cast(dict[str, object], program["channels"])
    mcp_setup = cast(str, channels["mcp"])

    assert "apt-get -o Acquire::Retries=3 update" in setup
    assert "apt-get -o Acquire::Retries=3 install" in setup
    assert harness.config.version == PI_DEFAULT_VERSION
    assert PI_DEFAULT_VERSION == "@earendil-works/pi-coding-agent@latest"
    assert f"npm install -g --ignore-scripts {PI_DEFAULT_VERSION}" in setup
    assert "mariozechner" not in setup
    assert '"baseUrl": "${OPENAI_BASE_URL}"' in mcp_setup
    assert '"api": "openai-completions"' in mcp_setup
    assert '"apiKey": "${OPENAI_API_KEY:-intercepted}"' in mcp_setup
    assert '"id": "model"' in mcp_setup
    assert '"name": "${OPENAI_MODEL}"' in mcp_setup
    assert f'"command": "{SANDBOX_PYTHON}"' in mcp_setup


def test_pi_harness_preserves_scoped_npm_versions() -> None:
    harness = Pi(config=PiConfig(version="@anthropic-ai/claude-code@1.2.3"))
    program = cast(dict[str, object], harness.program_config.data())
    setup = cast(str, program["setup"])

    assert "npm install -g --ignore-scripts @anthropic-ai/claude-code@1.2.3" in setup


def test_terminus_2_harness_builds_sandbox_program() -> None:
    harness = Terminus2(
        config=Terminus2Config(
            system_prompt="extra system prompt",
            program=Terminus2ProgramConfig(
                agent_workdir="/workspace",
                max_turns=7,
                python_version="3.12",
            ),
        )
    )
    program = cast(dict[str, object], harness.config.program.data())
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
    assert f"--with {TERMINUS_2_DEFAULT_VERSION}" in run_script
    assert "git+https://github.com" not in run_script
    assert "max_turns=7" in run_script

    script = run_script.split("python - <<'PY' 2>&1 | tee -a", 1)[1]
    script = script.split("\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(script, "terminus_2_agent.py", "exec")
    assert TERMINUS_2_DEFAULT_MODEL_NAME in script
    assert TERMINUS_2_DEFAULT_API_BASE_URL in script
    assert "OPENAI_MODEL" not in script
    assert "PRIME_API_KEY" not in script
    assert "async def prepare_logs_for_host(self) -> None" in script
    assert "max_turns=7" in script


def test_task_program_merges_into_command_program_without_collisions() -> None:
    harness = vf.Harness(
        config=vf.HarnessConfig(
            program=vf.ProgramConfig(
                command=["tool"],
                sandbox=True,
                files={"/harness.txt": "harness"},
                setup="echo harness",
                channels={"mcp": "echo harness tools"},
                env={"HARNESS": "1"},
                artifacts=vf.ArtifactsConfig.model_validate(
                    {"log": {"path": "/logs/harness.log", "format": "text"}}
                ),
                args=["--base"],
            ),
            sandbox=vf.SandboxConfig(image="python:3.11-slim"),
        )
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
        cast(vf.ConfigData, harness.config.program.data()), task, kind="command"
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


def test_command_program_patch_preserves_explicit_default_values() -> None:
    program = vf.ProgramConfig(setup_timeout=300).resolve_command(
        command=["tool"],
        setup_timeout=600,
    )

    assert program.data()["setup_timeout"] == 300


def test_task_program_rejects_harness_owned_keys() -> None:
    harness = vf.Harness(
        config=vf.HarnessConfig(
            program=vf.ProgramConfig(command=["tool"], sandbox=True),
            sandbox=vf.SandboxConfig(image="python:3.11-slim"),
        )
    )
    task = vf.Task({"prompt": [], "program": {"command": ["other"]}}).freeze()

    with pytest.raises(ValueError, match="task.program can only define"):
        merge_task_program(
            cast(vf.ConfigData, harness.config.program.data()),
            task,
            kind="command",
        )


def test_task_program_rejects_colliding_upload_paths() -> None:
    harness = vf.Harness(
        config=vf.HarnessConfig(
            program=vf.ProgramConfig(
                command=["tool"],
                sandbox=True,
                files={"/task/instruction.md": "harness"},
            ),
            sandbox=vf.SandboxConfig(image="python:3.11-slim"),
        )
    )
    task = vf.Task(
        {"prompt": [], "program": {"files": {"/task/instruction.md": "task"}}}
    ).freeze()

    with pytest.raises(ValueError, match="define the same keys"):
        merge_task_program(
            cast(vf.ConfigData, harness.config.program.data()),
            task,
            kind="command",
        )
