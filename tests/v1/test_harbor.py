import json
from pathlib import Path

import pytest

from tasksets.harbor_v1 import (
    HarborConfig,
    HarborTaskset,
    parse_task,
)
from verifiers.v1.runtimes import DockerConfig, ProgramResult, Runtime
from verifiers.v1.trace import Trace


def write_task(tmp_path: Path, verifier: str = "") -> Path:
    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Create the requested artifacts.")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (task_dir / "task.toml").write_text(
        """
artifacts = [
  "/tmp/answer.json",
  { source = "/data/results", destination = "saved/results", exclude = ["*.pt"] },
]

[task]
name = "tests/harbor-separate"

[environment]
docker_image = "agent:latest"
cpus = 1
memory_mb = 2048

[verifier]
timeout_sec = 30
"""
        + verifier
    )
    return task_dir


def test_parse_task_resolves_separate_verifier(tmp_path: Path) -> None:
    task = parse_task(
        write_task(
            tmp_path,
            """
environment_mode = "separate"
env = { TOKEN = "${HOST_TOKEN:-fallback}" }

[verifier.environment]
docker_image = "verifier:latest"
network_mode = "no-network"
cpus = 2
memory_mb = 1024
storage_mb = 4096
""",
        ),
        0,
        False,
    )

    assert task.image == "agent:latest"
    assert task.verifier.image == "verifier:latest"
    assert task.verifier.network_access is False
    assert task.verifier.resources.cpu == 2
    assert task.verifier.resources.memory == 1
    assert task.verifier.resources.disk == 4
    assert task.verifier.env == {"TOKEN": "${HOST_TOKEN:-fallback}"}
    assert task.verifier.artifacts == [
        "/tmp/answer.json",
        "/data/results",
    ]
    assert task.verifier.upload_tests is False


def test_parse_task_reuses_agent_environment_for_separate_verifier(
    tmp_path: Path,
) -> None:
    task = parse_task(write_task(tmp_path, 'environment_mode = "separate"\n'), 0, False)

    assert task.verifier.image == "agent:latest"
    assert task.verifier.resources.memory == 2
    assert task.verifier.upload_tests is True


class FakeRuntime(Runtime):
    def __init__(
        self,
        name: str,
        reward: str = "0.25",
        config: DockerConfig | None = None,
    ) -> None:
        super().__init__(name)
        self.config = config or DockerConfig()
        self.reward = reward
        self.read_error: Exception | None = None
        self.started = False
        self.stopped = False
        self.commands: list[tuple[list[str], dict[str, str]]] = []
        self.writes: list[tuple[str, bytes]] = []

    async def start(self) -> None:
        self.started = True

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append((argv, env))
        if "reward.json" in argv[-1]:
            return ProgramResult(
                exit_code=0 if self.reward else 1,
                stdout=self.reward,
                stderr="" if self.reward else "missing reward",
            )
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def read(self, path: str) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        if path == "/tmp/vf-harbor-artifacts.tgz":
            return b"artifact archive"
        raise OSError(path)

    async def write(self, path: str, data: bytes) -> None:
        self.writes.append((path, data))

    async def stop(self) -> None:
        self.stopped = True

    def cleanup(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_separate_verifier_uses_child_runtime_and_json_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = parse_task(
        write_task(
            tmp_path,
            """
environment_mode = "separate"
env = { TOKEN = "${HOST_TOKEN:-fallback}" }

[verifier.environment]
docker_image = "verifier:latest"
network_mode = "no-network"
cpus = 2
memory_mb = 1024
""",
        ),
        0,
        False,
    )
    taskset = HarborTaskset(HarborConfig())
    config = DockerConfig(image="agent:latest")
    agent = FakeRuntime("agent", config=config)
    verifier = FakeRuntime("verifier", json.dumps({"reward": 1, "detail": 0.5}))
    child_configs: list[DockerConfig] = []

    def fake_make_runtime(config, name=None):
        child_configs.append(config)
        assert name == "agent-verifier"
        return verifier

    monkeypatch.setattr("tasksets.harbor_v1.make_runtime", fake_make_runtime)
    trace = Trace(task=task)
    await taskset.score(trace, agent)

    assert trace.reward == 1.0
    assert child_configs == [
        DockerConfig(
            image="verifier:latest",
            cpu=2,
            memory=1,
            network_access=False,
        )
    ]
    assert verifier.started
    assert verifier.stopped
    assert verifier.writes == [("/tmp/vf-harbor-artifacts.tgz", b"artifact archive")]
    artifact_command = agent.commands[1][0]
    assert "/logs/artifacts" in artifact_command
    assert "/tmp/answer.json" in artifact_command
    assert "/data/results" in artifact_command
    assert "saved/results" not in artifact_command
    test_commands = [
        command for command in verifier.commands if "./test.sh" in command[0][-1]
    ]
    assert test_commands[0][1] == {"TOKEN": "fallback"}


@pytest.mark.asyncio
async def test_separate_verifier_stops_after_missing_reward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    task = parse_task(
        write_task(
            tmp_path,
            """
environment_mode = "separate"

[verifier.environment]
docker_image = "verifier:latest"
""",
        ),
        0,
        False,
    )
    taskset = HarborTaskset(HarborConfig())
    agent = FakeRuntime("agent", config=DockerConfig(image="agent:latest"))
    verifier = FakeRuntime("verifier", reward="")
    monkeypatch.setattr(
        "tasksets.harbor_v1.make_runtime", lambda *args, **kwargs: verifier
    )

    reward = await taskset.solved(task, agent)

    assert reward == 0.0
    assert verifier.stopped
    assert "Harbor reward read failed" in caplog.text


@pytest.mark.asyncio
async def test_separate_verifier_stops_when_artifact_transfer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = parse_task(
        write_task(
            tmp_path,
            """
environment_mode = "separate"

[verifier.environment]
docker_image = "verifier:latest"
""",
        ),
        0,
        False,
    )
    taskset = HarborTaskset(HarborConfig())
    agent = FakeRuntime("agent", config=DockerConfig(image="agent:latest"))
    agent.read_error = OSError("transfer failed")
    verifier = FakeRuntime("verifier")
    monkeypatch.setattr(
        "tasksets.harbor_v1.make_runtime", lambda *args, **kwargs: verifier
    )

    with pytest.raises(OSError, match="transfer failed"):
        await taskset.solved(task, agent)

    assert not verifier.started
    assert not verifier.stopped
