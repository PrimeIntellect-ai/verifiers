import subprocess
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping

import pytest

from verifiers.v1.clients import Client, RolloutContext
from verifiers.v1.dialects import Dialect
from verifiers.v1.env import resolve_runtime_config
from verifiers.v1.graph import PendingTurn
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import DockerConfig
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.types import Response, SamplingConfig
from verifiers.v1.tasksets.harbor.taskset import (
    HarborConfig,
    HarborTaskset,
    parse_task,
)
from verifiers.v1.trace import Trace


def write_task_toml(task_dir: Path, content: str) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(content.strip() + "\n")


class FakeRuntime:
    def __init__(self, rewards: list[str]) -> None:
        self.rewards = rewards
        self.writes: list[str] = []
        self.commands: list[list[str]] = []

    async def write(self, path: str, data: bytes) -> None:
        self.writes.append(path)

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append(argv)
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def read(self, path: str) -> bytes:
        if path.endswith("reward.txt") and self.rewards:
            return self.rewards.pop(0).encode()
        raise FileNotFoundError(path)


class MissingRewardRuntime:
    async def write(self, path: str, data: bytes) -> None:
        return None

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def read(self, path: str) -> bytes:
        raise FileNotFoundError(path)


class DummyClient(Client):
    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        raise AssertionError("the test harness should not call the model")


class HarborFixtureHarness(Harness[HarnessConfig]):
    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        return await runtime.run(
            [
                "sh",
                "-c",
                "mkdir -p /workspace && "
                "printf ok >/workspace/answer.txt && "
                "printf ok >/workspace/step1.txt && "
                "printf ok >/workspace/step2.txt",
            ],
            {},
        )


def docker_available() -> bool:
    result = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


async def run_harbor_rollout(taskset: HarborTaskset, task) -> Trace:
    runtime_config = resolve_runtime_config(DockerConfig(), task)
    rollout = Rollout(
        task=task,
        taskset=taskset,
        harness=HarborFixtureHarness(HarnessConfig(id="harbor-fixture")),
        ctx=RolloutContext(
            model="unused", client=DummyClient(), sampling=SamplingConfig()
        ),
        runtime_config=runtime_config,
        setup_timeout=60,
        harness_timeout=60,
        scoring_timeout=60,
    )
    return await rollout.run()


def make_single_step_e2e_task(root: Path) -> Path:
    task_dir = root / "single_step"
    write_task_toml(
        task_dir,
        """
        [task]
        name = "single_step"

        [environment]
        docker_image = "python:3.11-slim"
        workdir = "/workspace"
        """,
    )
    (task_dir / "instruction.md").write_text("Write ok to /workspace/answer.txt")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        """
        set -eu
        mkdir -p /logs/verifier
        if [ "$(cat /workspace/answer.txt 2>/dev/null || true)" = ok ]; then
          printf '{"reward": 1.0, "exact": 1.0}' >/logs/verifier/reward.json
        else
          printf '{"reward": 0.0, "exact": 0.0}' >/logs/verifier/reward.json
        fi
        """.strip()
        + "\n"
    )
    return task_dir


def make_multi_step_e2e_task(root: Path) -> Path:
    task_dir = root / "phase_2"
    write_task_toml(
        task_dir,
        """
        [environment]
        docker_image = "python:3.11-slim"
        workdir = "/workspace"

        [[steps]]
        name = "01_step"
        [steps.agent]
        timeout_sec = 60.0
        [steps.verifier]
        timeout_sec = 60.0

        [[steps]]
        name = "02_step"
        [steps.agent]
        timeout_sec = 60.0
        [steps.verifier]
        timeout_sec = 60.0
        """,
    )
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "helpers.sh").write_text(
        'reward() { printf \'{"reward": %s, "%s": %s}\' "$1" "$2" "$1" >/logs/verifier/reward.json; }\n'
    )
    for name, marker in [("01_step", "step1"), ("02_step", "step2")]:
        step_dir = task_dir / "steps" / name
        (step_dir / "tests").mkdir(parents=True)
        (step_dir / "instruction.md").write_text(f"Write ok to /workspace/{marker}.txt")
        (step_dir / "tests" / "test.sh").write_text(
            f"""
            set -eu
            mkdir -p /logs/verifier
            . /tests/helpers.sh
            if [ "$(cat /workspace/{marker}.txt 2>/dev/null || true)" = ok ]; then
              reward 1.0 {marker}
            else
              reward 0.0 {marker}
            fi
            """.strip()
            + "\n"
        )
    return task_dir


def make_multi_step_task(root: Path) -> Path:
    task_dir = root / "phase_2"
    write_task_toml(
        task_dir,
        """
        [environment]
        workdir = "/workspace/examples"
        memory_mb = 1024
        storage_mb = 2048

        [[steps]]
        name = "01_step"
        [steps.agent]
        timeout_sec = 60.0
        [steps.verifier]
        timeout_sec = 60.0

        [[steps]]
        name = "02_step"
        [steps.agent]
        timeout_sec = 60.0
        [steps.verifier]
        timeout_sec = 60.0
        """,
    )
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "helpers.sh").write_text("true\n")
    for name in ["01_step", "02_step"]:
        step_dir = task_dir / "steps" / name
        (step_dir / "workdir").mkdir(parents=True)
        (step_dir / "workdir" / "setup.sh").write_text("true\n")
        (step_dir / "tests").mkdir()
        (step_dir / "tests" / "test.sh").write_text("true\n")
        (step_dir / "instruction.md").write_text(f"{name} instruction")
    return task_dir


def test_harbor_loads_multi_step_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    make_multi_step_task(dataset_root)
    monkeypatch.setattr(
        "verifiers.v1.tasksets.harbor.taskset.dataset_dir",
        lambda dataset: dataset_root,
    )

    taskset = HarborTaskset(HarborConfig(dataset="fixtures", ignore_dockerfile=True))
    task = taskset.load_tasks()[0]

    assert task.prompt.startswith("Complete this Harbor multi-step task.")
    assert task.workdir == "/workspace/examples"
    assert task.resources.memory == 1.0
    assert task.resources.disk == 2.0
    assert [step.name for step in task.steps] == ["01_step", "02_step"]
    assert [step.harness_timeout for step in task.steps] == [60.0, 60.0]
    assert [step.scoring_timeout for step in task.steps] == [60.0, 60.0]
    assert task.timeout.harness == 120.0
    assert task.timeout.scoring == 120.0
    for step in task.steps:
        assert (Path(step.task_dir) / "workdir" / "setup.sh").is_file()
        assert step.prompt.strip()


@pytest.mark.asyncio
@pytest.mark.docker
async def test_harbor_v1_single_step_rollout_scores_in_real_docker_runtime(
    tmp_path: Path,
) -> None:
    if not docker_available():
        pytest.skip("docker daemon is not available")
    make_single_step_e2e_task(tmp_path)
    taskset = HarborTaskset(
        HarborConfig(dataset=str(tmp_path), dockerfile_policy="ignore")
    )
    task = taskset.load_tasks()[0]

    trace = await run_harbor_rollout(taskset, task)

    assert trace.errors == []
    assert trace.stop_condition == "agent_completed"
    assert trace.reward == 1.0
    assert trace.rewards == {"solved": 1.0}


@pytest.mark.asyncio
@pytest.mark.docker
async def test_harbor_v1_multi_step_rollout_records_step_results_in_real_docker_runtime(
    tmp_path: Path,
) -> None:
    if not docker_available():
        pytest.skip("docker daemon is not available")
    make_multi_step_e2e_task(tmp_path)
    taskset = HarborTaskset(
        HarborConfig(dataset=str(tmp_path), dockerfile_policy="ignore")
    )
    task = taskset.load_tasks()[0]

    trace = await run_harbor_rollout(taskset, task)

    assert trace.errors == []
    assert trace.stop_condition == "agent_completed"
    assert trace.reward == 1.0
    assert trace.info["harbor_steps"] == [
        {"name": "01_step", "rewards": {"reward": 1.0, "step1": 1.0}},
        {"name": "02_step", "rewards": {"reward": 1.0, "step2": 1.0}},
    ]
    assert trace.info["harbor_multi_step_reward"] == {
        "reward": 1.0,
        "step1": 0.5,
        "step2": 0.5,
    }


def test_harbor_multi_step_timeout_is_unbounded_when_any_step_is_unbounded(
    tmp_path: Path,
) -> None:
    task_dir = make_multi_step_task(tmp_path)
    config = (
        (task_dir / "task.toml")
        .read_text()
        .replace(
            "        [steps.verifier]\n        timeout_sec = 60.0\n\n        [[steps]]",
            "\n        [[steps]]",
            1,
        )
    )
    (task_dir / "task.toml").write_text(config)

    task = parse_task(task_dir, 0, HarborConfig())

    assert task.steps[0].scoring_timeout is None
    assert task.timeout.harness == 120.0
    assert task.timeout.scoring is None


def make_dockerfile_task(root: Path) -> Path:
    task_dir = root / "docker_task"
    write_task_toml(
        task_dir,
        """
        [task]
        name = "Docker Task"

        [environment]
        build_timeout_sec = 12.0
        workdir = "/workspace"
        """,
    )
    (task_dir / "instruction.md").write_text("Solve it")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("exit 0\n")
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM python:3.12-slim\n")
    return task_dir


def test_harbor_dockerfile_policy_defaults_to_ignore(tmp_path: Path) -> None:
    task_dir = make_dockerfile_task(tmp_path)

    task = parse_task(task_dir, 0, HarborConfig())

    assert task.image is None
    assert task.environment.kind == "dockerfile"


def test_harbor_dockerfile_policy_error_rejects(tmp_path: Path) -> None:
    task_dir = make_dockerfile_task(tmp_path)

    with pytest.raises(ValueError, match="dockerfile-policy build"):
        parse_task(task_dir, 0, HarborConfig(dockerfile_policy="error"))


def test_harbor_dockerfile_policy_builds_cached_local_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir = make_dockerfile_task(tmp_path)
    commands = []

    def fake_which(name: str) -> str | None:
        return "/usr/bin/docker" if name == "docker" else None

    def fake_run(argv, **kwargs):
        commands.append((argv, kwargs))
        if argv[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("verifiers.v1.tasksets.harbor.taskset.shutil.which", fake_which)
    monkeypatch.setattr("verifiers.v1.tasksets.harbor.taskset.subprocess.run", fake_run)

    task = parse_task(task_dir, 0, HarborConfig(dockerfile_policy="build"))

    assert task.image is not None
    assert task.image.startswith("vf-harbor-docker_task:")
    assert commands[-1][0] == [
        "/usr/bin/docker",
        "build",
        "-t",
        task.image,
        str(task_dir / "environment"),
    ]
    assert commands[-1][1]["timeout"] == 12.0


@pytest.mark.asyncio
async def test_harbor_missing_reward_files_score_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    task_dir = dataset_root / "missing_reward"
    write_task_toml(task_dir, '[task]\nname = "missing_reward"')
    (task_dir / "instruction.md").write_text("Solve it")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("exit 0\n")
    monkeypatch.setattr(
        "verifiers.v1.tasksets.harbor.taskset.dataset_dir",
        lambda dataset: dataset_root,
    )

    taskset = HarborTaskset(HarborConfig(dataset="fixtures", ignore_dockerfile=True))
    task = taskset.load_tasks()[0]
    runtime = MissingRewardRuntime()
    rewards = await taskset.run_verifier(
        runtime, tests_dir=tests_dir, solution_dir=task_dir / "solution"
    )

    assert rewards == {}
    assert await taskset.solved(task, Trace(task=task), runtime) == 0.0


@pytest.mark.asyncio
async def test_harbor_multi_step_solved_aggregates_step_rewards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    make_multi_step_task(dataset_root)
    monkeypatch.setattr(
        "verifiers.v1.tasksets.harbor.taskset.dataset_dir",
        lambda dataset: dataset_root,
    )

    taskset = HarborTaskset(HarborConfig(dataset="fixtures", ignore_dockerfile=True))
    task = taskset.load_tasks()[0]
    trace = Trace(task=task)
    runtime = FakeRuntime(['{"reward": 0.5, "style": 1.0}', '{"reward": 1.0}'])

    assert await taskset.solved(task, trace, runtime) == 0.75

    assert trace.task is task
    assert trace.info["harbor_multi_step_reward"] == {"reward": 0.75, "style": 0.5}
    assert [record["name"] for record in trace.info["harbor_steps"]] == [
        "01_step",
        "02_step",
    ]
    assert runtime.writes.count("/tmp/harbor_tests.tgz") == 2


@pytest.mark.asyncio
async def test_harbor_missing_multi_step_reward_counts_as_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    make_multi_step_task(dataset_root)
    monkeypatch.setattr(
        "verifiers.v1.tasksets.harbor.taskset.dataset_dir",
        lambda dataset: dataset_root,
    )

    taskset = HarborTaskset(HarborConfig(dataset="fixtures", ignore_dockerfile=True))
    task = taskset.load_tasks()[0]
    trace = Trace(task=task)
    runtime = FakeRuntime(['{"reward": 1.0}'])

    assert await taskset.solved(task, trace, runtime) == 0.5
    assert trace.info["harbor_steps"] == [
        {"name": "01_step", "rewards": {"reward": 1.0}},
        {"name": "02_step", "rewards": {}},
    ]
    assert trace.info["harbor_multi_step_reward"] == {"reward": 0.5}
