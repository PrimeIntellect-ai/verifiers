"""Harbor's verifier run: what it records on the trace and how a missing reward fails."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError, TaskError
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.tasksets.harbor.taskset import (
    HarborData,
    HarborTask,
    HarborTaskConfig,
    log_tails,
)

LOGS = "==> /logs/verifier/pg.log <==\nFATAL: could not start\n\n"


class FakeRuntime:
    """Answers the verifier's commands: test.sh, the log listing, the reward reads."""

    def __init__(self, reward: bytes | None, listing: str | Exception = LOGS):
        self.reward = reward
        self.listing = listing

    async def run(self, argv, env):
        if argv == ["bash", "/tests/test.sh"]:
            return ProgramResult(exit_code=3, stdout="ran tests\n", stderr="boom\n")
        if "find /logs/verifier" in argv[-1]:
            if isinstance(self.listing, Exception):
                raise self.listing
            return ProgramResult(exit_code=0, stdout=self.listing, stderr="")
        # `read_reward_json`'s existence probe: no reward.json.
        return ProgramResult(exit_code=1, stdout="", stderr="")

    async def read(self, path, max_bytes=None):
        assert path == "/logs/verifier/reward.txt"
        if self.reward is None:
            raise SandboxError(f"{path} not found", kind="missing")
        return self.reward


def make_task() -> HarborTask:
    return HarborTask(HarborData(idx=0, prompt="p", task_dir="/nonexistent"))


def make_trace(task: HarborTask) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="HarborTask", data=task.data),
    )


async def test_verifier_output_and_logs_land_on_the_trace():
    task = make_task()
    trace = make_trace(task)
    assert await task.run_verifier(FakeRuntime(b"0.5"), trace) == {"reward": 0.5}
    assert trace.info["verifier"] == {
        "exit_code": 3,
        "output": "ran tests\nboom\n",
        "files": LOGS,
    }


async def test_a_failed_log_listing_leaves_grading_to_the_reward():
    task = make_task()
    trace = make_trace(task)
    runtime = FakeRuntime(b"1", listing=SandboxError("wrapper died", kind="exec"))
    assert await task.run_verifier(runtime, trace) == {"reward": 1.0}
    assert trace.info["verifier"]["files"] == ""


async def test_a_verifier_that_writes_no_reward_fails_the_task_with_its_logs():
    task = make_task()
    with pytest.raises(TaskError) as raised:
        await task.run_verifier(FakeRuntime(None), make_trace(task))
    # A task failure, not a box failure: the reward's absence is the cause.
    assert isinstance(raised.value.__cause__, SandboxError)
    assert raised.value.__cause__.kind == "missing"
    assert "FATAL: could not start" in str(raised.value)


def test_log_tails_keep_the_end_of_every_log():
    noisy = "==> /logs/verifier/download.log <==\n" + "x" * 5000 + "\n"
    cause = "==> /logs/verifier/pg.log <==\nstart\nFATAL: no space\n"
    tails = log_tails(noisy + cause, size=100)
    assert tails.startswith("==> /logs/verifier/download.log <==\n")
    assert tails.count("x") <= 100
    assert tails.endswith(cause)


@pytest.mark.parametrize("authored", [None, "task-compose-host"])
@pytest.mark.parametrize("mirrors", [[], ["https://mirror.gcr.io"]])
def test_compose_host_configuration_preserves_authored_task(
    tmp_path, authored, mirrors
):
    environment = tmp_path / "environment"
    environment.mkdir()
    source = environment / "docker-compose.yaml"
    content = "services:\n  main:\n    build: .\n"
    source.write_text(content)
    data = HarborData(prompt="p", task_dir=str(tmp_path), compose_host_image=authored)
    original = data.model_dump_json()
    config = HarborTaskConfig(compose_registry_mirrors=mirrors)
    restored = HarborTaskConfig.model_validate_json(config.model_dump_json())

    spec = HarborTask(data, restored).deployment_spec()

    assert spec.host_image == authored
    assert spec.registry_mirrors == tuple(mirrors)
    assert spec.directory == str(tmp_path)
    assert spec.file == "environment/docker-compose.yaml"
    assert source.read_text() == content
    assert data.model_dump_json() == original
