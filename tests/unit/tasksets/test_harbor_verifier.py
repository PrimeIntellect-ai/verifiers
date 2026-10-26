"""Harbor's verifier run: what it records on the trace and how a missing reward fails."""

import asyncio

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.scoring import ScoringConfig
from verifiers.v1.errors import SandboxError, TaskError
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.tasksets.harbor import taskset
from verifiers.v1.tasksets.harbor.taskset import (
    WITHOUT_SYS_RESOURCE,
    HarborData,
    HarborTask,
    HarborTaskConfig,
    log_tails,
)

TEST_SH = ["sh", "-c", WITHOUT_SYS_RESOURCE, "vf-verifier", "bash", "/tests/test.sh"]

LOGS = "==> /logs/verifier/pg.log <==\nFATAL: could not start\n\n"


class FakeRuntime:
    """Answers the verifier's commands: test.sh, the log listing, the reward reads."""

    def __init__(self, reward: bytes | None, listing: str | Exception = LOGS):
        self.reward = reward
        self.listing = listing

    async def run(self, argv, env):
        if argv == TEST_SH:
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


class TrajectoryRuntime(FakeRuntime):
    """Also keeps what grading wrote and the commands it ran."""

    def __init__(self):
        super().__init__(b"1")
        self.writes, self.commands = {}, []

    async def write(self, path, data):
        self.writes[path] = data

    async def run(self, argv, env):
        self.commands.append(argv)
        return await super().run(argv, env)


async def test_grading_leaves_the_agent_transcript_where_harbor_graders_read_it(
    monkeypatch,
):
    # CyberStrike's process and output-quality judges skip unless they find Harbor's
    # /logs/agent/trajectory.json, which Harbor's own agents write.
    import json

    messages = [
        vf.UserMessage(content="Task Description: audit cJSON"),
        vf.AssistantMessage(
            content="Looking.",
            tool_calls=[vf.ToolCall(id="c1", name="bash", arguments='{"cmd": "ls"}')],
        ),
        vf.ToolMessage(tool_call_id="c1", content="cJSON.c"),
    ]
    monkeypatch.setattr(vf.Trace, "messages", property(lambda self: messages))
    task = make_task()
    runtime = TrajectoryRuntime()
    assert await task.run_verifier(runtime, make_trace(task)) == {"reward": 1.0}
    steps = json.loads(runtime.writes["/tmp/vf-trajectory.json"])["steps"]
    assert [s["source"] for s in steps] == ["user", "agent", "tool"]
    assert steps[0]["message"] == "Task Description: audit cJSON"
    assert steps[1]["tool_calls"] == [
        {"tool_call_id": "c1", "function_name": "bash", "arguments": '{"cmd": "ls"}'}
    ]
    staged = next(i for i, a in enumerate(runtime.commands) if "trajectory" in a[-1])
    assert "test -e /logs/agent/trajectory.json ||" in runtime.commands[staged][-1]
    assert staged < runtime.commands.index(TEST_SH)


async def test_grading_a_run_without_a_transcript_writes_no_trajectory():
    task = make_task()
    runtime = TrajectoryRuntime()
    await task.run_verifier(runtime, make_trace(task))
    assert runtime.writes == {}


class StagingRuntime(TrajectoryRuntime):
    async def run(self, argv, env):
        self.commands.append(argv)
        return ProgramResult(exit_code=0, stdout="", stderr="")


async def test_an_isolated_grader_owns_restored_artifacts_as_harbor_uploads_them(
    tmp_path,
):
    """Harbor uploads a separate verifier's artifacts root-owned under world-writable
    roots, so a verifier running as its own user can work beside them."""
    task = HarborTask(
        HarborData(
            idx=0,
            prompt="p",
            task_dir=str(tmp_path),
            scoring=ScoringConfig(mode="isolated"),
            verifier_image="grader:1",
        )
    )
    trace = make_trace(task)
    (tmp_path / "tests").mkdir()
    trace.state.artifacts = {
        "/logs/artifacts": b"tar",
        "/app/out.txt": b"tar",
        "/x": None,
    }
    runtime = StagingRuntime()
    await task.stage_verifier(trace, runtime)
    owned = next(a for a in runtime.commands if a[3:4] == ["vf-artifacts"])
    assert owned[4:] == ["/logs/artifacts", "/app/out.txt"]
    assert 'chown -hR 0:0 "$r" && chmod 777 "$r"' in owned[2]
    assert runtime.commands.index(owned) < len(runtime.commands) - 1  # before the tests


async def test_a_shared_grader_leaves_the_agents_files_as_they_are(tmp_path):
    (tmp_path / "tests").mkdir()
    task = HarborTask(HarborData(idx=0, prompt="p", task_dir=str(tmp_path)))
    trace = make_trace(task)
    trace.state.artifacts = {"/logs/artifacts": b"tar"}
    runtime = StagingRuntime()
    await task.stage_verifier(trace, runtime)
    assert not [a for a in runtime.commands if a[3:4] == ["vf-artifacts"]]


async def test_the_verifier_runs_as_its_declared_user():
    class UserRuntime(FakeRuntime):
        user = None

        def with_user(self, user, home=None):
            bound = UserRuntime(self.reward)
            bound.user = user
            return bound

        async def run(self, argv, env):
            if argv == TEST_SH:
                users.append(self.user)
            return await super().run(argv, env)

    users = []
    task = HarborTask(
        HarborData(idx=0, prompt="p", task_dir="/nonexistent", verifier_user="reliant")
    )
    await task.run_verifier(UserRuntime(b"1"), make_trace(task))
    assert users == ["reliant"]


class HangingRuntime(FakeRuntime):
    """test.sh never returns; `answers` says whether the box still runs other commands."""

    def __init__(self, answers: bool = True):
        super().__init__(b"1")
        self.answers = answers

    async def run(self, argv, env):
        if argv == TEST_SH or (argv == ["true"] and not self.answers):
            await asyncio.sleep(3600)
        return await super().run(argv, env)


def timed_task(tmp_path, timeout_sec: float) -> HarborTask:
    (tmp_path / "task.toml").write_text(
        f'[verifier]\nenvironment_mode = "separate"\ntimeout_sec = {timeout_sec}\n'
    )
    return HarborTask(HarborData(idx=0, prompt="p", task_dir=str(tmp_path)))


async def test_a_verifier_past_its_declared_timeout_is_graded_zero(tmp_path):
    # pyappkit-psygnal (2026-10-26): test.sh hung on a live box for 3.5 hours past the
    # task's 2 h timeout; `ignore_timeouts` had dropped it from the scoring budget.
    task = timed_task(tmp_path, 0.2)
    trace = make_trace(task)
    async with asyncio.timeout(10):
        assert await task.run_verifier(HangingRuntime(), trace) == {"reward": 0.0}
    assert trace.info["verifier"]["timed_out"] == 0.2
    assert trace.info["verifier"]["exit_code"] == 124
    assert trace.info["verifier"]["files"] == LOGS


async def test_a_verifier_whose_box_stops_answering_is_a_sandbox_error(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(taskset, "VERIFIER_PROBE_SECONDS", 0.2)
    task = timed_task(tmp_path, 0.2)
    with pytest.raises(SandboxError) as raised:
        async with asyncio.timeout(10):
            await task.run_verifier(HangingRuntime(answers=False), make_trace(task))
    assert raised.value.kind == "io"


async def test_the_scoring_budget_extends_a_shorter_declared_timeout(tmp_path):
    task = timed_task(tmp_path, 7200)
    assert task.verifier_timeout() == 7200
    longer = task.data.model_copy(update={"timeout": vf.TaskTimeout(scoring=9000.0)})
    assert HarborTask(longer).verifier_timeout() == 9000
    assert make_task().verifier_timeout() is None
