"""Harbor's verifier run: what it records on the trace."""

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask

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
