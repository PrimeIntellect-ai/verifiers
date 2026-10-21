"""Any task, not only Harbor, grades its untouched and gold attempts when a caller asks."""

from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.runtimes.backends.subprocess import SubprocessConfig


class FileTask(vf.Task[vf.TaskData, vf.State, vf.TaskConfig]):
    @vf.reward(weight=1.0)
    async def passed(self, trace: vf.Trace, runtime) -> float:
        result = await runtime.run(["bash", "-c", "grep -qx ok out.txt"], {})
        return float(result.exit_code == 0)


def task() -> FileTask:
    return FileTask(vf.TaskData(idx=0, name="file", prompt="Write ok to out.txt."))


async def test_a_plain_task_grades_its_gold_and_untouched_attempts(tmp_path: Path):
    gold = tmp_path / "solution"
    gold.mkdir()
    (gold / "solve.sh").write_text("echo ok > out.txt\n")
    result = await task().validate(
        SubprocessConfig(),
        baseline=True,
        gold=True,
        gold_source=gold,
        gold_target_dir="solution",
        gold_command=["bash", "solution/solve.sh"],
    )
    assert result.gold.enabled and result.gold.valid is True
    assert (
        result.baseline.enabled and result.baseline.valid is True
    )  # untouched scores 0.0


async def test_a_plain_task_without_a_request_has_no_model_free_check(tmp_path: Path):
    assert await task().validate(SubprocessConfig()) is None
    result = await task().validate(SubprocessConfig(), gold=True)
    assert (
        result.gold.valid is None and "gold solution unavailable" in result.gold.error
    )
    assert not result.baseline.enabled


async def test_the_reference_runs_as_the_agents_user(tmp_path: Path):
    """The reference is staged as root and runs as the task's agent user, as Harbor's
    oracle runs solve.sh."""
    from types import SimpleNamespace

    from verifiers.v1.runtimes import ProgramResult
    from verifiers.v1.validation import ValidationHarness, ValidationHarnessConfig

    ran: list[tuple[str | None, list[str]]] = []

    class Box:
        def __init__(self, user):
            self.user = user

        def with_user(self, user, home=None):
            return Box(user)

        async def run(self, argv, env):
            ran.append((self.user, argv))
            return ProgramResult(exit_code=0, stdout="", stderr="")

    harness = ValidationHarness(ValidationHarnessConfig(command=["bash", "solve.sh"]))
    harness.task, harness.kind = task(), "gold"
    trace = SimpleNamespace(info={})
    await harness.launch(None, trace, Box("agent"), None, None, [], None)
    assert ran == [("agent", ["bash", "solve.sh"])]
    assert trace.info["oracle"]["user"] == "agent"


async def test_the_staged_reference_is_reachable_by_the_agent(tmp_path: Path):
    """Staging the reference leaves the modes `docker cp` of its directory would,
    the target's own included, so an image's root-only /solution (0700) does not
    keep the agent from running solve.sh."""
    from types import SimpleNamespace

    from verifiers.v1.runtimes import provision_runtime
    from verifiers.v1.validation import ValidationHarness, ValidationHarnessConfig

    source = tmp_path / "solution"
    (source / "lib").mkdir(parents=True)
    (source / "solve.sh").write_text("true\n")
    (source / "lib" / "helper.sh").write_text("true\n")
    (source / "lib" / "helper.sh").chmod(0o775)
    target = tmp_path / "box-solution"
    target.mkdir(mode=0o700)
    harness = ValidationHarness(
        ValidationHarnessConfig(source=source, target_dir=str(target))
    )
    harness.task, harness.kind = task(), "gold"
    async with provision_runtime(SubprocessConfig()) as box:
        await harness.launch(None, SimpleNamespace(info={}), box, None, None, [], None)
    assert target.stat().st_mode & 0o777 == source.stat().st_mode & 0o755
    assert (target / "lib" / "helper.sh").stat().st_mode & 0o777 == 0o755
