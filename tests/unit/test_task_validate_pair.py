"""Any task, not only Harbor, grades its untouched and gold attempts when a caller asks."""

from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.runtimes.subprocess import SubprocessConfig


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
