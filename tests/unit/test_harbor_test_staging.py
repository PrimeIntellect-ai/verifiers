"""Staging a Harbor task's packaged tests leaves `/tests` holding exactly those tests.

In shared grading the agent's box is the grading box, so anything the agent left in
`/tests` (a `conftest.py`, a `pytest.ini`) would otherwise be loaded by `test.sh`. The
staging commands run for real under `sh`, with `/tests`, `/logs` and `/tmp` mapped into
a temporary directory."""

import asyncio
import re
import shutil

import pytest

pytest.importorskip("harbor")

from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.tasksets.harbor.taskset import HarborConfig, HarborTask, parse_task

pytestmark = pytest.mark.skipif(
    any(shutil.which(t) is None for t in ("sh", "tar", "find", "stat", "chown")),
    reason="needs sh, tar, find, stat and chown",
)

ROOTS = re.compile(r"(?<![\w./-])/(tests|logs|tmp)\b")


class Box:
    """A runtime whose `/tests`, `/logs` and `/tmp` live under `root`."""

    class backend:
        SUPPORTS_USER_SWITCHING = False

    def __init__(self, root):
        self.root = root

    def local(self, text):
        return ROOTS.sub(lambda m: f"{self.root}/{m.group(1)}", text)

    def with_user(self, user):
        return self

    async def write(self, path, data):
        target = self.root / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def run(self, argv, env):
        process = await asyncio.create_subprocess_exec(
            *(self.local(arg) for arg in argv),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return ProgramResult(process.returncode, stdout.decode(), stderr.decode())


def leave_nothing(tests):
    pass


def leave_a_conftest(tests):
    tests.mkdir(parents=True)
    (tests / "conftest.py").write_text("raise SystemExit\n")
    (tests / "helpers").mkdir()


def leave_a_file(tests):
    tests.parent.mkdir(parents=True)
    tests.write_text("not a directory\n")


def leave_a_symlink(tests):
    elsewhere = tests.parent / "elsewhere"
    elsewhere.mkdir(parents=True)
    (elsewhere / "conftest.py").write_text("raise SystemExit\n")
    tests.symlink_to(elsewhere, target_is_directory=True)


@pytest.mark.parametrize(
    "left", [leave_nothing, leave_a_conftest, leave_a_file, leave_a_symlink]
)
async def test_staged_tests_replace_whatever_was_in_tests(tmp_path, left):
    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Do nothing.")
    (task_dir / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[environment]\ndocker_image = "python:3.11-slim"\n'
    )
    config = HarborConfig()
    data = parse_task(task_dir, 0, config)
    assert data.scoring.mode == "shared"

    box = Box(tmp_path / "box")
    left(box.root / "tests")
    await HarborTask(data, config.task).stage_tests(box)
    tests = box.root / "tests"
    assert not tests.is_symlink()
    assert sorted(p.name for p in tests.iterdir()) == ["test.sh"]
