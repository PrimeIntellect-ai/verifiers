# Pre-agent filenames must retain their identity through patch capture.

from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.utils.git import capture_patch, snapshot_untracked


class LossyRuntime:
    async def run(self, argv, env):
        return vf.ProgramResult(exit_code=0, stdout="\ufffd\0", stderr="")

    async def read(self, path):
        return b"\xff\0"


async def test_capture_patch_treats_pathspec_magic_as_literal():
    git_env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    async with provision_runtime(vf.SubprocessConfig()) as runtime:
        setup = await runtime.run(
            [
                "sh",
                "-c",
                (
                    "git init -q && "
                    "git -c user.name=vf -c user.email=vf@example.com "
                    "commit --allow-empty -qm base && "
                    "printf shipped > ':(glob)*.py'"
                ),
            ],
            git_env,
        )
        assert setup.exit_code == 0, setup.stderr
        ignored = await snapshot_untracked(runtime, git_env)
        await runtime.write("changed.py", b"changed")

        trace = SimpleNamespace(info={})
        await capture_patch(trace, runtime, env=git_env, ignore=ignored)

    diff_headers = [
        line
        for line in trace.info["patch"].splitlines()
        if line.startswith("diff --git ")
    ]
    assert diff_headers == ["diff --git a/changed.py b/changed.py"]


async def test_snapshot_untracked_rejects_non_utf8_filename():
    with pytest.raises(UnicodeDecodeError):
        await snapshot_untracked(LossyRuntime())
