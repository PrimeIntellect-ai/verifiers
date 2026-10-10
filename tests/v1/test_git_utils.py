"""Focused coverage for the filename-handling gaps in #2196: ignore entries
must reach git as literal pathspecs, and non-UTF-8 filenames must fail loudly
instead of producing a wrong patch."""

from types import SimpleNamespace

import pytest

from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.utils.git import capture_patch, snapshot_untracked


class ScriptedRuntime:
    """Duck-typed runtime: records run() argv and plays back scripted results."""

    def __init__(self, results: list[ProgramResult], read_data: bytes = b""):
        self.calls: list[list[str]] = []
        self._results = list(results)
        self._read_data = read_data

    async def run(self, cmd: list[str], env: dict) -> ProgramResult:
        self.calls.append(list(cmd))
        return self._results.pop(0)

    async def read(self, path: str) -> bytes:
        return self._read_data


@pytest.mark.asyncio
async def test_capture_patch_passes_ignore_entries_as_literal_pathspecs():
    # A file named `*.py` must not glob-unstage every Python file the agent
    # touched, and a leading `:` must not be parsed as pathspec magic.
    ok = ProgramResult(exit_code=0, stdout="", stderr="")
    runtime = ScriptedRuntime([ok, ok], read_data=b"")
    trace = SimpleNamespace(info={})

    await capture_patch(trace, runtime, ignore=["*.py", "weird[name].txt", ":odd"])

    diff_argv = runtime.calls[0]
    assert diff_argv[4:] == [
        ":(literal)*.py",
        ":(literal)weird[name].txt",
        ":(literal):odd",
    ]


@pytest.mark.asyncio
async def test_snapshot_untracked_splits_nul_delimited_names():
    runtime = ScriptedRuntime(
        [ProgramResult(exit_code=0, stdout="a.txt\0dir/b bin\0", stderr="")]
    )
    assert await snapshot_untracked(runtime) == ["a.txt", "dir/b bin"]


@pytest.mark.asyncio
async def test_snapshot_untracked_rejects_lossily_decoded_names():
    # Runtimes decode stdout with errors="replace": a non-UTF-8 filename
    # arrives as U+FFFD and can never match the real file again. Returning it
    # would leave the file out of the ignore set and credit the agent with an
    # image file — fail before a wrong patch can be produced.
    runtime = ScriptedRuntime(
        [ProgramResult(exit_code=0, stdout="caf�.bin\0", stderr="")]
    )
    with pytest.raises(ValueError, match="not\\s+valid UTF-8"):
        await snapshot_untracked(runtime)
