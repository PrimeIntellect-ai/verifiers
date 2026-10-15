"""Host-side validation of artifact archives before they restore into a grading box."""

import io
import tarfile
from types import SimpleNamespace

import pytest

from verifiers.v1.utils.artifacts import _validate_restore, restore

GRADER = ("/tests", "/logs/verifier")


def archive(*entries: tuple[str, str | None]) -> bytes:
    """A tar of ``(name, link_target)`` entries: a directory when the name ends in
    ``/``, a symlink when a target is given, otherwise a small regular file."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, target in entries:
            info = tarfile.TarInfo(name.rstrip("/"))
            if name.endswith("/"):
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif target is not None:
                info.type = tarfile.SYMTYPE
                info.linkname = target
                tar.addfile(info)
            else:
                info.size = 2
                tar.addfile(info, io.BytesIO(b"ok"))
    return buffer.getvalue()


def box() -> tuple[SimpleNamespace, list[str]]:
    commands: list[str] = []

    async def run(argv, env):
        commands.append(argv[-1])
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def write(path, data):
        commands.append(f"write {path}")

    runtime = SimpleNamespace(
        config=SimpleNamespace(type="docker", workdir="/app"), run=run, write=write
    )
    return runtime, commands


@pytest.mark.asyncio
async def test_a_virtualenv_with_links_to_the_image_interpreter_restores() -> None:
    runtime, commands = box()
    venv = archive(
        ("app/", None),
        ("app/venv/", None),
        ("app/venv/bin/", None),
        ("app/venv/bin/python3", "/usr/bin/python3"),
        ("app/venv/bin/python", "python3"),
        ("app/lib", "../usr/lib"),
        ("app/current", "/app/venv"),
    )
    await restore(runtime, {"/app": venv}, protected=GRADER)
    assert any(command.startswith("tar -xf") for command in commands)


@pytest.mark.parametrize(
    "entries",
    [
        [("app/answer", "/tests/expected.json")],
        [("app/answer", "../tests")],
        [("app/answer", "/logs/verifier/reward.txt")],
        [("app/answer", "/logs")],
        [("app/answer", "/")],
        # Kernel resolution: `..` applies after the link before it resolved.
        [("app/up", "/usr"), ("app/answer", "up/../tests/expected.json")],
        [("app/a", "b"), ("app/b", "/tests")],
    ],
)
@pytest.mark.asyncio
async def test_a_link_that_lands_on_the_grader_files_is_refused(entries) -> None:
    runtime, commands = box()
    with pytest.raises(RuntimeError, match="grader's own files"):
        await restore(
            runtime, {"/app": archive(("app/", None), *entries)}, protected=GRADER
        )
    assert commands == []


@pytest.mark.asyncio
async def test_without_protected_paths_any_link_target_restores() -> None:
    runtime, _ = box()
    await restore(runtime, {"/app": archive(("app/", None), ("app/t", "/tests"))})


@pytest.mark.asyncio
async def test_a_link_loop_toward_the_grader_is_refused() -> None:
    runtime, _ = box()
    loop = archive(("app/", None), ("app/a", "b"), ("app/b", "a"))
    with pytest.raises(RuntimeError, match="loops"):
        await restore(runtime, {"/app": loop}, protected=GRADER)


def test_extraction_never_writes_through_a_link() -> None:
    with pytest.raises(RuntimeError, match="traverses a symlink"):
        _validate_restore(
            "/app", archive(("app/", None), ("app/a", "/etc"), ("app/a/cron", None))
        )
    with pytest.raises(RuntimeError, match="unsafe target"):
        _validate_restore("/app", archive(("app", "/srv")))


@pytest.mark.asyncio
async def test_one_archive_cannot_extract_through_a_link_another_restores() -> None:
    runtime, commands = box()
    collected = {
        "/app": archive(("app/", None), ("app/lib", "/usr/lib")),
        "/app/lib/evil": archive(("app/lib/evil/", None), ("app/lib/evil/x", None)),
    }
    with pytest.raises(RuntimeError, match="restored by '/app'"):
        await restore(runtime, collected)
    assert commands == []


@pytest.mark.asyncio
async def test_isolated_scoring_keeps_restored_links_out_of_the_tasks_grader_paths() -> (
    None
):
    import verifiers.v1 as vf
    from verifiers.v1.grading import stage_verifier
    from verifiers.v1.tasksets.harbor.taskset import GRADER_PATHS, HarborTask

    class GradedTask(vf.Task):
        RESTORE_PROTECTED = GRADER_PATHS

    assert HarborTask.RESTORE_PROTECTED == GRADER_PATHS
    runtime, commands = box()
    trace = vf.Trace(
        task=vf.TraceTask(type="GradedTask", data=vf.TaskData(idx=0)),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
    )
    trace.state.artifacts["/app"] = archive(("app/", None), ("app/t", "/tests"))
    with pytest.raises(RuntimeError, match="grader's own files"):
        await stage_verifier(GradedTask(vf.TaskData(idx=0)), trace, runtime)
    assert commands == []  # refused before the grading box was touched
    await stage_verifier(vf.Task(vf.TaskData(idx=0)), trace, runtime)
    assert commands
