"""A Harbor grader reads the solver's transcript and final reply from the trace, never
from the solver's box, and a gold solution's reply reaches it the same way."""

import shutil
from pathlib import Path

import pytest

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.tasksets.harbor import taskset as harbor
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask
from verifiers.v1.validation import validate_attempt

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not shutil.which("docker"), reason="needs Docker"),
]

IMAGE = "python:3.11-slim"
REPLY = "The parser now keeps the truncated tail."
# Rewards 1 when the staged final reply is exactly REPLY.
GRADES_REPLY = (
    "#!/bin/sh\n"
    f'if [ "$(cat /logs/agent/final_reply.md)" = "{REPLY}" ]; '
    "then echo 1 > /logs/verifier/reward.txt; "
    "else echo 0 > /logs/verifier/reward.txt; fi\n"
)


def harbor_task(tmp_path: Path, solve: str | None = None, reply: str | None = None):
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests/test.sh").write_text(GRADES_REPLY)
    if solve is not None or reply is not None:
        (tmp_path / "solution").mkdir(exist_ok=True)
        (tmp_path / "solution/solve.sh").write_text(solve or "true\n")
        if reply is not None:
            (tmp_path / "solution/reply.md").write_text(reply)
    return HarborTask(
        HarborData(idx=0, prompt="Fix the parser.", image=IMAGE, task_dir=str(tmp_path))
    )


def solver_trace(task: HarborTask, reply: str) -> vf.Trace:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask.of(task),
    )
    trace.nodes += [
        MessageNode(message=vf.UserMessage(content="Fix the parser.")),
        MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
    ]
    return trace


async def read(runtime: vf.Runtime, path: str) -> str | None:
    if (await runtime.run(["test", "-e", path], {})).exit_code:
        return None
    return (await runtime.read(path)).decode()


async def test_the_grader_reads_the_traces_reply_not_the_solvers_files(tmp_path):
    task = harbor_task(tmp_path)
    trace = solver_trace(task, REPLY)
    async with provision_runtime(vf.DockerConfig(image=IMAGE)) as runtime:
        # The solver forged its transcript and reply, and left a file of its own.
        await runtime.run(
            [
                "sh",
                "-c",
                (
                    "mkdir -p /logs/agent && echo forged > /logs/agent/final_reply.md && "
                    "echo '{}' > /logs/agent/trajectory.json && touch /logs/agent/extra"
                ),
            ],
            {},
        )
        assert await task.solved(runtime, trace) == {"reward": 1.0}
        assert await read(runtime, "/logs/agent/final_reply.md") == REPLY
        assert REPLY in await read(runtime, "/logs/agent/trajectory.json")
        assert await read(runtime, "/logs/agent/extra") is None


async def test_a_run_without_a_reply_stages_none(tmp_path):
    task = harbor_task(tmp_path)
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()), task=vf.TraceTask.of(task)
    )
    async with provision_runtime(vf.DockerConfig(image=IMAGE)) as runtime:
        await runtime.run(
            [
                "sh",
                "-c",
                f"mkdir -p /logs/agent && echo '{REPLY}' > /logs/agent/final_reply.md",
            ],
            {},
        )
        assert await task.solved(runtime, trace) == {"reward": 0.0}
        assert await read(runtime, "/logs/agent/final_reply.md") is None


@pytest.mark.parametrize(
    "solve,reply",
    [
        (f"echo '{REPLY}' > /solution/reply.md\n", None),  # written by solve.sh
        (None, REPLY),  # shipped in solution/
    ],
)
async def test_a_gold_solutions_reply_is_its_final_message(tmp_path, solve, reply):
    task = harbor_task(tmp_path, solve=solve, reply=reply)
    result = await task.validate(vf.DockerConfig(image=IMAGE))
    assert result.gold.reward == 1.0, result.gold.error
    assert result.baseline.reward == 0.0
    trace = vf.Trace.model_validate(result.gold.trace)
    assert trace.messages[-1].content.strip() == REPLY
    assert trace.num_turns == 0  # no model sampled it


async def test_a_non_gold_run_never_reads_the_reply_path(tmp_path):
    task = harbor_task(tmp_path)
    attempt = await validate_attempt(
        task,
        vf.DockerConfig(image=IMAGE),
        command=[
            "sh",
            "-c",
            f"mkdir -p /solution && echo '{REPLY}' > /solution/reply.md",
        ],
    )
    assert attempt.reward == 0.0, attempt.error


def test_an_artifact_into_the_agent_logs_is_refused(tmp_path, monkeypatch):
    task = tmp_path / "forges-transcript"
    (task / "tests").mkdir(parents=True)
    (task / "tests/test.sh").write_text("exit 0\n")
    (task / "instruction.md").write_text("Do it.")
    (task / "task.toml").write_text(
        'artifacts = ["/logs/agent/trajectory.json"]\n'
        '[environment]\ndocker_image = "python:3.11-slim"\n'
    )
    monkeypatch.setattr(harbor, "dataset_dir", lambda config: tmp_path)
    with pytest.raises(ValueError, match="/logs/agent"):
        list(harbor.HarborTaskset(harbor.HarborConfig()))
