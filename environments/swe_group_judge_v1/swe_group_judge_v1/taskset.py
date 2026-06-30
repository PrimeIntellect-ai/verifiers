"""swe_group_judge: an agentic group judge over SWE-style tasks (showcases AgenticGroupJudge).

The policy edits a repo in a sandbox; rollouts are NOT scored per-rollout. After each rollout
(runtime still live) ``finalize`` snapshots the agent's diff + transcript onto the trace, since
the runtime is torn down before group scoring. The single ``@group_reward`` then hands the whole
group to an agentic LLM judge (``vf.AgenticGroupJudge``): it rebuilds each candidate (base + that
rollout's patch) in a fresh sandbox from ``task.image``, runs the *pristine* tests against each,
and returns one comparative score per rollout. Those become the group's rewards — GRPO baselines
them in prime-rl.

Run with ``-r 2`` or more (group rewards compare a task's rollouts), a container runtime
(docker/prime — the judge rebuilds from ``task.image``), and a judge endpoint, e.g.::

    prime eval run swe-group-judge-v1 -r 4 \\
        --taskset.judge.model <model> --taskset.judge.base_url <url>

``load_tasks`` below ships one illustrative task; replace it with your SWE rows (each sets its
container ``image`` and a host ``tests_dir`` holding the pristine suite).
"""

import functools
import io
import tarfile
from pathlib import Path

import verifiers.v1 as vf

EXAMPLE_TESTS = str(Path(__file__).parent / "tasks" / "example" / "tests")


def make_tar(directory: str) -> bytes:
    """Tar a directory's contents (arcname='.') — the pristine tests staged into each candidate."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(directory, arcname=".")
    return buf.getvalue()


class SweTask(vf.Task):
    instruction: str
    """The change to make (becomes the rollout prompt)."""
    tests_dir: str
    """Host path to the pristine test suite, staged into each candidate's ``_tests/``."""


class JudgedConfig(vf.TasksetConfig):
    judge: vf.JudgeConfig = vf.JudgeConfig()
    """The judge model/endpoint (its own endpoint, not the policy)."""
    judge_runtime: vf.RuntimeConfig = vf.DockerConfig()
    """The judge sandbox; ``task.image`` is injected per group so it carries the task toolchain."""


class SweJudge(vf.AgenticGroupJudge):
    """The repo lives at ``/repo`` in the task image; tests come from the task's host dir."""

    repo_path = "/repo"

    def tests_tar(self, task: SweTask) -> bytes:
        return make_tar(task.tests_dir)


class SweGroupJudgeTaskset(vf.Taskset[SweTask, JudgedConfig]):
    def load_tasks(self) -> list[SweTask]:
        return [
            SweTask(
                idx=0,
                name="example",
                image="ghcr.io/your-org/your-repo:base",  # repo checked out at /repo
                prompt="Fix parse_config() so it returns {} on an empty file instead of raising.",
                instruction="Fix parse_config() on empty input.",
                tests_dir=EXAMPLE_TESTS,
            )
        ]

    @functools.cached_property
    def judge(self) -> SweJudge:
        return SweJudge(self.config.judge, self.config.judge_runtime)

    async def finalize(self, task: SweTask, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Runtime still live: snapshot the agent's diff + transcript for the group judge."""
        diff = await runtime.run(["sh", "-c", "cd /repo && git add -A && git diff --cached --binary"], {})
        trace.info["judge"] = {"patch": diff.stdout, "transcript": vf.render_transcript(trace)}

    @vf.group_reward(weight=1.0)
    async def panel(self, traces: list[vf.Trace], task: SweTask) -> list[float]:
        return await self.judge.score(traces, task)
