"""solver-grader: a solver plays the task, a code-executing grader verifies it in a sandbox."""

import json

import verifiers.v1 as vf

TRANSCRIPT_MD = "/tmp/transcript.md"
TRANSCRIPT_JSON = "/tmp/transcript.json"
VERDICT_FILE = "/tmp/verdict.json"

GRADE_PROMPT = f"""You are grading another agent's attempt at a task.

## The graded task
{{task_prompt}}

## Your workspace
Your sandbox is {{world}}. The agent's full transcript is uploaded at
{TRANSCRIPT_MD} (rendered) and {TRANSCRIPT_JSON} (the raw trace record).

## Your job
Verify the attempt empirically: reconstruct what the agent did from the transcript,
re-run or re-check the work with real commands where possible, and judge whether it
actually solves the task. When you are done, write your verdict as a single JSON
object to {VERDICT_FILE}:

    {{{{"score": <integer 0-10>, "reasoning": "<one paragraph>"}}}}

The verdict file is how you are scored — do not finish without writing it."""


class GradeTask(vf.Task):
    """The grader's task: the solver task's world mirrored onto a fresh box (the
    solver's runtime is gone by grading time), transcript uploaded before the
    grader starts, verdict scraped from the box after it finishes."""

    NEEDS_CONTAINER = True

    def __init__(self, data: vf.TaskData, files: dict[str, bytes]) -> None:
        super().__init__(data)
        self._files = files

    @classmethod
    def from_trace(cls, task: vf.Task, solution: vf.Trace) -> "GradeTask":
        world = (
            f"a fresh instance of the same environment the graded agent worked in "
            f"(image {task.data.image}), in its ORIGINAL state — the agent's edits "
            "are NOT applied; reconstruct them from the transcript to verify"
            if task.data.image is not None
            else "your own — the graded agent worked elsewhere"
        )
        return cls(
            vf.TaskData(
                idx=task.data.idx,
                prompt=GRADE_PROMPT.format(
                    task_prompt=task.data.prompt_text, world=world
                ),
                image=task.data.image,
                workdir=task.data.workdir,
                resources=task.data.resources,
            ),
            files={
                TRANSCRIPT_MD: solution.transcript.encode(),
                TRANSCRIPT_JSON: json.dumps(solution.to_record()).encode(),
            },
        )

    async def setup(self, trace, runtime):
        for path, content in self._files.items():
            await runtime.write(path, content)

    async def finalize(self, trace, runtime):
        # Scrape the verdict while the box is still alive; a grader that wrote
        # none fails the episode in the env's finalize (retryable).
        try:
            raw = await runtime.read(VERDICT_FILE)
        except Exception:
            return
        trace.info["verdict"] = json.loads(raw.decode())


class SolverGraderEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    grader: vf.AgentConfig = vf.AgentConfig()
    """The grader; its runtime must be a container
    (`--env.grader.harness.runtime.type docker|prime`)."""


class SolverGraderEnv(vf.Env[SolverGraderEnvConfig]):
    def setup(self, agents):
        # The grader grades the policy; its tokens are never training data.
        agents.grader.trainable = False

    async def run(self, task, agents):
        solution = await agents.solver.run(task)
        await agents.grader.run(GradeTask.from_trace(task, solution))

    async def finalize(self, task, traces):
        # A missing or malformed verdict raises, failing the episode (retryable)
        # rather than scoring the solver 0.
        by_agent = {t.agent.name: t for t in traces}
        solution, graded = by_agent["solver"], by_agent["grader"]
        verdict = graded.info.get("verdict")
        if not isinstance(verdict, dict) or "score" not in verdict:
            raise ValueError(
                f"grader wrote no verdict to {VERDICT_FILE}; last reply: "
                f"...{(graded.last_reply or '')[-200:]!r}"
            )
        score = verdict["score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"grader verdict score is not a number: {score!r}")
        solution.record_reward("grader", min(max(float(score), 0.0), 10.0) / 10.0)
