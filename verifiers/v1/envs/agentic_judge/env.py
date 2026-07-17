"""agentic-judge: a solver plays the task, a code-executing judge verifies it in a sandbox.

The sandboxed tier of agent-as-judge (`--env.id agentic-judge` over any taskset; the
bare tier is `--env.id judge`). Same verdict spec plugin, same recording on the
solver's trace — what changes is the judge's world: the judge is ALWAYS played in its
own sandbox, never on the host. Its verdict task mirrors the solver task's world
(same image/workdir/resources — a FRESH box, in its original state), and the graded
transcript is uploaded into it (`/tmp/transcript.md` rendered, `/tmp/transcript.json`
the raw trace record) so the judge can reconstruct and test the agent's work
empirically. The solver's runtime is gone by judge time — a judge that must inspect
the solver's live box is a custom env (`provision()` + borrowed `runtime=`), not this
wrapper.

The judge seat is unpinned by default, so it runs the taskset's default harness —
the same program the solver defaults to — but always in a container: a judge whose
harness resolves to the subprocess runtime is refused at construction (pin
`--env.judge.harness.runtime.type docker` or `prime`). A tool-less judge
(`direct`/`null`) belongs on `--env.id judge` and is refused here.
"""

import json

import verifiers.v1 as vf
from verifiers.v1.envs.judge.env import JudgeEnv, JudgeEnvConfig
from verifiers.v1.harness import Harness

TRANSCRIPT_MD = "/tmp/transcript.md"
TRANSCRIPT_JSON = "/tmp/transcript.json"


def _sandbox_note(solver: vf.TaskData) -> str:
    """What an agentic judge must know about its box before it starts verifying."""
    world = (
        f"a fresh instance of the same environment the graded agent worked in "
        f"(image {solver.image}), in its ORIGINAL state — the agent's edits are "
        "NOT applied; reconstruct them from the transcript to verify"
        if solver.image is not None
        else "your own — the graded agent worked elsewhere"
    )
    return (
        f"\n\n## Your workspace\nYour sandbox is {world}. The agent's full transcript "
        f"is uploaded at {TRANSCRIPT_MD} (rendered) and {TRANSCRIPT_JSON} (the raw "
        "trace record)."
    )


def _noted(prompt: str | vf.Messages, note: str) -> str | vf.Messages:
    if isinstance(prompt, str):
        return prompt + note
    return [*prompt, vf.UserMessage(content=note.strip())]


class JudgeTask(vf.Task):
    """An agentic judge's verdict task: the solver task's world mirrored onto the
    minted row, with the graded transcript uploaded before the judge starts.
    `NEEDS_CONTAINER` keeps `Agent.run`'s per-task backstop aligned with the role's
    declared container need — judge-written code never runs on the host."""

    NEEDS_CONTAINER = True

    def __init__(self, data: vf.TaskData, files: dict[str, bytes]) -> None:
        super().__init__(data)
        self._files = files

    async def setup(self, trace, runtime):
        for path, content in self._files.items():
            await runtime.write(path, content)


class AgenticJudgeEnvConfig(JudgeEnvConfig):
    judge: vf.AgentConfig = vf.AgentConfig(trainable=False)
    """The judge seat. Unpinned, it runs the taskset's default harness (the same
    program the solver defaults to) in a mirror of the solver's world — its runtime
    must be a container: `--env.judge.harness.runtime.type docker|prime`."""


class AgenticJudgeEnv(JudgeEnv, vf.Environment[AgenticJudgeEnvConfig]):
    def _check_judge_harness(self, harness: Harness) -> None:
        if not harness.EXECUTES_CODE:
            raise ValueError(
                "agentic-judge plays a code-executing judge in its own sandbox, but "
                f"harness {harness.config.id!r} is a tool-less chat loop — use "
                "--env.id judge for a bare judge."
            )

    def roles(self):
        # The judge investigates with real execution — never on the host: its role
        # needs a container (fail-fast at construction; a judge resolving to the
        # subprocess runtime is refused, with the flag to set). The harness-kind
        # check runs first.
        self._check_judge_harness(self._judge_harness())
        return {
            "solver": vf.Role(self.config.solver),
            "judge": vf.Role(self.config.judge, mcp=False, container=True),
        }

    async def rollout(self, task, agents):
        solution = await agents["solver"].run(task)
        prompt = self._spec.render(task.data, solution)
        # The judge gets the solver task's world: a fresh box of the same image
        # (original state — the solver's edits live only in the solver's box), the
        # transcript uploaded, the prompt told where things stand.
        judge_task = JudgeTask(
            vf.TaskData(
                idx=task.data.idx,
                prompt=_noted(prompt, _sandbox_note(task.data)),
                image=task.data.image,
                workdir=task.data.workdir,
                resources=task.data.resources.model_copy(),
                sources=(solution.id,),
                relation="judges",
            ),
            files={
                TRANSCRIPT_MD: solution.transcript.encode(),
                TRANSCRIPT_JSON: json.dumps(solution.to_record()).encode(),
            },
        )
        verdict = await agents["judge"].run(judge_task)
        return [solution, verdict]
