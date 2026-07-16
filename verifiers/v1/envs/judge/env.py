"""judge: a solver plays the task, a judge agent grades the finished attempt.

The agent-as-judge wrapper (`--env.id judge` over any taskset). Unlike the plugged
judge tiers (`config.judges` — a plain model call inside the rollout; `vf.Judge` — a
call in your own reward code), the verdict here is a real agent run with its own
trace: inspectable, budgeted, role-stamped, and — because the judge is just a role —
routable to any harness. The default judge rides the in-process `direct` chat loop
(a verdict ≈ one API call, `trainable=False`); point `--env.judge.harness.id` at a
real harness and the judge can investigate with tools instead of reading a transcript.

The verdict lands as a `judge` reward on the *solver's* trace (weight `--env.weight`),
composing with the task's own rewards. The judge's trace carries no reward.
"""

import re

import verifiers.v1 as vf

RUBRIC = """You are grading another model's answer to a task.

## Task
{prompt}

## Answer
{answer}

Judge whether the answer actually solves the task: correctness first, then
completeness. Think briefly, then give your verdict as the LAST line of your reply,
in exactly this form:

SCORE: <integer from 0 to 10>"""


def parse_score(reply: str | None) -> float:
    """The last `SCORE: <n>` in the judge's reply, clamped to [0, 10] and normalized
    to [0, 1]; an unparseable verdict scores 0 (a judge that can't follow the output
    contract is not a judgement)."""
    matches = re.findall(r"SCORE:\s*(\d+)", reply or "")
    if not matches:
        return 0.0
    return min(max(int(matches[-1]), 0), 10) / 10


class JudgeParams(vf.EnvParams):
    solver: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig(
        harness=vf.HarnessConfig(id="direct"), trainable=False
    )
    rubric: str = RUBRIC
    """The judge's prompt; `{prompt}` and `{answer}` are replaced with the task's
    prompt text and the solver's final reply (plain replacement, so rubric text may
    contain braces)."""
    weight: float = 1.0
    """Weight of the `judge` reward recorded on the solver's trace."""


class JudgeEnv(vf.Environment[JudgeParams]):
    def roles(self):
        return {"solver": self.params.solver, "judge": self.params.judge}

    async def rollout(self, task, agents):
        solution = await agents["solver"].run(task)
        # Plain replacement, not str.format: the rubric (or the task text riding in
        # it) may legitimately contain braces.
        prompt = self.params.rubric.replace("{prompt}", task.data.prompt_text).replace(
            "{answer}", solution.last_reply or "<no answer>"
        )
        judge_task = vf.Task(
            vf.TaskData(
                idx=task.data.idx,
                prompt=prompt,
                sources=(solution.id,),
                relation="judges",
            )
        )
        verdict = await agents["judge"].run(judge_task)
        return [solution, verdict]

    async def score(self, task, traces):
        solution, verdict = traces
        solution.record_reward(
            "judge", parse_score(verdict.last_reply), self.params.weight
        )
