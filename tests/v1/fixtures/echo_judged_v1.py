"""echo-judged: echo, scored by an agentic judge instead of string matching.

The judged twin of `echo-v1`: the model echoes a phrase, but the reward comes from a
`vf.JudgeSpec` agent run — a default-harness agent provisioned into the rollout's
runtime during SCORING, which reads the materialized transcript, decides whether the
phrase was echoed, and writes a typed verdict. The `@reward` just maps the verdict to a
number. The judge samples from the policy model ("policy"), so the e2e run needs no
second endpoint. A fixture taskset for the v1 e2e suite (id `echo-judged-v1`).
"""

from pydantic import BaseModel

import verifiers.v1 as vf

SYSTEM = "Repeat the user's message back to them exactly, with no extra words."


class EchoVerdict(BaseModel):
    echoed: bool
    """Whether the agent's reply contained the phrase."""
    evidence: str
    """The reply line that contains the phrase (or why it's missing)."""


class EchoTask(vf.Task):
    answer: str
    """The phrase the model should echo back."""


class EchoJudgedConfig(vf.TasksetConfig):
    phrases: list[str] = ["hello world"]


class EchoJudgedTaskset(vf.Taskset[EchoTask, EchoJudgedConfig]):
    def load_tasks(self) -> list[EchoTask]:
        return [
            EchoTask(
                idx=i,
                prompt=(
                    "Do not call tools or execute code. Reply immediately and include "
                    f"this exact phrase in your final response: {phrase}"
                ),
                system_prompt=SYSTEM,
                answer=phrase,
            )
            for i, phrase in enumerate(self.config.phrases)
        ]

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    async def judges(self, task: EchoTask) -> list[vf.JudgeSpec]:
        return [
            vf.JudgeSpec(
                name="echoed",
                prompt=(
                    f"An agent was asked to echo the exact phrase {task.answer!r}. "
                    "Read the transcript and decide whether the agent's reply contains "
                    "the phrase (ignore case, spacing, and punctuation)."
                ),
                verdict=EchoVerdict,
                budget=vf.AgentBudget(max_turns=8),
            )
        ]

    @vf.reward
    async def echoed(self, verdicts) -> float:
        return float(verdicts["echoed"].echoed)


__all__ = ["EchoJudgedTaskset"]
