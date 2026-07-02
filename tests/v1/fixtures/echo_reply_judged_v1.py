"""echo-reply-judged: echo, graded by a single-call reply-verdict judge.

The single-call twin of `echo-judged-v1`: the judge is a `null`-harness `vf.JudgeSpec`
— no tools, so its evidence (the phrase and the model's reply) rides in its prompt,
built from the trace via the injectable `judges(task, trace)` hook, and its final reply
is the verdict JSON. The classic one-call LLM judge expressed as an agent run. A
fixture taskset for the v1 suite (id `echo-reply-judged-v1`)."""

from pydantic import BaseModel

import verifiers.v1 as vf

SYSTEM = "Repeat the user's message back to them exactly, with no extra words."


class EchoVerdict(BaseModel):
    echoed: bool
    """Whether the agent's reply contained the phrase."""


class EchoTask(vf.Task):
    answer: str
    """The phrase the model should echo back."""


class EchoReplyJudgedConfig(vf.TasksetConfig):
    phrases: list[str] = ["hello world"]


class EchoReplyJudgedTaskset(vf.Taskset[EchoTask, EchoReplyJudgedConfig]):
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

    async def judges(self, task: EchoTask, trace: vf.Trace) -> list[vf.JudgeSpec]:
        return [
            vf.JudgeSpec(
                name="echoed",
                prompt=(
                    f"An agent was asked to echo the exact phrase {task.answer!r}. "
                    f"Its reply was: {trace.last_reply!r}. Decide whether the reply "
                    "contains the phrase (ignore case, spacing, and punctuation)."
                ),
                verdict=EchoVerdict,
                harness={"id": "null"},
            )
        ]

    @vf.reward
    async def echoed(self, verdicts) -> float:
        return float(verdicts["echoed"].echoed)


__all__ = ["EchoReplyJudgedTaskset"]
