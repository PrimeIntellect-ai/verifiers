"""A conversation with a user: turn-taking as resumed runs on one live box.

`converse(agent, task, user)` runs an agent on a task with a user who answers it. The
task's prompt is the user's opening message. Each turn is one ordinary run of the agent
on one box the conversation provisions (`agent.provision`):

- the first turn is `agent.run(task, runtime=box)`;
- each later turn continues the previous turn's trace with the user's reply as a normal
  user message: `agent.run(task, runtime=box, resume=previous, note=reply)`. The harness
  relaunches on the conversation (`Harness.resume`), on the box as the last turn left
  it: a continuation on the box its origin ran on runs neither `Task.setup` nor
  `Task.restore` (`Box.hosted`).

Every turn is a durable call (`key=`: `<key>/turn-<n>`) and the box is a durable box
(`<key>/box`), so with per-call saves (`AgentConfig.save = "per_call"`) a conversation
cut off mid-turn (a crash, a drain) and run again gets its finished turns back from
their records and continues the cut-off turn from its latest save, mid-call, on the box
restored from that save. The user's replies must be durable the same way: a
`ModelUser` reply is an agent run (a durable call); a user made of host calls records
its replies through the caller's durable host calls. A reply that differs on a run
again would continue a turn against a box that moved past it.

A message for a running turn (a steer) is not a turn: it reaches the running harness
through its inbox (`vf.steer`, `Harness.SUPPORTS_INBOX`: bash and null).

The turns run with the task's scoring deferred (`Task.defer_scoring`). When the
conversation ends, the end state on the box is graded once (`grade_live`), with the
conversation on the final trace (`trace.info["conversation"]`, `Conversation`), where a
grader reads it. The conversation ends when:

- the user is done (`UserReply.done`);
- the user's reply cap is reached (`max_turns`);
- a turn stopped on a limit or a timeout (`solver_stopped`): the agent never reads a
  reply. It is still graded;
- a turn failed (`solver_failed`): the final trace is that failed trace, not graded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from verifiers.v1.agent import Agent
from verifiers.v1.configs.scoring import resolve_scoring
from verifiers.v1.grading import bind_scoring_env, grade_submission
from verifiers.v1.rollout import OnSave
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, UserMessage
from verifiers.v1.utils.artifacts import collect
from verifiers.v1.utils.decorators import invoke

__all__ = [
    "CONVERSATION_KEY",
    "Conversation",
    "ConversationTurn",
    "ModelUser",
    "ScriptedUser",
    "User",
    "UserReply",
    "converse",
    "grade_live",
]

CONVERSATION_KEY = "conversation"
"""`trace.info` key of the conversation record (`Conversation`)."""

StopReason = Literal["open", "user_done", "turn_cap", "solver_stopped", "solver_failed"]
"""Why a conversation ended (`converse`); `open` while it runs."""


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    info: dict[str, Any] = Field(default_factory=dict)
    """What the user recorded about its reply (`UserReply.info`)."""


class Conversation(BaseModel):
    """The conversation as both sides saw it: the user's opening (the task prompt), then
    the agent's reply and the user's answer, in turn. A user's closing message, which
    the agent never reads, is the last turn when the user had one."""

    turns: list[ConversationTurn] = Field(default_factory=list)
    stop: StopReason = "open"

    @property
    def opening(self) -> str:
        return self.turns[0].text if self.turns else ""

    @property
    def replies(self) -> int:
        """The user's answers the agent read: user turns after the opening, a closing
        message aside."""
        answers = [turn for turn in self.turns[1:] if turn.role == "user"]
        if self.stop == "user_done" and self.turns and self.turns[-1].role == "user":
            answers = answers[:-1]
        return len(answers)

    @property
    def last_agent_message(self) -> str:
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                return turn.text
        return ""

    @classmethod
    def of(cls, trace: Trace) -> Conversation | None:
        """The conversation recorded on `trace`, if any."""
        record = trace.info.get(CONVERSATION_KEY)
        return cls.model_validate(record) if isinstance(record, dict) else None


class UserReply(BaseModel):
    text: str = ""
    """The user's message. With `done`, a closing message the agent never reads."""
    done: bool = False
    """The user ends the conversation."""
    info: dict[str, Any] = Field(default_factory=dict)
    """What the user records about this reply on its turn (`ConversationTurn.info`)."""


class User(Protocol):
    """The user's side of a conversation: one reply to the conversation so far, whose
    last turn is the agent's latest message. `trace` is the agent's latest turn, where
    the user may record its own usage."""

    def reply(
        self, conversation: Conversation, *, trace: Trace
    ) -> Awaitable[UserReply]: ...


class ScriptedUser:
    """Answers with fixed replies in order; done after the last one."""

    def __init__(self, replies: Sequence[str | UserReply]) -> None:
        self.replies = [
            r if isinstance(r, UserReply) else UserReply(text=r) for r in replies
        ]

    async def reply(self, conversation: Conversation, *, trace: Trace) -> UserReply:
        del trace
        index = conversation.replies
        return (
            self.replies[index] if index < len(self.replies) else UserReply(done=True)
        )


USER_PERSONA = """You are role-playing a USER who asked an AI assistant for help. You are the user; the assistant does the work. Your first message in the conversation is your request.

What you want, what you know, and how you behave:

{scenario}

Rules:
- Stay in character: short, natural messages. Never act as the assistant, never write the solution, code or commands yourself.
- Tell what you know only when the assistant asks about it or its work touches it.
- When you do not know something, say so and let the assistant choose.
- When the assistant has finished what you want, or you are sure it cannot, reply with exactly {done} and nothing else."""

DONE = "###DONE###"

_GREETING = "Hello! How can I help you today?"
"""The assistant's greeting on the user's side only: a chat model's conversation starts
with a user message, and the user's request answers it."""


class ModelUser:
    """A modeled user: `agent` (any harness; the tool-less `null` harness fits) plays
    the user with `persona` as its system prompt. Each reply is one durable run of
    `agent` (`key=<key>/<n>`) on the conversation with the roles swapped; the reply
    `done` (alone) ends the conversation."""

    def __init__(
        self,
        agent: Agent,
        scenario: str,
        *,
        persona: str = USER_PERSONA,
        done: str = DONE,
        key: str = "user",
    ) -> None:
        self.agent = agent
        self.system_prompt = persona.replace("{done}", done).replace(
            "{scenario}", scenario
        )
        self.done = done
        self.key = key

    def messages(self, conversation: Conversation) -> list:
        swapped: list = [UserMessage(content=_GREETING)]
        for turn in conversation.turns:
            kind = AssistantMessage if turn.role == "user" else UserMessage
            swapped.append(kind(content=turn.text))
        return swapped

    async def reply(self, conversation: Conversation, *, trace: Trace) -> UserReply:
        del trace
        index = conversation.replies
        task = Task(
            TaskData(
                idx=index,
                system_prompt=self.system_prompt,
                prompt=self.messages(conversation),
            )
        )
        run = await self.agent.run(task, key=f"{self.key}/{index}")
        if not run.ok:
            error = run.last_error
            raise RuntimeError(
                f"the user's reply failed: {error.type}: {error.message}"
                if error is not None
                else "the user's reply failed"
            )
        text = run.last_reply.strip()
        if text == self.done or text.endswith(self.done):
            return UserReply(text=text.removesuffix(self.done).strip(), done=True)
        return UserReply(text=text)


def _record(trace: Trace, conversation: Conversation) -> None:
    trace.info[CONVERSATION_KEY] = conversation.model_dump(mode="json")


async def converse(
    agent: Agent,
    task: Task,
    user: User,
    *,
    key: str = "converse",
    max_turns: int = 8,
    tools: Mapping[str, Any] | None = None,
    on_save: OnSave | None = None,
    on_turn: Callable[[Trace], None] | None = None,
    on_score: Callable[[Trace, Runtime], Awaitable[None]] | None = None,
    grade: bool = True,
) -> Trace:
    """Run `agent` on `task` with `user` answering it, one resumed run per turn on one
    box, and grade the end state once (see the module docs). Returns the final turn's
    trace: graded, with the conversation at `trace.info["conversation"]`; a failed
    turn's trace, ungraded. `max_turns` caps the user's replies. `on_turn` sees each
    turn's trace when it lands (a stage that saves or annotates them); `on_score` runs
    after task scoring, as `Agent.run`'s does."""
    if max_turns < 0:
        raise ValueError("max_turns must be at least 0")
    opening = task.data.prompt_text
    if not opening.strip():
        raise ValueError("a conversation opens on the task's prompt: it has none")
    conversation = Conversation(turns=[ConversationTurn(role="user", text=opening)])
    turns = task.defer_scoring()
    async with agent.provision(task, key=f"{key}/box") as box:
        trace = await agent.run(
            turns, runtime=box, tools=tools, on_save=on_save, key=f"{key}/turn-0"
        )
        number = 0
        while True:
            if on_turn is not None:
                on_turn(trace)
            if not trace.ok:
                conversation.stop = "solver_failed"
                break
            conversation.turns.append(
                ConversationTurn(role="assistant", text=trace.last_reply)
            )
            if trace.stop_condition not in (None, "agent_completed"):
                # A limit, a stop or a timeout ended the turn: the agent would never
                # read a reply.
                conversation.stop = "solver_stopped"
                break
            if conversation.replies >= max_turns:
                conversation.stop = "turn_cap"
                break
            answer = await user.reply(conversation, trace=trace)
            if answer.done:
                if answer.text.strip():
                    conversation.turns.append(
                        ConversationTurn(
                            role="user", text=answer.text, info=answer.info
                        )
                    )
                conversation.stop = "user_done"
                break
            if not answer.text.strip():
                raise ValueError("the user replied with an empty message")
            conversation.turns.append(
                ConversationTurn(role="user", text=answer.text, info=answer.info)
            )
            # The next turn's trace carries this record from its origin (`info`).
            _record(trace, conversation)
            number += 1
            trace = await agent.run(
                turns,
                runtime=box,
                tools=tools,
                on_save=on_save,
                resume=trace,
                note=answer.text,
                key=f"{key}/turn-{number}",
            )
        _record(trace, conversation)
        if grade and trace.ok:
            scoring = resolve_scoring(task.scoring, agent.config.scoring)
            await grade_live(task, trace, box, scoring=scoring, on_score=on_score)
    return trace


async def grade_live(
    task: Task,
    trace: Trace,
    runtime: Runtime,
    *,
    scoring=None,
    on_score: Callable[[Trace, Runtime], Awaitable[None]] | None = None,
) -> None:
    """Grade the end state in a live `runtime` (borrowed: it stays up) onto `trace`, as
    a rollout's close grades it: with isolated scoring the task's finalize and the
    deliverable's collection (when the trace holds none), then a fresh verifier
    (`grade_submission`); with shared scoring the task's grader in `runtime`. Rewards
    and metrics land on `trace`; a scoring failure raises."""
    policy = scoring if scoring is not None else task.scoring
    if policy.mode == "isolated":
        if not trace.state.artifacts:
            await invoke(task.finalize, {"trace": trace, "runtime": runtime})
            if not trace.state.artifacts:
                trace.state.artifacts = await collect(
                    runtime, task.data.artifacts, max_bytes=task.data.artifact_max_bytes
                )
        await grade_submission(task, trace, runtime, scoring=policy, on_score=on_score)
        return
    grader = task.grading_task(runtime, scoring=policy)
    bound = bind_scoring_env(runtime, policy)
    await invoke(grader.stage_verifier, {"trace": trace, "runtime": bound})
    timeout = task.data.timeout.scoring
    async with asyncio.timeout(timeout):
        await grader.score(trace, bound)
        if on_score is not None:
            await on_score(trace, bound)
