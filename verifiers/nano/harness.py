"""The rollout engine: turn a task into a finished transcript.

A `Harness` owns control flow only. It drives one loop: get a model response,
gather the follow-up (tool results if the model called tools, else the user
simulator's next message), and repeat until either there is no follow-up (the
natural stop) or a `@stop` condition fires. `max_turns` is itself a built-in
`@stop`, so a single-turn rollout is just `max_turns=1`.

The per-rollout collaborators (client, model, sampling, user, toolset) are passed
in a `RolloutContext` rather than stashed on the instance, so a harness holds no
mutable per-rollout state.
"""

import time
from dataclasses import dataclass
from typing import Generic

from pydantic_config import BaseConfig

from verifiers.nano.clients import Client
from verifiers.nano.decorators import discover_decorated, stop
from verifiers.nano.errors import RolloutError
from verifiers.nano.task import TaskT
from verifiers.nano.tools import Toolset
from verifiers.nano.transcript import Transcript, TranscriptT, Turn
from verifiers.nano.types import Messages, Response, SamplingConfig, Tool, UserMessage
from verifiers.nano.user import User


@dataclass(frozen=True)
class RolloutContext:
    """The collaborators a single rollout needs. Built by the Environment."""

    client: Client
    model: str
    sampling: SamplingConfig
    user: User | None = None
    toolset: Toolset | None = None


class HarnessConfig(BaseConfig):
    max_turns: int = 1
    """Maximum model turns before the rollout is forced to stop (>= 1)."""


class Harness(Generic[TaskT, TranscriptT]):
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    @stop
    async def max_turns_reached(self, transcript: TranscriptT) -> bool:
        return len(transcript.trajectory) >= self.config.max_turns

    async def rollout(
        self,
        task: TaskT,
        ctx: RolloutContext,
        *,
        transcript_cls: type[TranscriptT] = Transcript,
    ) -> TranscriptT:
        transcript = transcript_cls(task=task)
        transcript.messages = self.initial_messages(task)
        transcript.timing.generation.start = time.time()
        try:
            for hook in discover_decorated(self, "setup"):
                await hook(transcript)
            await self.run_turns(ctx, transcript)
        except RolloutError as e:
            transcript.capture_error(e)
        finally:
            for hook in discover_decorated(self, "cleanup"):
                await hook(transcript)
            transcript.is_completed = True
            transcript.timing.generation.end = time.time()
            transcript.metrics["num_turns"] = float(len(transcript.trajectory))
        return transcript

    async def run_turns(self, ctx: RolloutContext, transcript: TranscriptT) -> None:
        tools = ctx.toolset.tools if ctx.toolset is not None else None
        while True:
            response = await self.get_model_response(ctx, transcript, tools=tools)
            follow_ups = await self.follow_ups(ctx, transcript, response)
            transcript.messages.extend(follow_ups)
            if not follow_ups:
                transcript.stop("done")
                return
            if fired := await self.fire_stop(transcript):
                transcript.stop(fired)
                return

    async def follow_ups(
        self, ctx: RolloutContext, transcript: TranscriptT, response: Response
    ) -> Messages:
        """The messages to feed back next: tool results, else the user's reply,
        else nothing (which ends the rollout)."""
        if response.message.tool_calls and ctx.toolset is not None:
            return await ctx.toolset.dispatch(response.message.tool_calls)
        if ctx.user is not None:
            return await ctx.user.get_response(
                transcript.task, transcript, transcript.messages
            )
        return []

    async def get_model_response(
        self,
        ctx: RolloutContext,
        transcript: TranscriptT,
        tools: list[Tool] | None = None,
    ) -> Response:
        prompt = list(transcript.messages)
        response = await ctx.client.get_response(prompt, ctx.model, ctx.sampling, tools)
        transcript.messages.append(response.message)
        transcript.trajectory.append(Turn(prompt=prompt, response=response))
        if response.finish_reason == "length":
            transcript.is_truncated = True
        return response

    async def fire_stop(self, transcript: TranscriptT) -> str | None:
        for condition in discover_decorated(self, "stop"):
            if await condition(transcript):
                return condition.__name__
        return None

    @staticmethod
    def initial_messages(task: TaskT) -> Messages:
        return [UserMessage(content=task.instruction)]
