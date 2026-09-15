"""A small flywheel as a flow: a builder writes a grading folder while solvers
attempt the request; a reviewer judges the folder and can send the builder back;
the attempts are judged against the accepted folder.

    python -m verifiers.v1.flow examples.flow.flywheel:flywheel rows.jsonl runs/first @ config.toml
"""

from __future__ import annotations

import asyncio

import verifiers.v1 as vf
from verifiers.v1.flow import Ctx, FlowConfig, agent, command, fn


class FlywheelConfig(FlowConfig):
    builder: vf.AgentConfig = vf.AgentConfig(harness={"id": "bash"})
    solver: vf.AgentConfig = vf.AgentConfig(harness={"id": "bash"})
    reviewer: vf.AgentConfig = vf.AgentConfig(harness={"id": "bash"})
    attempts: int = 3


class Job(vf.TaskData):
    request: str
    folder: str = "/work"


class Say(vf.Task[Job]):
    """A task whose prompt is its data's request; the seat's reply is the work."""


def task(prompt: str, **data) -> Say:
    return Say(Job(prompt=prompt, **data))


def decision(trace: vf.Trace) -> str:
    return (
        trace.last_reply.strip().split()[-1].lower() if trace.last_reply.strip() else ""
    )


def bank(judgments: dict[int, vf.Trace]) -> dict:
    correct = sum(decision(t) == "correct" for t in judgments.values())
    return {"correct": correct, "judged": len(judgments)}


async def flywheel(ctx: Ctx, row: dict) -> dict:
    request = row["request"]
    solves = asyncio.ensure_future(  # the solvers run while the folder is built
        ctx.spread(
            "solve",
            [
                agent("solver", task(request, request=request))
                for _ in range(ctx.config.attempts)
            ],
        )
    )
    async with ctx.runtime("builder") as box:
        notes = ""
        for visit in range(3):
            with ctx.scope(f"visit{visit}"):
                await ctx.step(
                    "build",
                    agent(
                        "builder",
                        task(
                            f"Write grading checks for: {request}\n{notes}",
                            request=request,
                        ),
                        runtime=box,
                    ),
                )
                lint = await ctx.step(
                    "lint", command(["sh", "-c", "ls /work/grading"], runtime=box)
                )
                review = await ctx.step(
                    "review",
                    agent(
                        "reviewer",
                        task(
                            f"Review the grading folder for: {request}. End with accept or revise.",
                            request=request,
                        ),
                        runtime=box,
                    ),
                )
            if lint.exit_code == 0 and decision(review) == "accept":
                break
            notes = f"The reviewer said:\n{review.last_reply}"
        else:
            return {"rejected": "the folder never passed review"}
        attempts = await solves
        judgments = await ctx.spread(
            "judge",
            [
                agent(
                    "reviewer",
                    task(
                        f"Judge this attempt against the grading folder. End with correct or incorrect.\n\n{t.last_reply}",
                        request=request,
                    ),
                    runtime=box,
                )
                for t in attempts.values()
            ],
        )
    return await ctx.step("bank", fn(bank, judgments))


__all__ = ["FlywheelConfig", "flywheel"]
