"""tool_user_sim (v1): a calculator that must call an MCP tool AND answer a simulated user.

The minimal env that combines task tools with a user simulator. Each turn the user simulator
poses an addition problem; the model calls the `calc_add` tool to compute it, then replies with
the answer; the simulator poses the next problem, and after the last one flags `user_finished`
(the `@vf.stop` ends the rollout). The task carries no prompt — the simulator opens the
conversation.

Tools (the harness runs its own tool loop) + a user simulator is exactly the case that forks the
message graph under transparent user injection (see verifiers#1871): `num_branches` is 1 only
when the conversation stays linear, so it is the regression signal for driving the user simulator
as an explicit harness feature.
"""

import re

import verifiers.v1 as vf

from tool_user_sim_v1.servers.tools import CalcToolset
from tool_user_sim_v1.servers.user import CalcState, CalcUser

PROBLEMS = [
    (2, 3),
    (10, 20),
    (7, 8),
]  # the user poses these in order; their sums are the answers
SYSTEM = (
    "You are a calculator. For each problem the user gives you, call the `calc_add` tool to add "
    "the two numbers, then reply with only the result inside <answer></answer> tags. Always use "
    "the tool; do not add the numbers yourself."
)


class ToolUserSimTask(vf.Task):
    problems: list[list[int]]
    """The `(a, b)` pairs the user poses, one per turn; their sums are the expected answers."""


class ToolUserSimConfig(vf.TasksetConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()
    user: vf.UserConfig = vf.UserConfig()


class ToolUserSimTaskset(vf.Taskset[ToolUserSimTask, ToolUserSimConfig, CalcState]):
    @vf.stop
    async def user_finished(self, trace: vf.Trace) -> bool:
        return trace.state.user_finished

    def load_tasks(self) -> list[ToolUserSimTask]:
        return [
            ToolUserSimTask(
                idx=0,
                prompt=None,  # the user simulator opens the conversation
                system_prompt=SYSTEM,
                problems=[list(p) for p in PROBLEMS],
            )
        ]

    def tools(self, task: ToolUserSimTask) -> list[vf.Toolset]:
        return [CalcToolset(self.config.tools)]

    def user(self, task: ToolUserSimTask) -> vf.User:
        return CalcUser(self.config.user)

    @vf.reward(weight=1.0)
    async def solved(self, task: ToolUserSimTask, trace: vf.Trace) -> float:
        """Fraction of the user's problems answered correctly, read from the sampled assistant
        turns (branch-independent), so it stays correct even when the graph forks — the bug shows
        up structurally in `num_branches`, not here."""
        expected = [a + b for a, b in task.problems]
        answered = [
            int(m.group(1))
            for msg in trace.assistant_messages
            if (m := re.search(r"<answer>\s*(-?\d+)\s*</answer>", msg.content or ""))
        ]
        hits = sum(a == e for a, e in zip(answered, expected))
        return hits / len(expected)

    @vf.metric
    async def num_branches(self, trace: vf.Trace) -> float:
        """1 when the conversation graph is linear; >1 when transparent user injection forked it
        (the regression — verifiers#1871)."""
        return float(trace.num_branches)

    @vf.metric
    async def tail_branch_turns(self, trace: vf.Trace) -> float:
        """Sampled assistant turns visible on the last branch alone — what naive
        `trace.branches[-1]` scoring sees. Equals the true turn count only when the graph is
        linear; a forked graph leaves only the tail here."""
        last = trace.branches[-1] if trace.branches else None
        return float(sum(1 for n in last.nodes if n.sampled) if last else 0)


__all__ = ["ToolUserSimTaskset"]
