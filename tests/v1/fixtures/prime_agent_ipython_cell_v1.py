"""Prime Agent IPython cell invocation shape through native ACP."""

import json

import verifiers.v1 as vf

SENTINEL = "prime-agent-acp-cell-ok"
CELL = f"print('{SENTINEL}')"


def _segment_info(segment: vf.Segment) -> dict:
    return {
        "last_reply": segment.last_reply,
        "tool_calls": [
            {"name": call.name, "arguments": call.arguments}
            for message in segment.messages
            if isinstance(message, vf.AssistantMessage)
            for call in message.tool_calls or []
        ],
        "tool_outputs": [
            str(message.content)
            for message in segment.messages
            if isinstance(message, vf.ToolMessage)
        ],
        "terminated": segment.terminated,
    }


def has_ipython_cell_call(trace: vf.Trace) -> bool:
    """Whether the model ran the requested cell and the kernel actually executed it.

    Both halves matter. A fabricated tool call plus the right reply text must not
    score, so the cell prints a sentinel and the tool RESULT has to contain it:
    only a real kernel execution produces that output.
    """
    segments = trace.info.get("prime_agent_segments", [])
    if len(segments) != 1:
        return False
    segment = segments[0]
    if segment["terminated"] or segment["last_reply"].strip() != "DONE":
        return False
    invoked = False
    for call in segment["tool_calls"]:
        try:
            arguments = json.loads(call["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        # Exact equality, so an extra reconstruction statement cannot sneak in.
        if call.get("name") == "ipython" and arguments == {"code": CELL}:
            invoked = True
            break
    if not invoked:
        return False
    return any(SENTINEL in output for output in segment["tool_outputs"])


class PrimeAgentIpythonCellTask(vf.Task):
    @vf.reward(weight=1.0)
    async def ipython_cell(self, trace: vf.Trace) -> float:
        return float(has_ipython_cell_call(trace))


class PrimeAgentIpythonCellEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            segment = await interaction.turn(
                "Use IPython to execute exactly this cell:\n\n"
                f"{CELL}\n\n"
                "Then reply with exactly DONE."
            )
            interaction.trace.info["prime_agent_segments"] = [_segment_info(segment)]


class PrimeAgentIpythonCellTaskset(
    vf.Taskset[PrimeAgentIpythonCellTask, vf.TasksetConfig]
):
    def load(self) -> list[PrimeAgentIpythonCellTask]:
        return [
            PrimeAgentIpythonCellTask(
                vf.TaskData(
                    idx=0,
                    prompt=None,
                    system_prompt=(
                        "Follow each instruction exactly. Use IPython when requested."
                    ),
                )
            )
        ]


__all__ = [
    "PrimeAgentIpythonCellEnv",
    "PrimeAgentIpythonCellTaskset",
    "has_ipython_cell_call",
]
