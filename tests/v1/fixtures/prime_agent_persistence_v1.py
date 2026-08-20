"""Two-segment Prime Agent interaction that requires one live IPython kernel."""

import re

import verifiers.v1 as vf

TOKEN = re.compile(r"\b[0-9a-f]{64}\b")
FIRST_CELL = "import secrets\n_vf_marker = secrets.token_hex(32)\nprint(_vf_marker)"
SECOND_CELL = "print(_vf_marker)"


def _segment_info(segment: vf.Segment) -> dict:
    return {
        "last_reply": segment.last_reply,
        "tool_outputs": [
            str(message.content)
            for message in segment.messages
            if isinstance(message, vf.ToolMessage)
        ],
        "tool_calls": [
            call.arguments
            for message in segment.messages
            if isinstance(message, vf.AssistantMessage)
            for call in message.tool_calls or []
        ],
        "terminated": segment.terminated,
    }


class PrimeAgentPersistenceTask(vf.Task):
    @vf.reward(weight=1.0)
    async def persisted(self, trace: vf.Trace) -> float:
        segments = trace.info.get("prime_agent_segments", [])
        if len(segments) != 2:
            return 0.0
        first, second = segments
        first_tokens = set(TOKEN.findall("\n".join(first["tool_outputs"])))
        second_tokens = set(TOKEN.findall("\n".join(second["tool_outputs"])))
        second_calls = "\n".join(second["tool_calls"])
        return float(
            bool(first_tokens & second_tokens)
            and "_vf_marker" in second_calls
            and "secrets" not in second_calls
            and "NameError" not in "\n".join(second["tool_outputs"])
            and first["last_reply"].strip() == "READY"
            and not first["terminated"]
            and not second["terminated"]
        )


class PrimeAgentPersistenceEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        # Keep the runtime alive after the rollout so cleanup is directly observable.
        async with agents.agent.provision(task) as runtime:
            async with agents.agent.interaction(task, runtime=runtime) as interaction:
                first = await interaction.turn(
                    "Use IPython to execute exactly this cell:\n\n"
                    f"{FIRST_CELL}\n\n"
                    "Then reply with exactly READY. Do not include the printed value."
                )
                segments = [first]
                if not first.terminated:
                    segments.append(
                        await interaction.turn(
                            "Use IPython to execute exactly this cell without defining, "
                            f"assigning, or reconstructing anything:\n\n{SECOND_CELL}\n\n"
                            "Then reply with exactly the printed value."
                        )
                    )
                trace = interaction.trace
                trace.info["prime_agent_segments"] = [
                    _segment_info(segment) for segment in segments
                ]
            state = f".vf-prime-agent-{trace.id}"
            cleaned = await runtime.run(["test", "!", "-e", state], {})
            trace.info["prime_agent_state_cleaned"] = cleaned.exit_code == 0


class PrimeAgentPersistenceTaskset(
    vf.Taskset[PrimeAgentPersistenceTask, vf.TasksetConfig]
):
    def load(self) -> list[PrimeAgentPersistenceTask]:
        return [
            PrimeAgentPersistenceTask(
                vf.TaskData(
                    idx=0,
                    prompt=None,
                    system_prompt=(
                        "Follow every instruction exactly. Use IPython when requested, "
                        "and never reconstruct missing kernel state from conversation text."
                    ),
                )
            )
        ]


__all__ = ["PrimeAgentPersistenceEnv", "PrimeAgentPersistenceTaskset"]
