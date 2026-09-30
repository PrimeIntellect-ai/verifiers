"""Executable delivery examples using real harness processes and scripted inference.

Put `rlm`, `claude-agent-acp`, `claude`, and `node` on PATH, then run:
    VF_DELIVERY_BEHAVIORS=1 uv run pytest tests/v1/test_delivery_behaviors.py -v -s

VF_DELIVERY_OUTPUT optionally selects a directory for timeline.json and normal
Verifiers traces.jsonl files (one run per case). No model API key is required.
`idle` sends A after the opening turn finishes, then B during A's resumed work.
`idle-burst` submits A and B together after the opening turn finishes.
Other cases submit A and B during the same held model/tool operation. `queue`
means an ordinary user message deferred until the current turn yields.
"""

import asyncio
import json
import os
import shutil
import socket
from pathlib import Path

import pytest
import uvicorn
from delivery_behaviors import LABELS, Scenario, labels
from starlette.applications import Starlette
from starlette.routing import Route

import verifiers.v1 as vf
from verifiers.v1.acp import ACPHarness
from verifiers.v1.episode import Episode, EvalRunInfo
from verifiers.v1.harnesses.claude_code.harness import (
    ClaudeCodeHarness,
    ClaudeCodeHarnessConfig,
)
from verifiers.v1.harnesses.rlm.harness import RLMHarness, RLMHarnessConfig

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("VF_DELIVERY_BEHAVIORS") != "1",
        reason="opt-in real-harness behavioral examples",
    ),
]


@pytest.mark.parametrize("kind", ["rlm", "claude-code"])
@pytest.mark.parametrize("state", ["idle-burst", "idle", "model", "tool", "wait"])
@pytest.mark.parametrize(
    "modes",
    [("steer", "steer"), ("steer", "queue"), ("queue", "steer"), ("queue", "queue")],
    ids=["steer-steer", "steer-message", "message-steer", "message-message"],
)
async def test_delivery_behavior(kind, state, modes, tmp_path, monkeypatch):
    if state == "wait" and kind != "rlm":
        pytest.skip("wait is a nano-rlm native tool")
    commands = ["rlm"] if kind == "rlm" else ["node", "claude-agent-acp", "claude"]
    bins = {command: shutil.which(command) for command in commands}
    if not all(bins.values()):
        pytest.skip(f"install these harness executables first: {commands}")
    scenario = Scenario(kind, state, hold_call=2 if state.startswith("idle") else 1)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    scenario.endpoint = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(
        uvicorn.Config(
            Starlette(
                routes=[
                    Route("/v1/chat/completions", scenario.respond, methods=["POST"]),
                    Route("/v1/messages", scenario.respond, methods=["POST"]),
                    Route("/barrier", scenario.barrier),
                ]
            ),
            log_level="error",
        )
    )
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    monkeypatch.setenv("DELIVERY_SCRIPTED_KEY", "local-scripted")
    base = RLMHarness if kind == "rlm" else ClaudeCodeHarness

    class LocalHarness(base):
        NEEDS_CONTAINER = False

        async def setup(self, runtime):
            await ACPHarness.setup(self, runtime)

        async def prepare_acp(self, *args, **kwargs):
            config = await super().prepare_acp(*args, **kwargs)
            if kind == "rlm":
                config.command = [bins["rlm"], "--acp"]
            else:
                config.command = [bins["node"], bins["claude-agent-acp"]]
                config.env["CLAUDE_CODE_EXECUTABLE"] = bins["claude"]
            return config

    config = (
        RLMHarnessConfig(id="rlm", builtin_skills=[], max_depth=0, compaction=False)
        if kind == "rlm"
        else ClaudeCodeHarnessConfig(id="claude-code")
    )
    agent = vf.make_agent(
        vf.AgentConfig(
            harness=config,
            model="scripted-delivery" if kind == "rlm" else "claude-sonnet-4-6",
            client={
                "type": "eval",
                "base_url": scenario.endpoint + "/v1",
                "api_key_var": "DELIVERY_SCRIPTED_KEY",
            },
            runtime={"type": "subprocess"},
            max_turns=8,
            timeout={"rollout": 45},
        )
    )
    agent.harness = LocalHarness(config)
    name = f"delivery-{kind}-{state}-{'-'.join(modes)}"
    task = vf.Task(
        vf.TaskData(
            idx=0,
            name=name,
            prompt="Report which incoming messages you have received.",
            system_prompt="Behavioral demonstration with scripted inference. Each reply states which messages are visible in that model request.",
        )
    )
    sends = []
    opening = None
    try:
        async with asyncio.timeout(50), agent, agent.interaction(task) as interaction:
            real_step = interaction._run.step

            async def step(messages=None):
                scenario.record(
                    "turn_started",
                    messages=labels(
                        [m.model_dump() for m in messages] if messages else []
                    ),
                )
                result = await real_step(messages)
                scenario.record("turn_yielded")
                return result

            interaction._run.step = step

            async def send(index):
                scenario.record("submitted", message=LABELS[index], mode=modes[index])
                receipt = await interaction.send(
                    LABELS[index], mode=modes[index], message_id=LABELS[index]
                )
                scenario.record("receipt", message=LABELS[index], **receipt)
                return receipt

            opening = asyncio.create_task(interaction.turn())
            if state == "idle-burst":
                await opening
                scenario.record("agent_idle")
                sends.extend(asyncio.create_task(send(i)) for i in range(2))
                await scenario.model_entered.wait()
            elif state == "idle":
                await opening
                scenario.record("agent_idle")
                sends.append(asyncio.create_task(send(0)))
                await scenario.model_entered.wait()
                sends.append(asyncio.create_task(send(1)))
            else:
                if state == "model":
                    await scenario.model_entered.wait()
                elif state == "tool":
                    await scenario.tool_entered.wait()
                else:
                    await scenario.response_sent.wait()
                    # Let the harness enter its native wait after reading the response.
                    await asyncio.sleep(0.2)
                sends.extend(asyncio.create_task(send(i)) for i in range(2))
            # Keep the operation held while both ACP requests traverse the real processes.
            await asyncio.sleep(0.25)
            scenario.record("release_operation")
            scenario.model_release.set()
            scenario.tool_release.set()
            await opening
            receipts = await asyncio.gather(*sends)
            assert all(r["outcome"] == "injected" for r in receipts)
            assert set(scenario.calls[-1]["messages"]) == set(LABELS)
            for label, mode in zip(LABELS, modes, strict=True):
                submitted = next(
                    e["index"]
                    for e in scenario.events
                    if e["event"] == "submitted" and e["message"] == label
                )
                seen = next(
                    e["index"]
                    for e in scenario.events
                    if e["event"] == "model_request" and label in e["messages"]
                )
                assert seen > submitted
                if mode == "steer" and not state.startswith("idle"):
                    assert not any(
                        e["event"] == "turn_yielded"
                        for e in scenario.events[submitted:seen]
                    )
                if (
                    mode == "queue"
                    and state != "idle-burst"
                    and not (state == "idle" and label == LABELS[0])
                ):
                    assert any(
                        e["event"] == "turn_yielded"
                        for e in scenario.events[submitted:seen]
                    )
            if (
                kind == "rlm"
                and not state.startswith("idle")
                and modes == ("steer", "steer")
            ):
                assert scenario.calls[1]["messages"] == list(LABELS)
            if not state.startswith("idle") and modes == ("queue", "queue"):
                assert next(c for c in scenario.calls if c["messages"])[
                    "messages"
                ] == list(LABELS)
            interaction.trace.info.update(
                scenario=name,
                scripted_inference=True,
                executables=bins,
                events=scenario.events,
                model_inputs=scenario.calls,
            )
        assert interaction.trace.ok, interaction.trace.errors
    finally:
        scenario.model_release.set()
        scenario.tool_release.set()
        for pending in [opening, *sends]:
            if pending is not None and not pending.done():
                pending.cancel()
        await asyncio.gather(
            *(p for p in [opening, *sends] if p is not None), return_exceptions=True
        )
        server.should_exit = True
        await serving
        output = Path(os.environ.get("VF_DELIVERY_OUTPUT", tmp_path)) / name
        output.mkdir(parents=True, exist_ok=True)
        (output / "timeline.json").write_text(
            json.dumps(
                {
                    "scenario": name,
                    "events": scenario.events,
                    "model_inputs": scenario.calls,
                },
                indent=2,
            )
        )
        if "interaction" in locals():
            trace = interaction.trace
            episode = Episode(
                task=trace.task,
                traces=[trace],
                ok=trace.ok,
                run=EvalRunInfo(id=name, name=name),
            )
            (output / "traces.jsonl").write_text(episode.model_dump_json() + "\n")
        print(
            "\n" + name + ": " + " → ".join(str(c["messages"]) for c in scenario.calls)
        )
