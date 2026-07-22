"""End-to-end v1 eval smoke tests.

Placement coverage is pairwise (see tests/v1/conftest.py): each list below names the
combinations a test runs — every axis value at least once plus the cross-boundary pairs
with distinct networking — instead of fanning the full cross product. prime/modal rows
are local-only (their marks are excluded in CI)."""

import pytest

_m = pytest.mark


def _pair(a: str, b: str, id: str, *extra_marks):
    marks = [getattr(_m, a.replace("-", "_")), getattr(_m, b.replace("-", "_"))]
    return pytest.param(a, b, marks=[*marks, *extra_marks], id=id)


# harness x harness runtime: every harness once, both local runtimes hit, one remote row
# per provider. codex/claude-code are excluded here (unreliable on a no-op echo chat
# task) — test_agentic covers them.
CHAT_PLACEMENTS = [
    _pair("null", "subprocess", "null-harness-in-subprocess"),
    _pair("bash", "docker", "bash-harness-in-docker"),
    _pair("rlm", "subprocess", "rlm-harness-in-subprocess"),
    _pair("kimi-code", "docker", "kimi-code-harness-in-docker"),
    _pair("bash", "prime", "bash-harness-in-prime"),
    _pair("bash", "modal", "bash-harness-in-modal"),
]

# harness x harness runtime for the shell task: every coding agent once (null is a chat
# loop with no shell), both local runtimes hit, one remote row per provider.
AGENTIC_PLACEMENTS = [
    _pair("bash", "subprocess", "bash-harness-in-subprocess"),
    _pair("rlm", "docker", "rlm-harness-in-docker"),
    _pair("kimi-code", "subprocess", "kimi-code-harness-in-subprocess"),
    _pair("codex", "docker", "codex-harness-in-docker"),
    _pair("claude-code", "subprocess", "claude-code-harness-in-subprocess"),
    _pair("bash", "prime", "bash-harness-in-prime"),
    _pair("bash", "modal", "bash-harness-in-modal"),
]

# The scripted user runs in the eval process itself (no placement axis); the harness
# runtime is the exchange's only axis.
USER_RUNTIMES = [
    pytest.param("subprocess", marks=[_m.subprocess], id="harness-in-subprocess"),
    pytest.param("docker", marks=[_m.docker], id="harness-in-docker"),
    pytest.param("prime", marks=[_m.prime], id="harness-in-prime"),
    pytest.param("modal", marks=[_m.modal], id="harness-in-modal"),
]

# harness runtime x tool placement: every axis value once plus the two-container case
# (harness and tool in separate docker boxes) and a prime-colocated row (a tool in its
# OWN prime sandbox needs port exposure; colocated rides the harness's box).
TOOL_PLACEMENTS = [
    _pair("subprocess", "colocated", "harness-in-subprocess-with-tool-colocated"),
    _pair("docker", "colocated", "harness-in-docker-with-tool-colocated"),
    _pair("subprocess", "docker", "harness-in-subprocess-with-tool-in-docker"),
    _pair("docker", "subprocess", "harness-in-docker-with-tool-in-subprocess"),
    _pair("docker", "docker", "harness-in-docker-with-tool-in-docker"),
    _pair("prime", "colocated", "harness-in-prime-with-tool-colocated"),
    _pair("modal", "colocated", "harness-in-modal-with-tool-colocated"),
    _pair("subprocess", "modal", "harness-in-subprocess-with-tool-in-modal"),
]

# The state channel rides the same reachability as TOOL_PLACEMENTS; cover each axis
# value once rather than re-running the whole list.
TOOL_STATE_PLACEMENTS = [
    _pair("subprocess", "colocated", "harness-in-subprocess-with-tool-colocated"),
    _pair("docker", "subprocess", "harness-in-docker-with-tool-in-subprocess"),
    _pair("subprocess", "docker", "harness-in-subprocess-with-tool-in-docker"),
    _pair("modal", "colocated", "harness-in-modal-with-tool-colocated"),
]

# Shared servers always run in their own runtime (colocation is per-rollout, shared is
# eval-level): same-runtime pairs plus one cross-boundary row.
SHARED_TOOL_PLACEMENTS = [
    _pair("subprocess", "subprocess", "harness-in-subprocess-with-tool-in-subprocess"),
    _pair("docker", "docker", "harness-in-docker-with-tool-in-docker"),
    _pair("subprocess", "docker", "harness-in-subprocess-with-tool-in-docker"),
    _pair("modal", "modal", "harness-in-modal-with-tool-in-modal"),
]


@pytest.mark.e2e
@pytest.mark.parametrize("harness,harness_runtime", CHAT_PLACEMENTS, indirect=True)
async def test_single_turn(run_v1, harness, harness_runtime, tmp_path):
    """Single-turn (echo a short phrase back)."""
    (trace,) = await run_v1(
        "echo-v1",
        harness=harness,
        harness_overrides={"runtime": {"type": harness_runtime}},
        output_dir=tmp_path,
        max_turns=2,
    )
    assert trace.ok
    assert trace.num_turns == 1
    assert trace.stop_condition == "agent_completed"
    assert trace.reward == 1.0
    # The seat's resolved identity rides the trace (policy metadata for trainers).
    assert trace.agent is not None and trace.agent.sampling.temperature == 0
    # Every sampled turn has one per-call record, linked to its assistant node.
    sampled = [i for i, n in enumerate(trace.nodes) if n.sampled]
    assert [c.node for c in trace.calls if c.error is None] == sampled
    for call in trace.calls:
        assert call.model and call.sampling is not None
        assert call.time.duration > 0


@pytest.mark.e2e
@pytest.mark.parametrize("harness_runtime", USER_RUNTIMES, indirect=True)
async def test_user(run_v1, harness_runtime, tmp_path):
    """Multi-turn, driven by a scripted user — an interaction loop in the env's
    `run()` — across the harness runtime axis. The task is prompt-less, so one
    run covers the whole exchange shape: the caller opens (the user speaks first),
    each later turn resumes the harness onto the conversation, and leaving the loop
    ends the exchange (`user_closed`). The user runs in the eval process itself, so
    there is no placement axis."""
    (trace,) = await run_v1(
        "echo-user-sim-v1",
        harness="null",
        harness_overrides={"runtime": {"type": harness_runtime}},
        output_dir=tmp_path,
        max_turns=6,
    )
    assert trace.ok
    assert trace.num_turns >= 2  # genuinely multi-turn
    assert trace.stop_condition == "user_closed"  # leaving the interaction ended it
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_interaction(live_ctx):
    """Drive an agent turn-by-turn through `agent.interaction()` — the caller IS
    the run's user. Runs on the tool-less `null` harness: nothing but the exchange
    itself, yet a real rollout — trace, scoring, and the `user_closed` stop all
    apply."""
    import verifiers.v1 as vf
    from verifiers.v1.harnesses.null import NullHarnessConfig

    agent = vf.make_agent(
        vf.AgentConfig(
            harness=NullHarnessConfig(id="null"),
            model=live_ctx.model,
            sampling=live_ctx.sampling,
        ),
        client=live_ctx.client,
    )
    task = vf.Task(
        vf.TaskData(
            idx=0,
            prompt=None,  # the interaction's caller opens the conversation
            system_prompt="Repeat the user's message back exactly, no extra words.",
        )
    )
    async with agent.interaction(task) as interaction:
        first = await interaction.turn("hello world")
        assert isinstance(first, vf.Segment)
        assert not first.terminated
        assert [message.role for message in first.messages] == ["assistant"]
        assert "hello world" in first.last_reply.lower()
        second = await interaction.turn("goodbye world")
        assert not second.terminated
        assert [message.role for message in second.messages] == ["assistant"]
        assert "goodbye world" in second.last_reply.lower()
    trace = interaction.trace
    assert trace is not None and trace.errors == []
    assert trace.stop_condition == "user_closed"  # closing the interaction ended it
    assert trace.num_turns == 2


@pytest.mark.e2e
@pytest.mark.parametrize("harness_runtime,tool_runtime", TOOL_PLACEMENTS, indirect=True)
async def test_tool(run_v1, harness_runtime, tool_runtime, tmp_path):
    """A `vf.Toolset` (an echo tool) across its placement (`tool_runtime`: colocated in
    the harness's runtime, or its own runtime) x the harness `runtime`. The tool stamps
    its output with a token the prompt never reveals, so reward 1.0 proves the tool was
    reachable from wherever the harness runs and actually ran. Eval-wide SHARED servers
    are a different scope (`Taskset.tools`) with their own env-server-path coverage:
    `test_shared_tool_isolation`."""
    (trace,) = await run_v1(
        "echo-tool-v1",
        harness="null",
        harness_overrides={"runtime": {"type": harness_runtime}},
        output_dir=tmp_path,
        max_turns=6,
        taskset_overrides={"task": {"tools": tool_runtime}},
    )
    assert trace.ok
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0
    # The interception server captured the advertised tools onto the trace (for tool-use SFT):
    # the null harness offered the task's MCP tool as `echo_back`, schema included.
    assert trace.tools is not None
    (echo_tool,) = [t for t in trace.tools if t.name == "echo_back"]
    assert "message" in echo_tool.parameters.get("properties", {})


@pytest.mark.e2e
@pytest.mark.parametrize(
    "harness_runtime,tool_runtime", TOOL_STATE_PLACEMENTS, indirect=True
)
async def test_tool_state(run_v1, harness_runtime, tool_runtime, tmp_path):
    """The shared-state round-trip: a `@vf.tool` increments the typed `trace.state` each call (synced
    over the interception server) and the `@reward` reads it back — reward 1.0 proves tool writes
    reach the host's `trace.state`, exercised colocated and own-runtime (a SHARED server's
    per-rollout state channel is covered by `test_shared_tool_isolation`)."""
    (trace,) = await run_v1(
        "counter-tool-v1",
        harness="null",
        harness_overrides={"runtime": {"type": harness_runtime}},
        output_dir=tmp_path,
        max_turns=8,
        taskset_overrides={"task": {"tools": tool_runtime}},
    )
    assert trace.ok
    assert trace.num_turns >= 2  # at least two tool calls accumulated
    assert trace.reward == 1.0


@pytest.mark.e2e
@pytest.mark.parametrize(
    "harness_runtime,tool_runtime", SHARED_TOOL_PLACEMENTS, indirect=True
)
async def test_shared_tool_isolation(
    run_v1_server, harness_runtime, tool_runtime, tmp_path
):
    """A shared writable tool isolates state across concurrent rollouts and runtimes."""
    traces = await run_v1_server(
        "scratchpad-v1",
        harness="null",
        harness_overrides={"runtime": {"type": harness_runtime}},
        output_dir=tmp_path,
        num_tasks=2,
        n=1,
        max_turns=4,
        taskset_overrides={"tools": tool_runtime},
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace.ok
        assert trace.num_turns >= 2  # tool call + answer
        assert trace.reward == 1.0


@pytest.mark.e2e
async def test_tool_response_image(run_v1, tmp_path):
    """MCP image content from a tool result survives into the v1 trace (needs a vision model)."""
    (trace,) = await run_v1(
        "tool-response-image-v1",
        harness="null",
        harness_overrides={"runtime": {"type": "subprocess"}},
        model="openai/gpt-5.6-luna",
        reasoning_effort="none",
        output_dir=tmp_path,
        max_turns=4,
    )
    assert trace.ok
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_rubric_judge(run_v1, tmp_path):
    """A config-plugged rubric judge scores the rollout on top of the taskset's own reward.

    The single criterion is trivially satisfiable ("answer yes"), so any live judge model
    scores it 1.0 — the test asserts the plumbing (config narrowing -> judge call -> reward +
    per-criterion metric on the trace), not judge quality."""
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(
        "[[criteria]]\n"
        'name = "always_yes"\n'
        'text = "Always satisfied — answer yes regardless of the response."\n'
    )
    (trace,) = await run_v1(
        "echo-v1",
        harness="null",
        harness_overrides={"runtime": {"type": "subprocess"}},
        output_dir=tmp_path,
        taskset_overrides={"task": {"judges": [{"id": "rubric", "path": str(rubric)}]}},
        max_turns=2,
    )
    assert trace.ok
    assert trace.rewards["rubric"] > 0  # the judge's verdict landed in the reward
    assert trace.metrics["rubric/always_yes"] == 1.0
    assert trace.info["judge"]  # the call was recorded onto the trace


@pytest.mark.e2e
@pytest.mark.parametrize("harness,harness_runtime", AGENTIC_PLACEMENTS, indirect=True)
async def test_agentic(run_v1, harness, harness_runtime, tmp_path):
    """Agentic: write a phrase to a file with the agent's shell, checked in the runtime."""
    (trace,) = await run_v1(
        "echo-agentic-v1",
        harness=harness,
        harness_overrides={"runtime": {"type": harness_runtime}},
        output_dir=tmp_path,
        max_turns=10,
    )
    assert trace.ok
    assert trace.num_turns >= 1  # ran a command, then finished
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_multi_agent_env(run_v1, tmp_path):
    """An `Env` subclass shipped with its taskset (duet-v1): two roles run the
    task, `score()` episodes a sibling-dependent metric, and one episode carries
    two role-stamped traces."""
    import json

    traces = await run_v1(
        "duet-v1",
        harness=None,  # both duet seats pin their own harness
        output_dir=tmp_path,
        max_turns=2,
    )
    assert len(traces) == 2  # one episode, one trace per role
    assert sorted(t.agent_name for t in traces) == ["a", "b"]
    (b,) = [t for t in traces if t.agent_name == "b"]
    assert b.trainable is False
    for trace in traces:
        assert trace.ok
        assert trace.reward == 1.0  # each seat's own task reward
        assert trace.metrics["duet"] == 1.0  # the sibling-dependent signal
    # On disk: one episode line carrying both traces, each self-stamped on its
    # agent info (completion order — the gathered seats land in either order).
    (line,) = (tmp_path / "traces.jsonl").read_text().splitlines()
    row = json.loads(line)
    assert row["env"] == "duet-v1"
    by_name = {t["agent"]["name"]: t for t in row["traces"]}
    assert set(by_name) == {"a", "b"}
    assert by_name["a"]["agent"]["trainable"] is True
    assert by_name["b"]["agent"]["trainable"] is False


@pytest.mark.e2e
async def test_env_id_best_of_n(run_v1, tmp_path):
    """`--env.id` pairs a bundled env with an arbitrary taskset: best-of-n over the
    plain echo taskset — n solver attempts in one episode, sibling-scored."""
    traces = await run_v1(
        "echo-v1",
        harness=None,  # a multi-agent env refuses the run-level harness
        env={"id": "best-of-n", "n": 2, "agent": {"harness": {"id": "null"}}},
        output_dir=tmp_path,
        max_turns=2,
    )
    assert len(traces) == 2  # one episode, two attempts
    assert all(t.agent_name == "agent" and t.ok for t in traces)
    assert any(t.metrics["best"] == 1.0 for t in traces)
    assert all(t.metrics["pass_at_n"] == 1.0 for t in traces)  # echo always passes


@pytest.mark.e2e
async def test_env_id_agentic_judge(run_v1, tmp_path):
    """The agentic judge over the echo taskset (needs docker): the judge lands in
    its own box with the graded transcript uploaded, investigates with real
    execution, and its parsed verdict lands on the solver's trace under the spec's
    reward key. Wiring, not taste: the judge followed the verdict-file contract's
    output contract — the grade itself is the model's call."""
    traces = await run_v1(
        "echo-v1",
        harness=None,  # seats pin their own harness; there is no run-level one
        env={
            "id": "agentic-judge",
            "solver": {"harness": {"id": "null"}},
            # The judge reads the transcript and reasons before it writes the
            # verdict file; the shared 2048-token run cap truncates it mid-audit.
            "judge": {
                "harness": {"runtime": {"type": "docker"}},
                "max_output_tokens": 8192,
            },
        },
        output_dir=tmp_path,
        max_turns=10,
        rollout_timeout=600,
    )
    assert sorted(t.agent_name for t in traces) == ["judge", "solver"]
    (solver,) = [t for t in traces if t.agent_name == "solver"]
    (judge,) = [t for t in traces if t.agent_name == "judge"]
    assert solver.ok and judge.ok
    assert judge.trainable is False
    assert solver.rewards["echoed"] == 1.0  # the task's own reward still runs
    assert isinstance(judge.info.get("verdict"), dict)  # scraped off the box
    assert 0.0 <= solver.rewards["judge"] <= 1.0


@pytest.mark.e2e
async def test_env_id_user_sim(run_v1, tmp_path):
    """The user-sim env over the echo taskset: a modeled user (null harness) opens
    the conversation from the task's prompt-as-scenario; the assistant's trace is
    judged by the task's own reward; both sides land agent-stamped on one episode."""
    traces = await run_v1(
        "echo-v1",
        harness=None,  # multi-agent: each seat pins its own
        env={"id": "user-sim", "assistant": {"harness": {"id": "null"}}},
        output_dir=tmp_path,
        max_turns=6,
        rollout_timeout=300,
    )
    assert sorted(t.agent_name for t in traces) == ["assistant", "user"]
    (assistant,) = [t for t in traces if t.agent_name == "assistant"]
    (user,) = [t for t in traces if t.agent_name == "user"]
    assert assistant.ok and user.ok
    assert user.trainable is False
    assert user.num_turns >= 1  # the modeled user actually spoke
    assert assistant.metrics["user_turns"] >= 1
    # `mask_prompt`: the scenario is hidden from the assistant's harness (the run's
    # visible data) while the task's own rewards still scored the real row. The
    # masked view is what persists (provenance is the row's idx); both sides land
    # as ONE durable episode.
    assert assistant.task.data.prompt is None
    assert "echoed" in assistant.rewards
    from verifiers.v1.cli.output import read_episodes
    from verifiers.v1.trace import WireTrace

    (record,) = read_episodes(tmp_path, WireTrace)
    assert {t.agent_name for t in record.traces} == {"assistant", "user"}
    assert record.id  # both traces are persisted under one durable episode identity


@pytest.mark.e2e
async def test_env_id_user_sim_with_tools(run_v1, tmp_path):
    """THE tau-bench shape — a tool-using assistant composed with a modeled user.
    The assistant's MCP tool loop runs entirely inside a harness segment and the
    user exchange advances between segments, so the two can never race or amputate
    each other (the failure mode of injecting user turns at the model boundary).
    Reward 1.0 proves the tool actually ran mid-conversation: its token never
    appears in any prompt."""
    traces = await run_v1(
        "echo-tool-v1",
        harness=None,  # multi-agent: each seat pins its own
        env={"id": "user-sim", "assistant": {"harness": {"id": "null"}}},
        output_dir=tmp_path,
        max_turns=8,
        # Reasoning models can spend thousands of tokens on a turn; the default
        # 2048 truncates the reply mid-relay, cutting the stamp out of the text.
        max_tokens=8192,
        rollout_timeout=300,
    )
    (assistant,) = [t for t in traces if t.agent_name == "assistant"]
    (user,) = [t for t in traces if t.agent_name == "user"]
    assert assistant.ok and user.ok
    assert assistant.task.data.prompt is None  # the scenario stayed off the wire
    assert user.num_turns >= 1  # the modeled user actually drove the exchange
    assert assistant.rewards["echoed"] == 1.0  # the tool ran, mid-conversation
    # The tool was advertised to the masked chat exactly as to any run.
    assert assistant.tools is not None
    assert any(tool.name == "echo_back" for tool in assistant.tools)


@pytest.mark.e2e
async def test_kuhn_poker_self_play(run_v1, tmp_path):
    """The turn-coupled proof env: one Kuhn poker hand, both seats live interactions
    of the run's own model (self-play), refereed host-side, paid out zero-sum."""
    traces = await run_v1(
        "kuhn-poker-v1",
        harness=None,  # both seats pin the null harness themselves
        output_dir=tmp_path,
        max_turns=8,
        # The Q decision (the one mixed-strategy spot in Kuhn) can cost a reasoning
        # model thousands of tokens; the default 2048 truncates to an empty reply AND
        # exhausts the rollout budget, so the invalid-move retry is never served.
        max_tokens=8192,
        rollout_timeout=300,
    )
    assert sorted(t.agent_name for t in traces) == ["player0", "player1"]
    payoffs = {t.agent_name: t.rewards["payoff"] for t in traces}
    assert payoffs["player0"] + payoffs["player1"] == 0  # zero-sum
    assert abs(payoffs["player0"]) in (1.0, 2.0)
    for trace in traces:
        assert trace.ok
        assert trace.info["kuhn"]["seat"] in (0, 1)
    # A played-out hand has both seats speaking. A forfeit (the model never produced
    # a legal move) still pays out zero-sum, but the hand dies mid-exchange, so only
    # the seat that acted is guaranteed a turn — a hand where NOBODY spoke means the
    # exchange machinery never ran at all.
    if traces[0].info["kuhn"]["forfeited"] is None:
        assert all(t.num_turns >= 1 for t in traces)
    else:
        assert any(t.num_turns >= 1 for t in traces)


@pytest.mark.e2e
async def test_multi_agent_env_server(run_v1_server, tmp_path):
    """The same env through the env-server pool: the worker rebuilds the role-typed
    config from wire data, and the multi-trace episode rides the serve protocol."""
    traces = await run_v1_server(
        "duet-v1",
        harness=None,  # both duet seats pin their own harness
        output_dir=tmp_path,
        max_turns=2,
    )
    assert len(traces) == 2
    assert sorted(t.agent_name for t in traces) == ["a", "b"]
    for trace in traces:
        assert trace.ok
        assert trace.metrics["duet"] == 1.0


@pytest.mark.e2e
async def test_replay_round_trip(run_v1, tmp_path):
    """eval -> replay -> replay-the-replay. Offline re-scoring must preserve the saved
    task's wire form: replay reads traces as `Trace[WireTaskData, ...]`, so its own output
    dumps through that schema — the taskset-specific fields (reverse-text's `answer`) ride
    `model_extra` and must survive into the replay's `traces.jsonl`, or the next replay's
    typed rebuild fails and the trace-only `@reward` silently stops running (the
    wire-narrowing regression). Trace-only rewards are deterministic given the transcript,
    so all three generations must agree."""
    import tomllib
    from pathlib import Path

    from verifiers.v1.cli.output import CONFIG_FILE
    from verifiers.v1.cli.replay import run_replay
    from verifiers.v1.configs.replay import ReplayConfig

    run_dir = tmp_path / "run"
    (source,) = await run_v1(
        "reverse-text-v1",
        harness="null",
        harness_overrides={"runtime": {"type": "subprocess"}},
        output_dir=run_dir,
        max_turns=2,
    )
    assert source.ok
    assert "lcs" in source.rewards

    async def replay(source_dir: Path, out: Path):
        # The CLI's layering, minus the argv plumbing: the saved run's config is the base
        # (`ReplayConfig` ignores its eval-only keys), the source's output_dir is dropped.
        data = tomllib.loads((source_dir / CONFIG_FILE).read_text())
        data.pop("output_dir", None)
        config = ReplayConfig(**{**data, "rich": False})
        (trace,) = await run_replay(config, source_dir, out)
        return trace

    first = await replay(run_dir, tmp_path / "replay1")
    second = await replay(tmp_path / "replay1", tmp_path / "replay2")
    for replayed in (first, second):
        assert replayed.ok
        # The typed rebuild ran (not the base-Task fallback): the trace-only reward re-ran
        # and recomputed the same value.
        assert replayed.rewards.keys() == source.rewards.keys()
        assert replayed.reward == pytest.approx(source.reward)
    # The wire task keeps its taskset-specific fields in the replay's own output.
    raw = (tmp_path / "replay2" / "traces.jsonl").read_text()
    assert '"answer"' in raw
