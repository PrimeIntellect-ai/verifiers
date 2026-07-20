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

# harness runtime x user placement: colocated in both local runtimes, each own-runtime
# across the opposite boundary, modal rows local-only. No prime rows: a user sim in a
# prime sandbox needs prime port exposure (unreachable from the host here).
USER_PLACEMENTS = [
    _pair("subprocess", "colocated", "harness-in-subprocess-with-user-colocated"),
    _pair("docker", "colocated", "harness-in-docker-with-user-colocated"),
    _pair("subprocess", "docker", "harness-in-subprocess-with-user-in-docker"),
    _pair("docker", "subprocess", "harness-in-docker-with-user-in-subprocess"),
    _pair("modal", "colocated", "harness-in-modal-with-user-colocated"),
    _pair("subprocess", "modal", "harness-in-subprocess-with-user-in-modal"),
]

# harness runtime x tool placement: as USER_PLACEMENTS plus the two-container case
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
    assert trace.errors == []
    assert trace.num_turns == 1
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
@pytest.mark.parametrize("harness_runtime,user_runtime", USER_PLACEMENTS, indirect=True)
async def test_user(run_v1, harness_runtime, user_runtime, tmp_path):
    """Multi-turn, driven by a (container-safe) `vf.User` simulator, across the user's
    placement (`user_runtime`: colocated in the harness's runtime, or its own runtime) x
    the harness `runtime`. Either way the framework drives the user and must reach it
    from wherever the harness runs."""
    (trace,) = await run_v1(
        "echo-user-sim-v1",
        harness="null",
        harness_overrides={"runtime": {"type": harness_runtime}},
        output_dir=tmp_path,
        max_turns=6,
        taskset_overrides={"task": {"user": user_runtime}},
    )
    assert trace.errors == []
    assert trace.num_turns >= 2  # genuinely multi-turn
    assert trace.reward == 1.0


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
    assert trace.errors == []
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
    assert trace.errors == []
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
        assert trace.errors == []
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
    assert trace.errors == []
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
    assert trace.errors == []
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
    assert trace.errors == []
    assert trace.num_turns >= 1  # ran a command, then finished
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_multi_agent_env(run_v1, tmp_path):
    """An `Environment` subclass shipped with its taskset (duet-v1): two roles run the
    task, `score()` episodes a sibling-dependent metric, and one eval rollout lands one
    episode carrying two role-stamped traces."""
    import json

    traces = await run_v1(
        "duet-v1",
        harness=None,  # both duet seats pin their own harness
        output_dir=tmp_path,
        max_turns=2,
    )
    assert len(traces) == 2  # one env-rollout, one trace per role
    assert sorted(t.role for t in traces) == ["a", "b"]
    (b,) = [t for t in traces if t.role == "b"]
    assert b.trainable is False
    for trace in traces:
        assert trace.errors == []
        assert trace.reward == 1.0  # each seat's own task reward
        assert trace.metrics["duet"] == 1.0  # the sibling-dependent signal
    # On disk: one episode line carrying both traces, role-stamped.
    (line,) = (tmp_path / "traces.jsonl").read_text().splitlines()
    row = json.loads(line)
    assert row["env"] == "duet-v1"
    assert [t["role"] for t in row["traces"]] == ["a", "b"]
    assert [t.get("trainable") for t in row["traces"]] == [True, False]


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
    assert len(traces) == 2  # one env-rollout, two attempts
    assert all(t.role == "agent" and t.errors == [] for t in traces)
    assert any(t.metrics["best"] == 1.0 for t in traces)
    assert all(t.metrics["pass_at_n"] == 1.0 for t in traces)  # echo always passes


@pytest.mark.e2e
async def test_env_id_agentic_judge(run_v1, tmp_path):
    """The agentic judge over the echo taskset (needs docker): the judge lands in
    its own box with the graded transcript uploaded, investigates with real
    execution, and its parsed verdict lands on the solver's trace under the spec's
    reward key. Wiring, not taste: the judge followed the default `score` spec's
    output contract — the grade itself is the model's call."""
    traces = await run_v1(
        "echo-v1",
        harness=None,  # seats pin their own harness; there is no run-level one
        env={
            "id": "agentic-judge",
            "solver": {"harness": {"id": "null"}},
            "judge": {"harness": {"runtime": {"type": "docker"}}},
        },
        output_dir=tmp_path,
        max_turns=10,
        rollout_timeout=600,
    )
    assert sorted(t.role for t in traces) == ["judge", "solver"]
    (solver,) = [t for t in traces if t.role == "solver"]
    (judge,) = [t for t in traces if t.role == "judge"]
    assert solver.errors == [] and judge.errors == []
    assert judge.trainable is False
    assert solver.rewards["echoed"] == 1.0  # the task's own reward still runs
    assert "SCORE:" in (judge.last_reply or "")
    assert 0.0 <= solver.rewards["judge"] <= 1.0


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
    assert sorted(t.role for t in traces) == ["a", "b"]
    for trace in traces:
        assert trace.errors == []
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
    assert source.errors == []
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
        assert replayed.errors == []
        # The typed rebuild ran (not the base-Task fallback): the trace-only reward re-ran
        # and recomputed the same value.
        assert replayed.rewards.keys() == source.rewards.keys()
        assert replayed.reward == pytest.approx(source.reward)
    # The wire task keeps its taskset-specific fields in the replay's own output.
    raw = (tmp_path / "replay2" / "traces.jsonl").read_text()
    assert '"answer"' in raw
