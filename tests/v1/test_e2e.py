"""End-to-end v1 eval smoke tests."""

import pytest


@pytest.mark.e2e
async def test_single_turn(run_v1, harness, harness_runtime, tmp_path):
    """Single-turn (echo a short phrase back)."""
    if harness in {"codex", "claude-code"}:
        pytest.skip("coding agents are not reliable on a no-op echo chat task")
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
    # Every sampled turn has one raw per-call record, linked to its assistant node.
    sampled = [i for i, n in enumerate(trace.nodes) if n.sampled]
    assert [c.node for c in trace.calls if c.error is None] == sampled
    for call in trace.calls:
        assert call.request is not None
        assert call.time.duration > 0
        if call.error is None:
            assert call.response is not None


@pytest.mark.e2e
async def test_user(run_v1, harness_runtime, user_runtime, tmp_path):
    """Multi-turn, driven by a (container-safe) `vf.User` simulator, across the full matrix of the
    user's runtime (`user_runtime`: colocated in the harness's runtime, or its own runtime) x the
    harness `runtime`. Either way the framework drives the user and must reach it from wherever the
    harness runs."""
    # A user sim in a prime sandbox — its own, or colocated in a prime harness — must be reached by
    # the host framework via prime port exposure, whose URL isn't reachable from the host here
    # (region=us doesn't help). Skip until it is.
    user_rt = user_runtime.get("runtime", {}).get("type")
    if user_rt == "prime" or (
        user_runtime.get("colocated") and harness_runtime == "prime"
    ):
        pytest.skip(
            "user sim in a prime sandbox needs prime port exposure (unreachable from host here)"
        )
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
async def test_tool(run_v1, harness_runtime, tool_runtime, tmp_path):
    """A `vf.Toolset` (an echo tool) across the full matrix of its runtime (`tool_runtime`:
    colocated in the harness's runtime, or its own runtime) x the harness `runtime`. The tool
    stamps its output with a token the prompt never reveals, so reward 1.0 proves the tool was
    reachable from wherever the harness runs and actually ran. Eval-wide SHARED servers are a
    different scope (`Taskset.tools`) with their own env-server-path coverage:
    `test_shared_tool_isolation`."""
    # Reaching a tool server in its own prime sandbox needs prime port exposure, whose URL isn't
    # reachable from the host here (region=us doesn't help). Skip until it is.
    if tool_runtime.get("runtime", {}).get("type") == "prime":
        pytest.skip(
            "tool server in a prime sandbox needs prime port exposure (unreachable from host here)"
        )
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
async def test_tool_state(run_v1, harness_runtime, tool_runtime, tmp_path):
    """The shared-state round-trip: a `@vf.tool` increments the typed `trace.state` each call (synced
    over the interception server) and the `@reward` reads it back — reward 1.0 proves tool writes
    reach the host's `trace.state`. Fanned across the tool's placement (`tool_runtime`) x the harness
    `runtime`, so the state channel is exercised colocated and own-runtime (a SHARED server's
    per-rollout state channel is covered by `test_shared_tool_isolation`)."""
    # Reaching a tool server in its own prime sandbox needs prime port exposure, whose URL isn't
    # reachable from the host here (region=us doesn't help). Skip until it is.
    if tool_runtime.get("runtime", {}).get("type") == "prime":
        pytest.skip(
            "tool server in a prime sandbox needs prime port exposure (unreachable from host here)"
        )
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
async def test_shared_tool_isolation(
    run_v1_server, harness_runtime, tool_runtime, tmp_path
):
    """A shared writable tool isolates state across concurrent rollouts and runtimes."""
    tool_rt = tool_runtime.get("runtime", {}).get("type")
    if tool_rt is None:
        pytest.skip("shared-isolation fans the tool's own runtime; this case has none")
    # Reaching a tool server in its own prime sandbox needs prime port exposure, whose URL isn't
    # reachable from the host here (region=us doesn't help). Skip until it is.
    if tool_rt == "prime":
        pytest.skip(
            "tool server in a prime sandbox needs prime port exposure (unreachable from host here)"
        )
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
async def test_agentic(run_v1, harness, harness_runtime, tmp_path):
    """Agentic: write a phrase to a file with the agent's shell, checked in the runtime."""
    if harness == "null":
        pytest.skip(
            "null is a chat loop with no shell — it can't do the file-write task"
        )
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
