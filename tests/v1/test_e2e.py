"""End-to-end eval runs on trivial tasksets — each scores reward 1.0, with no errors.

Every task is one greedy rollout (`temperature=0`, set in `run_v1`) on a single task with
turn/timeout caps. The matrix axes are the three runtimes a rollout places things in: the
**agent** (harness) runtime (`agent_runtime`), the **user** simulator's runtime (`user_runtime`),
and the **tool** server's runtime (`tool_runtime`) — each spanning subprocess/docker/prime (modal
excluded), with docker/prime marked `slow`/`prime` so the default run stays on subprocess.

`test_user` and `test_tool` fan a server's own runtime against the agent runtime (the full
reachability matrix); `test_single_turn`/`test_agentic` fan the harness against the agent runtime.
"""

import pytest


@pytest.mark.e2e
async def test_single_turn(run_v1, harness, agent_runtime, tmp_path):
    """Single-turn (echo a short phrase back)."""
    (trace,) = await run_v1(
        "echo-v1",
        harness=harness,
        agent_runtime=agent_runtime,
        output_dir=tmp_path,
        max_turns=2,
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns == 1
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_user(run_v1, agent_runtime, user_runtime, skip_if_unexposable, tmp_path):
    """Multi-turn, driven by a (container-safe) `vf.User` simulator, across the full matrix of the
    user's runtime (`user_runtime`: colocated in the agent's runtime, or its own runtime) x the
    agent `runtime`. Either way the framework drives the user and must reach it from wherever the
    agent runs."""
    (trace,) = await run_v1(
        "echo-user-sim-v1",
        harness="default",
        agent_runtime=agent_runtime,
        output_dir=tmp_path,
        max_turns=6,
        taskset_overrides={"user": user_runtime},
    )
    skip_if_unexposable(trace)
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # genuinely multi-turn
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_tool(
    run_v1, run_v1_server, agent_runtime, tool_runtime, skip_if_unexposable, tmp_path
):
    """A `vf.Toolset` (an echo tool) across the full matrix of its runtime (`tool_runtime`:
    colocated in the agent's runtime, shared once per eval, or its own runtime) x the agent
    `runtime`. The tool stamps its output with a token the prompt never reveals, so reward 1.0
    proves the tool was reachable from wherever the agent runs and actually ran.

    The `shared` case runs through the env-server worker pool (`run_v1_server`, prime-rl's path,
    where serving the shared tool once is the server's job) — a regression guard for the env server
    running rollouts without entering its serving context (a shared server would otherwise be rebuilt
    per rollout or error with "shared server was launched with a task"). Other runtimes run
    in-process."""
    run = run_v1_server if tool_runtime.get("shared") else run_v1
    (trace,) = await run(
        "echo-tool-v1",
        harness="default",
        agent_runtime=agent_runtime,
        output_dir=tmp_path,
        max_turns=6,
        taskset_overrides={"tools": tool_runtime},
    )
    skip_if_unexposable(trace)
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_tool_state(
    run_v1, agent_runtime, tool_runtime, skip_if_unexposable, tmp_path
):
    """The shared-state round-trip: a `@vf.tool` increments the typed `trace.state` each call (synced
    over the interception server) and the `@reward` reads it back — reward 1.0 proves tool writes
    reach the host's `trace.state`. Fanned across the tool's placement (`tool_runtime`) x the agent
    `runtime`, so the state channel is exercised colocated and own-runtime. `shared` is skipped: a
    shared server is eval-level (one instance for the whole eval), so per-rollout state isn't wired
    to it."""
    if tool_runtime.get("shared"):
        pytest.skip(
            "shared tool servers are eval-level — per-rollout state isn't wired to them"
        )
    (trace,) = await run_v1(
        "counter-tool-v1",
        harness="default",
        agent_runtime=agent_runtime,
        output_dir=tmp_path,
        max_turns=8,
        taskset_overrides={"tools": tool_runtime},
    )
    skip_if_unexposable(trace)
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # at least two tool calls accumulated
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_tool_state_via_env_server(run_v1_server, tmp_path):
    """The state round-trip through the env-server worker pool (prime-rl's path): the tool's
    `self.state` writes reach the worker's `trace.state` over the pooled interception server, the
    reward reads it, and the state-derived reward survives the wire even though `trace.state` itself
    is transient (never serialized)."""
    (trace,) = await run_v1_server(
        "counter-tool-v1",
        harness="default",
        agent_runtime="subprocess",
        output_dir=tmp_path,
        max_turns=8,
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_tool_response_image(run_v1, tmp_path):
    """MCP image content from a tool result survives into the v1 trace (needs a vision model)."""
    (trace,) = await run_v1(
        "tool-response-image-v1",
        harness="default",
        agent_runtime="subprocess",
        model="qwen/qwen3-vl-8b-instruct",
        output_dir=tmp_path,
        max_turns=4,
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_agentic(run_v1, agentic_harness, agent_runtime, tmp_path):
    """Agentic: write a phrase to a file with the harness's bash tool, checked in the runtime."""
    (trace,) = await run_v1(
        "echo-agentic-v1",
        harness=agentic_harness,
        enable_bash=agentic_harness == "default",
        agent_runtime=agent_runtime,
        output_dir=tmp_path,
        max_turns=10,
    )
    assert trace.errors == []
    assert trace.num_turns >= 1  # ran a command, then finished
    assert trace.reward == 1.0
