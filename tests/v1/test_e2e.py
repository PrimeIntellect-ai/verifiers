"""End-to-end eval runs on trivial tasksets — each scores reward 1.0, with no errors.

Every task is one greedy rollout (`temperature=0`, set in `run_v1`) on a single task with
turn/timeout caps. The matrix axes are the runtimes a rollout places things in: the **harness**
runtime (`harness_runtime`), the **user** simulator's runtime (`user_runtime`), the **tool** server's
runtime (`tool_runtime`), and a **shared** tool server's own runtime (`shared_tool_runtime`) — each
spanning subprocess/docker/prime (modal excluded), with docker/prime marked `slow`/`prime` so the
default run stays on subprocess.

`test_user` and `test_tool` fan a server's own runtime against the harness runtime (the full
reachability matrix); `test_single_turn`/`test_agentic` fan the harness against the harness runtime.
`test_shared_tool_isolation` runs two concurrent rollouts against one SHARED writable tool server
(fork off/on) on each `shared_tool_runtime` (harness on the same runtime), asserting each keeps its
own state.
"""

import pytest


@pytest.mark.e2e
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
    assert not trace.is_truncated
    assert trace.num_turns == 1
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_user(
    run_v1, harness_runtime, user_runtime, skip_if_unexposable, tmp_path
):
    """Multi-turn, driven by a (container-safe) `vf.User` simulator, across the full matrix of the
    user's runtime (`user_runtime`: colocated in the harness's runtime, or its own runtime) x the
    harness `runtime`. Either way the framework drives the user and must reach it from wherever the
    harness runs."""
    (trace,) = await run_v1(
        "echo-user-sim-v1",
        harness="default",
        harness_overrides={"runtime": {"type": harness_runtime}},
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
    run_v1, run_v1_server, harness_runtime, tool_runtime, skip_if_unexposable, tmp_path
):
    """A `vf.Toolset` (an echo tool) across the full matrix of its runtime (`tool_runtime`:
    colocated in the harness's runtime, shared once per eval, or its own runtime) x the harness
    `runtime`. The tool stamps its output with a token the prompt never reveals, so reward 1.0
    proves the tool was reachable from wherever the harness runs and actually ran.

    The `shared` case runs through the env-server worker pool (`run_v1_server`, prime-rl's path,
    where serving the shared tool once is the server's job) — a regression guard for the env server
    running rollouts without entering its serving context (a shared server would otherwise be rebuilt
    per rollout or error with "shared server was launched with a task"). Other runtimes run
    in-process."""
    run = run_v1_server if tool_runtime.get("shared") else run_v1
    (trace,) = await run(
        "echo-tool-v1",
        harness="default",
        harness_overrides={"runtime": {"type": harness_runtime}},
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
    run_v1, harness_runtime, tool_runtime, skip_if_unexposable, tmp_path
):
    """The shared-state round-trip: a `@vf.tool` increments the typed `trace.state` each call (synced
    over the interception server) and the `@reward` reads it back — reward 1.0 proves tool writes
    reach the host's `trace.state`. Fanned across the tool's placement (`tool_runtime`) x the harness
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
        harness_overrides={"runtime": {"type": harness_runtime}},
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
@pytest.mark.parametrize(
    "fork",
    [pytest.param(False, id="fork-off"), pytest.param(True, id="fork-on")],
)
async def test_shared_tool_isolation(
    run_v1_server, shared_tool_runtime, fork, monkeypatch, skip_if_unexposable, tmp_path
):
    """A SHARED, writable tool server keeps each rollout's state isolated across concurrent rollouts,
    fanned across the shared server's own runtime (`shared_tool_runtime`) x fork. `scratchpad-v1` gives
    each task a unique word and rewards 1.0 iff the rollout reads back its OWN word — so two concurrent
    rollouts (two distinct words on the ONE shared instance) both scoring 1.0 proves no cross-rollout
    corruption. fork=off isolates via the per-rollout `self.state` channel; fork=on bypasses
    `self.state` (writes a process-global slot) and relies on the forked-child process isolation — so
    each mode exercises a different path.

    The harness runs on the same runtime as the shared server, so the server can reach the rollout's
    `/state` + `/task` channel (localhost when both local; the interception pool's tunnel when both
    remote — a remote shared server needs a remote harness). Runs through the env-server pool
    (`run_v1_server`, prime-rl's path), where serving the one shared instance is the server's job."""
    if fork:
        # write the word to a process-global slot instead of self.state, so ONLY fork can isolate it
        monkeypatch.setenv("SCRATCHPAD_ISOLATE", "0")
    traces = await run_v1_server(
        "scratchpad-v1",
        harness="default",
        # harness on the same runtime as the shared server
        harness_overrides={"runtime": {"type": shared_tool_runtime}},
        output_dir=tmp_path,
        num_tasks=2,  # two distinct words, run concurrently against the one shared server
        n=1,
        max_turns=4,
        taskset_overrides={
            "tools": {
                "shared": True,
                "fork": fork,
                "runtime": {"type": shared_tool_runtime},
            }
        },
    )
    assert len(traces) == 2
    for trace in traces:
        skip_if_unexposable(trace)
        assert trace.errors == []
        assert not trace.is_truncated
        assert trace.num_turns >= 2  # tool call + answer
        assert (
            trace.reward == 1.0
        )  # read back its OWN word — no cross-rollout corruption


@pytest.mark.e2e
async def test_tool_response_image(run_v1, tmp_path):
    """MCP image content from a tool result survives into the v1 trace (needs a vision model)."""
    (trace,) = await run_v1(
        "tool-response-image-v1",
        harness="default",
        harness_overrides={"runtime": {"type": "subprocess"}},
        model="qwen/qwen3-vl-8b-instruct",
        output_dir=tmp_path,
        max_turns=4,
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_agentic(run_v1, agentic_harness, harness_runtime, tmp_path):
    """Agentic: write a phrase to a file with the harness's bash tool, checked in the runtime."""
    (trace,) = await run_v1(
        "echo-agentic-v1",
        harness=agentic_harness,
        harness_overrides={
            "runtime": {"type": harness_runtime},
            # the default harness gates its bash tool behind this flag; the agent CLIs don't take it
            **({"enable_bash": True} if agentic_harness == "default" else {}),
        },
        output_dir=tmp_path,
        max_turns=10,
    )
    assert trace.errors == []
    assert trace.num_turns >= 1  # ran a command, then finished
    assert trace.reward == 1.0
