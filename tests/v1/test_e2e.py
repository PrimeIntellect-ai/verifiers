"""End-to-end eval runs on trivial tasksets — each scores reward 1.0, with no errors.

Every task is one greedy rollout (`temperature=0`, set in `run_v1`) on a single task with
turn/timeout caps. The matrix axes are the runtimes a rollout places things in: the **harness**
runtime (`harness_runtime`), the **user** simulator's runtime (`user_runtime`), and the **tool**
server's runtime (`tool_runtime`) — each spanning subprocess/docker/prime/modal. Every matrix value
carries a pytest mark, so subsets select with `-m` (see `conftest.py`).

`test_user` and `test_tool` fan a server's own runtime against the harness runtime (the full
reachability matrix); `test_single_turn`/`test_agentic` fan the harness against the harness runtime.
`test_shared_tool_isolation` runs two concurrent rollouts against one SHARED writable tool server
across the full `harness_runtime` x (shared server's own) `tool_runtime` matrix — incl. mixed-locality
combos — asserting each keeps its own `self.state`.
"""

import pytest


@pytest.mark.e2e
async def test_single_turn(run_v1, harness, harness_runtime, tmp_path):
    """Single-turn (echo a short phrase back)."""
    if harness == "codex":
        pytest.skip("codex is a coding agent, not reliable on a no-op echo chat task")
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
        taskset_overrides={"user": user_runtime},
    )
    assert trace.errors == []
    assert trace.num_turns >= 2  # genuinely multi-turn
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_tool(run_v1, run_v1_server, harness_runtime, tool_runtime, tmp_path):
    """A `vf.Toolset` (an echo tool) across the full matrix of its runtime (`tool_runtime`:
    colocated in the harness's runtime, shared once per eval, or its own runtime) x the harness
    `runtime`. The tool stamps its output with a token the prompt never reveals, so reward 1.0
    proves the tool was reachable from wherever the harness runs and actually ran.

    The `shared` case runs through the env-server worker pool (`run_v1_server`, prime-rl's path,
    where serving the shared tool once is the server's job) — a regression guard for the env server
    running rollouts without entering its serving context (a shared server would otherwise be rebuilt
    per rollout or error with "shared server was launched with a task"). Other runtimes run
    in-process."""
    # Reaching a tool server in its own prime sandbox needs prime port exposure, whose URL isn't
    # reachable from the host here (region=us doesn't help). Skip until it is.
    if tool_runtime.get("runtime", {}).get("type") == "prime":
        pytest.skip(
            "tool server in a prime sandbox needs prime port exposure (unreachable from host here)"
        )
    run = run_v1_server if tool_runtime.get("shared") else run_v1
    (trace,) = await run(
        "echo-tool-v1",
        harness="null",
        harness_overrides={"runtime": {"type": harness_runtime}},
        output_dir=tmp_path,
        max_turns=6,
        taskset_overrides={"tools": tool_runtime},
    )
    assert trace.errors == []
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_tool_state(run_v1, harness_runtime, tool_runtime, tmp_path):
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
        taskset_overrides={"tools": tool_runtime},
    )
    assert trace.errors == []
    assert trace.num_turns >= 2  # at least two tool calls accumulated
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_shared_tool_isolation(
    run_v1_server, harness_runtime, tool_runtime, tmp_path
):
    """A SHARED, writable tool server keeps each rollout's `self.state` isolated across concurrent
    rollouts, over the FULL `harness_runtime` x (shared server's own) `tool_runtime` matrix — including
    the mixed-locality combos (e.g. a local harness with a remote shared tool), which is exactly where
    the rollout's `/state` channel must be bridged to the tool's runtime. `scratchpad-v1` gives each
    task a unique word and rewards 1.0 iff the rollout reads back its OWN word, so two concurrent
    rollouts (two distinct words on the ONE shared instance) both scoring 1.0 proves the per-rollout
    `self.state` channel keeps them isolated with no cross-rollout corruption.

    Placement is fixed to `shared`, so only the own-runtime cases of `tool_runtime` apply (the
    colocated/shared params have no distinct runtime to fan — skipped). Runs through the env-server
    pool (`run_v1_server`, prime-rl's path), where serving the one shared instance is the server's
    job."""
    # colocated / shared placement has no distinct own runtime to fan here
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
        num_tasks=2,  # two distinct words, run concurrently against the one shared server
        n=1,
        max_turns=4,
        taskset_overrides={
            "tools": {
                "shared": True,
                **tool_runtime,  # the shared tool's own runtime ({"runtime": {"type": ...}})
            }
        },
    )
    assert len(traces) == 2
    for trace in traces:
        assert trace.errors == []
        assert trace.num_turns >= 2  # tool call + answer
        # read back its OWN word — no cross-rollout corruption
        assert trace.reward == 1.0


@pytest.mark.e2e
async def test_tool_response_image(run_v1, tmp_path):
    """MCP image content from a tool result survives into the v1 trace (needs a vision model)."""
    (trace,) = await run_v1(
        "tool-response-image-v1",
        harness="null",
        harness_overrides={"runtime": {"type": "subprocess"}},
        model="qwen/qwen3-vl-8b-instruct",
        output_dir=tmp_path,
        max_turns=4,
    )
    assert trace.errors == []
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0


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
