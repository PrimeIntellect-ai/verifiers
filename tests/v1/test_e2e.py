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
async def test_llm_judge_topology(tmp_path):
    """The built-in `llm-judge` topology, live: a solver runs an echo task, the
    non-trainable judge (in-process `direct` harness — one API call) grades its final
    message against the task and its ground truth, and the verdict lands on the solver's
    trace as a deferred reward. Asserts the plumbing (seed factory -> solver episode ->
    judge episode -> declared instance scoring), not judge quality — a correct echo should
    grade well above zero."""
    from verifiers.v1.cli.eval.runner import run_topology_eval
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.topology import TopologyRunner

    config = EvalConfig(
        topology={"id": "llm-judge", "taskset": {"id": "echo-v1"}},
        num_tasks=1,
        max_turns=2,
        sampling={"max_tokens": 2048, "temperature": 0},
        timeout={"rollout": 180, "scoring": 60},
        retries={"rollout": {"max_retries": 2, "include": ["ProviderError"]}},
        rich=False,
        output_dir=tmp_path,
    )
    env = TopologyRunner(config.topology, config)
    solver, judge = await run_topology_eval(env, config)
    assert solver.errors == [] and judge.errors == []
    assert (solver.agent, judge.agent) == ("solver", "judge")
    assert judge.parents == [solver.id] and judge.trainable is False
    assert solver.rewards["echoed"] == 1.0  # the task's own reward still ran
    assert solver.metrics["judge_committed"] == 1.0
    assert solver.rewards["judge"] > 0.5  # the verdict landed on the SOLVER


@pytest.mark.e2e
@pytest.mark.subprocess
async def test_agentic_judge_topology(tmp_path):
    """The built-in `agentic-judge` topology, live: the solver's entire serialized trace
    is uploaded into the judge's runtime, and the judge — a real agent on the bash+edit
    `default` harness — reads the file with its tools before committing to a score."""
    from verifiers.v1.cli.eval.runner import run_topology_eval
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.topology import TopologyRunner

    config = EvalConfig(
        topology={"id": "agentic-judge", "taskset": {"id": "echo-v1"}},
        num_tasks=1,
        max_turns=6,
        sampling={"max_tokens": 4096, "temperature": 0},
        timeout={"rollout": 300, "scoring": 60},
        retries={"rollout": {"max_retries": 2, "include": ["ProviderError"]}},
        rich=False,
        output_dir=tmp_path,
    )
    env = TopologyRunner(config.topology, config)
    solver, judge = await run_topology_eval(env, config)
    assert solver.errors == [] and judge.errors == []
    assert judge.parents == [solver.id] and judge.trainable is False
    assert judge.num_turns >= 2  # it actually investigated (read the file, then scored)
    assert solver.metrics["judge_committed"] == 1.0
    assert solver.rewards["judge"] > 0.5


@pytest.mark.e2e
async def test_writer_editors_topology(tmp_path):
    """The `writer-editors-v1` example, live (all agents on the in-process `direct`
    harness): draft -> editor critique (fan-out of 1) -> revision (fan-in), then one
    `vf.Judge` call puts the same `improvement` reward on every trace."""
    from verifiers.v1.cli.eval.runner import run_topology_eval
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.topology import TopologyRunner

    config = EvalConfig(
        topology={"id": "writer-editors-v1", "num_editors": 1},
        num_tasks=1,
        max_turns=2,
        sampling={"max_tokens": 4096, "temperature": 0},
        timeout={"rollout": 300, "scoring": 120},
        retries={"rollout": {"max_retries": 2, "include": ["ProviderError"]}},
        rich=False,
        output_dir=tmp_path,
    )
    env = TopologyRunner(config.topology, config)
    draft, edit, revision = await run_topology_eval(env, config)
    assert [t.agent for t in (draft, edit, revision)] == ["writer", "editor", "writer"]
    assert all(t.errors == [] for t in (draft, edit, revision))
    assert revision.parents == [draft.id, edit.id]  # the fan-in
    improvements = {t.rewards["improvement"] for t in (draft, edit, revision)}
    assert len(improvements) == 1  # one verdict, every trace
    assert revision.info.get("judge")  # the judge call was recorded on the final draft


@pytest.mark.e2e
async def test_chess_topology(tmp_path):
    """The `chess-v1` example, live: two direct-harness seats play one game through
    live sessions — each seat ONE multi-turn trace with the opponent's moves as its
    user turns, the host-side board adjudicating. Asserts the session plumbing (both
    episodes alternate, end cleanly, outcomes sum to one game), not chess skill."""
    from verifiers.v1.cli.eval.runner import run_topology_eval
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.topology import TopologyRunner

    config = EvalConfig(
        topology={"id": "chess-v1", "max_plies": 6, "illegal_retries": 2},
        num_tasks=1,
        max_turns=16,
        sampling={"max_tokens": 512, "temperature": 0},
        timeout={"rollout": 420, "scoring": 60},
        rich=False,
        output_dir=tmp_path,
    )
    env = TopologyRunner(config.topology, config)
    traces = await run_topology_eval(env, config)
    seats = {t.agent: t for t in traces}
    assert set(seats) == {"white", "black"}
    assert all(t.errors == [] for t in traces)
    assert sum(t.rewards["outcome"] for t in traces) == 1.0  # one game's points
    assert all(t.num_turns >= 1 for t in traces)  # both seats actually played
    assert seats["white"].info["chess"]["plies"] >= 2  # ...against each other


@pytest.mark.e2e
async def test_debate_topology(tmp_path):
    """The `debate-v1` example, live: three concurrent seats of one agent config give
    openings, rebuttals, and peer votes through suspended sessions. Asserts the N-ary
    plumbing (3 coherent multi-turn traces, votes tallied into declared rewards)."""
    from verifiers.v1.cli.eval.runner import run_topology_eval
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.topology import TopologyRunner

    config = EvalConfig(
        topology={"id": "debate-v1", "num_debaters": 3, "num_rounds": 1},
        num_tasks=1,
        max_turns=8,
        sampling={"max_tokens": 1024, "temperature": 0},
        timeout={"rollout": 420, "scoring": 60},
        rich=False,
        output_dir=tmp_path,
    )
    env = TopologyRunner(config.topology, config)
    traces = await run_topology_eval(env, config)
    assert [t.agent for t in traces] == ["debater"] * 3
    assert all(t.errors == [] for t in traces)
    assert all(t.num_turns == 3 for t in traces)  # opening + rebuttal + vote
    total_votes = sum(t.info["debate"]["votes_received"] for t in traces)
    valid_ballots = sum(t.info["debate"]["voted_validly"] for t in traces)
    assert total_votes == valid_ballots  # every valid ballot landed on someone
    assert all("support" in t.rewards for t in traces)


@pytest.mark.e2e
@pytest.mark.subprocess
async def test_shared_runtime_topology(tmp_path):
    """The `shared-runtime-v1` example, live: `go` provisions one box via
    `run.agent("writer").provision(task)`, the writer's `finalize` writes its reply into
    it, and the reader — borrowed into the SAME box — verifies the artifact in `setup`.
    The borrowed-runtime plumbing end to end."""
    from verifiers.v1.cli.eval.runner import run_topology_eval
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.topology import TopologyRunner

    config = EvalConfig(
        topology={"id": "shared-runtime-v1"},
        num_tasks=1,
        max_turns=2,
        sampling={"max_tokens": 512, "temperature": 0},
        timeout={"rollout": 180, "scoring": 60},
        retries={"rollout": {"max_retries": 2, "include": ["ProviderError"]}},
        rich=False,
        output_dir=tmp_path,
    )
    env = TopologyRunner(config.topology, config)
    written, read = await run_topology_eval(env, config)
    assert written.errors == [] and read.errors == []
    assert (written.agent, read.agent) == ("writer", "reader")
    assert read.parents == [written.id]
    # both rollouts rode the one provisioned box, and neither owned it
    assert written.info["agent"]["runtime"]["borrowed"] is True
    assert read.info["agent"]["runtime"]["borrowed"] is True
    assert (
        written.info["agent"]["runtime"]["descriptor"]
        == read.info["agent"]["runtime"]["descriptor"]
    )
    assert read.rewards["read_shared_note"] == 1.0  # the handoff verified in-runtime
    assert written.rewards["handoff_succeeded"] == 1.0  # mirrored onto the writer


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
