"""End-to-end v1 eval smoke tests."""

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
    # The resolved sampling rides the trace (policy metadata for trainers).
    assert trace.sampling is not None and trace.sampling.temperature == 0


@pytest.mark.e2e
async def test_user(run_v1, harness_runtime, tmp_path):
    """Multi-turn, driven by a scripted user — a chat-session loop in the env's
    `rollout()` — across the harness runtime matrix. The task is prompt-less, so one
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
    assert trace.errors == []
    assert trace.num_turns >= 2  # genuinely multi-turn
    assert trace.stop_condition == "user_closed"  # leaving the session loop ended it
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_chat(live_ctx):
    """Drive an agent turn-by-turn through `agent.chat()` — the caller IS the run's
    user. Runs on the in-process `direct` harness
    (its live coverage too): no subprocess, nothing provisioned, yet a real rollout —
    trace, scoring, and the `user_closed` stop all apply."""
    import verifiers.v1 as vf
    from verifiers.v1.harnesses.direct import DirectHarness, DirectHarnessConfig

    agent = vf.Agent(DirectHarness(DirectHarnessConfig()), live_ctx)
    task = vf.Task(
        vf.TaskData(
            idx=0,
            prompt=None,  # chat opens the conversation
            system_prompt="Repeat the user's message back exactly, no extra words.",
        )
    )
    async with agent.chat(task) as session:
        first = await session.turn("hello world")
        assert not first.stopped
        assert "hello world" in first.text.lower()
        second = await session.turn("goodbye world")
        assert not second.stopped
        assert "goodbye world" in second.text.lower()
    trace = session.trace
    assert trace is not None and trace.errors == []
    assert trace.stop_condition == "user_closed"  # closing the chat ended the run
    assert trace.num_turns == 2


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
async def test_multi_agent_env(run_v1, tmp_path):
    """An `Environment` subclass shipped with its taskset (duet-v1): two roles run the
    task, `score()` records a sibling-dependent metric, and one eval rollout lands one
    record carrying two role-stamped traces."""
    import json

    traces = await run_v1(
        "duet-v1",
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
    # On disk: one record line carrying both traces, role-stamped.
    (line,) = (tmp_path / "traces.jsonl").read_text().splitlines()
    row = json.loads(line)
    assert row["env"] == "duet-v1"
    assert [t["role"] for t in row["traces"]] == ["a", "b"]
    assert [t.get("trainable") for t in row["traces"]] == [True, False]


@pytest.mark.e2e
async def test_env_id_best_of_n(run_v1, tmp_path):
    """`--env.id` pairs a bundled env with an arbitrary taskset: best-of-n over the
    plain echo taskset — n solver attempts in one record, sibling-scored."""
    traces = await run_v1(
        "echo-v1",
        harness="null",
        env={"id": "best-of-n", "n": 2},
        output_dir=tmp_path,
        max_turns=2,
    )
    assert len(traces) == 2  # one env-rollout, two attempts
    assert all(t.role == "solver" and t.errors == [] for t in traces)
    assert any(t.metrics["best"] == 1.0 for t in traces)
    assert all(t.metrics["pass_at_n"] == 1.0 for t in traces)  # echo always passes


@pytest.mark.e2e
async def test_env_id_judge(run_v1, tmp_path):
    """The judge env over the echo taskset: the solver's trace gains a `judge` reward
    from a real (direct-harness, untrainable) judge agent's verdict."""
    traces = await run_v1(
        "echo-v1",
        harness="null",
        env={"id": "judge"},
        output_dir=tmp_path,
        max_turns=2,
    )
    assert sorted(t.role for t in traces) == ["judge", "solver"]
    (solver,) = [t for t in traces if t.role == "solver"]
    (judge,) = [t for t in traces if t.role == "judge"]
    assert solver.errors == [] and judge.errors == []
    assert judge.trainable is False
    assert solver.rewards["echoed"] == 1.0  # the task's own reward still runs
    # Wiring, not taste: the judge followed the default `score` spec's output
    # contract and its parsed verdict is exactly what landed on the solver (the
    # grade itself is the model's call — a bare echo can be judged middling).
    assert "SCORE:" in (judge.last_reply or "")
    assert 0.0 <= solver.rewards["judge"] <= 1.0


@pytest.mark.e2e
async def test_env_id_judge_over_tool_taskset(run_v1, tmp_path):
    """The headline pairing: a TOOL-USING solver judged on its whole process. The
    judge seat plays env-minted plain tasks (`vf.Role(cfg, mcp=False)`), so the env
    loads over a tool-declaring taskset — the pairing construction used to refuse —
    and `full_trace` shows the judge the solver's tool calls."""
    traces = await run_v1(
        "echo-tool-v1",
        harness="null",
        env={"id": "judge", "spec": {"view": "full_trace"}},
        output_dir=tmp_path,
        max_turns=6,
    )
    (solver,) = [t for t in traces if t.role == "solver"]
    (judge,) = [t for t in traces if t.role == "judge"]
    assert solver.errors == [] and judge.errors == []
    assert solver.num_turns >= 2  # the solver actually used its tool
    assert solver.rewards["echoed"] == 1.0  # the task's own reward still runs
    assert "SCORE:" in (judge.last_reply or "")
    assert 0.0 <= solver.rewards["judge"] <= 1.0


@pytest.mark.e2e
async def test_env_id_judge_rubric_spec(run_v1, tmp_path):
    """One rubric file, agent-executed: the judge env's verdict spec is a judge
    plugin, so the same grading criteria a `taskset.task.judges` entry runs as a
    bare call here drive a real judge agent — per-criterion metrics and the weighted
    total land on the solver exactly as the plugged tier records them."""
    import json

    rubric = tmp_path / "grading.json"
    rubric.write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "name": "echoed",
                        "text": "Does the response repeat the user's phrase?",
                    },
                    {
                        "name": "no_extras",
                        "text": "Is the response free of extra commentary?",
                    },
                ]
            }
        )
    )
    traces = await run_v1(
        "echo-v1",
        harness="null",
        env={"id": "judge", "spec": {"id": "rubric", "path": str(rubric)}},
        output_dir=tmp_path / "out",
        max_turns=2,
        rollout_timeout=300,
    )
    (solver,) = [t for t in traces if t.role == "solver"]
    (judge,) = [t for t in traces if t.role == "judge"]
    assert solver.errors == [] and judge.errors == []
    assert 0.0 <= solver.rewards["rubric"] <= 1.0
    assert set(solver.metrics) >= {"rubric/echoed", "rubric/no_extras"}


@pytest.mark.e2e
async def test_env_id_user_sim(run_v1, tmp_path):
    """The user-sim env over the echo taskset: a modeled user (direct harness) opens
    the conversation from the task's prompt-as-scenario; the assistant's trace is
    judged by the task's own reward; both sides land role-stamped on one record."""
    traces = await run_v1(
        "echo-v1",
        harness="null",
        env={"id": "user-sim"},
        output_dir=tmp_path,
        max_turns=6,
        rollout_timeout=300,
    )
    assert sorted(t.role for t in traces) == ["assistant", "user"]
    (assistant,) = [t for t in traces if t.role == "assistant"]
    (user,) = [t for t in traces if t.role == "user"]
    assert assistant.errors == [] and user.errors == []
    assert user.trainable is False
    assert user.num_turns >= 1  # the modeled user actually spoke
    assert assistant.metrics["user_turns"] >= 1
    # `mask_prompt`: the scenario is hidden from the assistant's harness (the run's
    # visible data) while the task's own rewards still scored the real row — and the
    # record keeps the unmasked task for provenance.
    assert assistant.task.data.prompt is None
    assert "echoed" in assistant.rewards
    from verifiers.v1.cli.output import read_records
    from verifiers.v1.trace import WireTrace

    (record,) = read_records(tmp_path, WireTrace)
    assert record.task.data.prompt is not None


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
        harness="null",
        env={"id": "user-sim"},
        output_dir=tmp_path,
        max_turns=8,
        rollout_timeout=300,
    )
    (assistant,) = [t for t in traces if t.role == "assistant"]
    (user,) = [t for t in traces if t.role == "user"]
    assert assistant.errors == [] and user.errors == []
    assert assistant.task.data.prompt is None  # the scenario stayed off the wire
    assert user.num_turns >= 1  # the modeled user actually drove the exchange
    assert assistant.rewards["echoed"] == 1.0  # the tool ran, mid-conversation
    # The tool was advertised to the masked chat exactly as to any run.
    assert assistant.tools is not None
    assert any(tool.name == "echo_back" for tool in assistant.tools)


@pytest.mark.e2e
async def test_kuhn_poker_self_play(run_v1, tmp_path):
    """The turn-coupled proof env: one Kuhn poker hand, both seats live chat sessions
    of the run's own model (self-play), refereed host-side, paid out zero-sum."""
    traces = await run_v1(
        "kuhn-poker-v1",
        output_dir=tmp_path,
        max_turns=8,
        # The Q decision (the one mixed-strategy spot in Kuhn) can cost a reasoning
        # model thousands of tokens; the default 2048 truncates to an empty reply AND
        # exhausts the rollout budget, so the invalid-move retry is never served.
        max_tokens=8192,
        rollout_timeout=300,
    )
    assert sorted(t.role for t in traces) == ["player0", "player1"]
    payoffs = {t.role: t.rewards["payoff"] for t in traces}
    assert payoffs["player0"] + payoffs["player1"] == 0  # zero-sum
    assert abs(payoffs["player0"]) in (1.0, 2.0)
    for trace in traces:
        assert trace.errors == []
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
    config from wire data, and the multi-trace record rides the serve protocol."""
    traces = await run_v1_server(
        "duet-v1",
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
