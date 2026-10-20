"""Multi-agent episodes: an `Env` with two roles (duet), best-of-n over a plain
taskset, and turn-coupled self-play (kuhn-poker)."""

import json


async def test_multi_agent_env(run):
    """An `Env` subclass shipped with its taskset (duet): two roles run the
    task, `score()` episodes a sibling-dependent metric, and one episode carries
    two role-stamped traces."""
    (episode,) = await run(
        "duet",
        harness=None,  # both duet seats pin their own harness
        max_turns=2,
    )
    traces = episode.traces
    assert all(trace.num_branches == 1 for trace in traces)
    assert len(traces) == 2  # one episode, one trace per role
    assert sorted(t.agent.name for t in traces) == ["a", "b"]
    (b,) = [t for t in traces if t.agent.name == "b"]
    assert b.agent.trainable is False
    for trace in traces:
        assert trace.ok
        assert trace.reward == 1.0  # each seat's own task reward
        assert trace.metrics["duet"] == 1.0  # the sibling-dependent signal
    # The wire form (what traces.jsonl stores): one episode carrying both traces, each
    # self-stamped on its agent info (completion order — the seats land in either order).
    row = json.loads(episode.model_dump_json(exclude_none=True))
    assert row["env"] == {"id": "duet"}
    by_name = {t["agent"]["name"]: t for t in row["traces"]}
    assert set(by_name) == {"a", "b"}
    assert by_name["a"]["agent"]["trainable"] is True
    assert by_name["b"]["agent"]["trainable"] is False


async def test_env_id_best_of_n(run):
    """`--env.id` pairs a bundled env with an arbitrary taskset: best-of-n over the
    plain echo taskset — n solver attempts in one episode, sibling-scored."""
    (episode,) = await run(
        "echo",
        harness=None,  # a multi-agent env refuses the run-level harness
        env={"id": "best-of-n", "n": 2, "agent": {"harness": {"id": "null"}}},
        max_turns=2,
    )
    traces = episode.traces
    assert all(trace.num_branches == 1 for trace in traces)
    assert len(traces) == 2  # one episode, two attempts
    assert all(t.agent.name == "agent" and t.ok for t in traces)
    assert any(t.metrics["best"] == 1.0 for t in traces)
    assert all(t.metrics["pass_at_n"] == 1.0 for t in traces)  # echo always passes


async def test_kuhn_poker_self_play(run):
    """The turn-coupled proof env: one Kuhn poker hand, both seats live interactions
    of the run's own model (self-play), refereed host-side, paid out zero-sum."""
    (episode,) = await run(
        "kuhn-poker",
        harness=None,  # both seats pin the null harness themselves
        max_turns=8,
        # The Q decision (the one mixed-strategy spot in Kuhn) can cost a reasoning
        # model thousands of tokens; the default 2048 truncates to an empty reply AND
        # exhausts the rollout budget, so the invalid-move retry is never served.
        max_tokens=8192,
        rollout_timeout=300,
    )
    traces = episode.traces
    assert all(trace.num_branches == 1 for trace in traces)
    assert sorted(t.agent.name for t in traces) == ["player0", "player1"]
    payoffs = {t.agent.name: t.rewards["payoff"].score for t in traces}
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
