"""One task, a scripted user: the user speaks first and the harness resumes onto the
conversation each turn (echo-user-sim), across the harness x runtime grid; and the
caller drives an agent turn by turn through `agent.interaction`."""

import pytest
from grid import assert_one_branch, grid

import verifiers.v1 as vf


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("resume",)), indirect=True
)
async def test_user(run, harness, harness_runtime):
    """Multi-turn, driven by a scripted user — an interaction loop in the env's
    `run()`. The task is prompt-less, so one run covers the whole exchange shape:
    the caller opens (the user speaks first), each later turn resumes the harness
    onto the conversation, and leaving the loop ends the exchange (`user_closed`).
    The user runs in the eval process itself, so there is no placement axis."""
    (episode,) = await run(
        "echo-user-sim",
        harness=harness,
        runtime=harness_runtime,
        max_turns=6,
        max_tokens=8192,
    )
    (trace,) = episode.traces
    assert_one_branch(trace, harness, forks=("openclaw",))
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 2  # genuinely multi-turn
    assert trace.stop_condition == "user_closed"  # leaving the interaction ended it
    assert trace.reward == 1.0


async def test_interaction(live_ctx):
    """Drive an agent turn-by-turn through `agent.interaction()` — the caller IS
    the run's user. Runs on the tool-less `null` harness: nothing but the exchange
    itself, yet a real rollout — trace, scoring, and the `user_closed` stop all
    apply."""
    from verifiers.v1.harnesses.null import NullHarnessConfig

    agent = vf.make_agent(
        vf.AgentConfig(
            harness=NullHarnessConfig(id="null"),
            model=live_ctx.model,
            sampling=live_ctx.sampling,
            client=live_ctx.client,
        ),
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
    assert trace.num_branches == 1
