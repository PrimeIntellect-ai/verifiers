"""End-to-end eval runs on trivial tasksets — each scores reward 1.0, with no errors.

Every task is one greedy rollout (`temperature=0`, set in `run_v1`) on a single task with
turn/timeout caps. The reward tests fan out across the full **harness x runtime** matrix (the
`harness` x `runtime` fixtures): the built-in harnesses (default, rlm) x the runtimes
(subprocess/docker/prime, modal excluded).

The capability-sensitive tests read the harness flags: the tools test expects a harness
without `SUPPORTS_TASK_TOOLS` (rlm) to be refused up front (raise); the multi-turn test skips a
harness without `SUPPORTS_USER_SIM` (rlm). The agentic task pins the default harness (only it
exposes the bash tool) and varies runtime.
"""

import pytest


@pytest.mark.e2e
async def test_single_turn(run_v1, harness, runtime, tmp_path):
    """Single-turn (echo a short phrase back)."""
    (trace,) = await run_v1(
        "echo-v1", harness=harness, runtime=runtime, output_dir=tmp_path, max_turns=2
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns == 1
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_multi_turn(run_v1, harness, runtime, harness_supports, tmp_path):
    """Multi-turn, driven by a (container-safe) user simulator. Skipped on harnesses without
    `SUPPORTS_USER_SIM` (rlm: a single-instruction interface that can't take injected turns)."""
    if not harness_supports(harness, "SUPPORTS_USER_SIM"):
        pytest.skip(f"{harness} does not support a user simulator")
    (trace,) = await run_v1(
        "echo-multi-v1",
        harness=harness,
        runtime=runtime,
        output_dir=tmp_path,
        max_turns=6,
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # genuinely multi-turn
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_multi_turn_with_tools(
    run_v1, harness, runtime, harness_supports, tmp_path
):
    """Multi-turn with a colocated tool. An harness whose `SUPPORTS_TASK_TOOLS` is False can't
    drive the task's tools, so the pairing is refused at build time instead of running."""
    if not harness_supports(harness, "SUPPORTS_TASK_TOOLS"):
        with pytest.raises(ValueError, match="task tools"):
            await run_v1(
                "glossary-v1", harness=harness, runtime=runtime, output_dir=tmp_path
            )
        return
    (trace,) = await run_v1(
        "glossary-v1",
        harness=harness,
        runtime=runtime,
        output_dir=tmp_path,
        max_turns=6,
    )
    assert trace.errors == []
    assert not trace.is_truncated
    assert trace.num_turns >= 2  # tool call + answer
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_agentic(run_v1, runtime, tmp_path):
    """Agentic: write a phrase to a file with the bash tool, checked in the runtime. Only the
    default harness exposes the bash tool, so this varies runtime but pins the harness."""
    (trace,) = await run_v1(
        "echo-agentic-v1",
        harness="default",
        enable_bash=True,
        runtime=runtime,
        output_dir=tmp_path,
        max_turns=10,
    )
    assert trace.errors == []
    assert trace.num_turns >= 1  # ran a command, then finished
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_task_tools_own_runtime(
    run_v1, server_runtime, skip_if_unexposable, tmp_path
):
    """A task's tool server runs in its OWN runtime (not colocated) — cover every server
    runtime, with the agent on subprocess. The harness reaches the tool over the runtime's
    resolved URL (a remote sandbox publishes its port; a host one is localhost)."""
    (trace,) = await run_v1(
        "glossary-v1",
        harness="default",
        runtime="subprocess",
        output_dir=tmp_path,
        max_turns=6,
        taskset_overrides={
            "tools": {"colocated": False, "runtime": {"type": server_runtime}}
        },
    )
    skip_if_unexposable(trace)
    assert trace.errors == []
    assert trace.reward == 1.0


@pytest.mark.e2e
async def test_user_own_runtime(run_v1, server_runtime, skip_if_unexposable, tmp_path):
    """The user simulator runs in its OWN runtime — cover every server runtime (host
    subprocess, its own docker container, or its own remote sandbox), with the agent on
    subprocess. Reached host-side via the runtime's published URL or localhost."""
    (trace,) = await run_v1(
        "echo-multi-v1",
        harness="default",
        runtime="subprocess",
        output_dir=tmp_path,
        max_turns=6,
        taskset_overrides={"user": {"runtime": {"type": server_runtime}}},
    )
    skip_if_unexposable(trace)
    assert trace.errors == []
    assert trace.num_turns >= 2
    assert trace.reward == 1.0
