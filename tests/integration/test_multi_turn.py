"""One task, many turns: the model completes a shell task (echo-agentic: write a phrase
to a file, checked in the runtime) across the harness x runtime grid, and on prime over
every other `dialect` a harness config can select. A turn the harness re-sends
differently, beyond what its `replay_view` declares, forks the trace."""

from collections import defaultdict

import pytest
from grid import RUNTIMES, assert_one_branch, dialect_grid, grid, mark

MULTI_TURN = [
    *grid(runtimes=RUNTIMES, needs=("shell",)),
    *dialect_grid(needs=("shell",)),
    pytest.param(
        "bash",
        {"type": "prime", "allow": []},
        marks=[mark("bash"), mark("prime")],
        id="bash-in-prime-framework-only",
    ),
]


@pytest.mark.parametrize("harness,harness_runtime", MULTI_TURN, indirect=True)
async def test_multi_turn(run, harness, harness_runtime):
    episodes = await run(
        "echo-agentic",
        harness=harness,
        runtime=harness_runtime,
        max_turns=10,
        max_tokens=8192,
        n=12
        if harness.id == "claude-code" and harness_runtime["type"] == "prime"
        else 1,
    )
    for episode in episodes:
        (trace,) = episode.traces
        if harness.id == "claude-code" and trace.num_branches > 1:
            siblings = defaultdict(list)
            for i, node in enumerate(trace.nodes):
                siblings[node.parent].append(i)
            for parent, indices in siblings.items():
                for j in indices[1:]:
                    a, b = trace.nodes[indices[0]], trace.nodes[j]
                    print(
                        "CLAUDE FORK",
                        parent,
                        a.sampled,
                        b.sampled,
                        a.message.role,
                        b.message.role,
                        len(a.tools),
                        len(b.tools),
                        a.tools == b.tools,
                    )
                    for key in (
                        "content",
                        "reasoning_content",
                        "provider_state",
                        "tool_calls",
                        "tool_call_id",
                    ):
                        left = getattr(a.message, key, None)
                        right = getattr(b.message, key, None)
                        if left != right:
                            print(
                                "CLAUDE DIFFERENCE",
                                key,
                                type(left).__name__,
                                type(right).__name__,
                                len(left or []),
                                len(right or []),
                            )
                            if isinstance(left, str) and isinstance(right, str):
                                print(
                                    "CLAUDE WHITESPACE", left.strip() == right.strip()
                                )
                    if a.message.role == b.message.role == "system":
                        for label in (
                            "command prefix",
                            "command injection",
                            "file paths",
                            "title",
                            "summariz",
                            "safety classifier",
                            "bash commands",
                            "permission",
                            "skill",
                        ):
                            print(
                                "CLAUDE SYSTEM CATEGORY",
                                label,
                                label in str(a.message.content).lower(),
                                label in str(b.message.content).lower(),
                            )
    for episode in episodes:
        (trace,) = episode.traces
        assert_one_branch(trace, harness, forks=("openclaw",))
        assert trace.ok, [error.message for error in trace.errors]
        assert trace.num_turns >= 1  # ran a command, then finished
        assert trace.reward == 1.0
