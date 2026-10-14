"""Agentic judging uses shared or isolated scoring to verify a solver in a sandbox."""


async def test_agentic_judge_shared_verifier(run, tmp_path):
    """The shared agentic judge provisions one restricted solver-owned box and
    safely reuses it for the judge's empirical verification."""
    policy = tmp_path / "judge_policy.txt"
    policy.write_text("Check EMPIRICALLY that the agent echoed the word back.")
    (episode,) = await run(
        "echo-v1",
        harness=None,
        env={
            "id": "agentic-judge",
            "solver": {
                "harness": {"id": "bash"},
                "runtime": {"type": "docker", "block": ["example.com"]},
                "scoring": {"mode": "shared"},
            },
            "judge": {
                "harness": {"id": "bash"},
                "max_output_tokens": 8192,
            },
            "task": {
                "prompt": {"path": str(policy)},
                "hint": "Do not rely on README.md",
            },
            "score": {"task_weight": 0.5},
        },
        max_turns=10,
        rollout_timeout=600,
    )
    traces = episode.traces
    assert sorted(t.agent.name for t in traces) == ["judge", "solver"]
    (solver,) = [t for t in traces if t.agent.name == "solver"]
    (judge,) = [t for t in traces if t.agent.name == "judge"]
    assert solver.ok and judge.ok
    assert judge.agent.trainable is False
    assert solver.agent.runtime is not None and judge.agent.runtime is not None
    assert solver.agent.runtime.id == judge.agent.runtime.id
    # The task's own reward keeps its raw score; the rescale lands on the weight.
    assert solver.rewards["echoed"].score == 1.0
    assert solver.rewards["echoed"].weight == 0.5
    assert isinstance(judge.info.get("verdict"), dict)
    assert 0.0 <= solver.rewards["judge"].score <= 1.0


async def test_env_id_agentic_judge(run, tmp_path):
    """The isolated agentic judge destroys the restricted solver box, transfers
    its declared artifact, and restores it in a fresh box with the same policy."""
    policy = tmp_path / "isolated_judge_policy.txt"
    policy.write_text(
        "Check EMPIRICALLY that answer.txt contains exactly the requested phrase."
    )
    (episode,) = await run(
        "echo-agentic-v1",
        harness=None,
        env={
            "id": "agentic-judge",
            "solver": {
                "harness": {"id": "bash"},
                "runtime": {"type": "docker", "block": ["example.com"]},
                "scoring": {"mode": "isolated"},
            },
            "judge": {
                "harness": {"id": "bash"},
                "max_output_tokens": 8192,
            },
            "task": {
                "prompt": {"path": str(policy)},
                "hint": "Do not rely on README.md",
            },
            "score": {"task_weight": 0.5},
        },
        max_turns=10,
        rollout_timeout=600,
    )
    traces = episode.traces
    assert sorted(t.agent.name for t in traces) == ["judge", "solver"]
    (solver,) = [t for t in traces if t.agent.name == "solver"]
    (judge,) = [t for t in traces if t.agent.name == "judge"]
    assert solver.ok and judge.ok
    assert judge.agent.trainable is False
    assert solver.agent.runtime is not None and judge.agent.runtime is not None
    assert solver.agent.runtime.id != judge.agent.runtime.id
    assert solver.agent.runtime.type == judge.agent.runtime.type == "docker"
    assert "/app/answer.txt" in solver.state.artifacts
    assert solver.rewards["wrote_phrase"].score == 1.0
    assert solver.rewards["wrote_phrase"].weight == 0.5
    assert isinstance(judge.info.get("verdict"), dict)
    assert 0.0 <= solver.rewards["judge"].score <= 1.0
