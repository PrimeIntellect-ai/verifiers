"""Judged rollouts: a config-plugged rubric judge scores on top of the task's reward, and
the agentic judge envs verify a solver empirically in a sandbox (shared box, or a fresh
box restored from the solver's artifact)."""


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
        runtime={"type": "subprocess"},
        output_dir=tmp_path,
        taskset_overrides={"task": {"judges": [{"id": "rubric", "path": str(rubric)}]}},
        max_turns=2,
    )
    assert trace.ok
    assert trace.rewards["rubric"].score > 0  # the judge's verdict landed in the reward
    assert trace.metrics["rubric/always_yes"] == 1.0
    assert trace.info["judge_calls"]  # the call was recorded onto the trace


async def test_env_id_shared_agentic_judge(run_v1, tmp_path):
    """The shared agentic judge provisions one restricted solver-owned box and
    safely reuses it for the judge's empirical verification."""
    policy = tmp_path / "judge_policy.txt"
    policy.write_text("Check EMPIRICALLY that the agent echoed the word back.")
    traces = await run_v1(
        "echo-v1",
        harness=None,
        env={
            "id": "shared-agentic-judge",
            "solver": {
                "harness": {"id": "bash"},
                "runtime": {"type": "docker", "block": ["example.com"]},
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
        output_dir=tmp_path,
        max_turns=10,
        rollout_timeout=600,
    )
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


async def test_env_id_agentic_judge(run_v1, tmp_path):
    """The isolated agentic judge destroys the restricted solver box, transfers
    its declared artifact, and restores it in a fresh box with the same policy."""
    policy = tmp_path / "isolated_judge_policy.txt"
    policy.write_text(
        "Check EMPIRICALLY that answer.txt contains exactly the requested phrase."
    )
    traces = await run_v1(
        "echo-agentic-v1",
        harness=None,
        env={
            "id": "agentic-judge",
            "solver": {
                "harness": {"id": "bash"},
                "runtime": {"type": "docker", "block": ["example.com"]},
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
        output_dir=tmp_path,
        max_turns=10,
        rollout_timeout=600,
    )
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
