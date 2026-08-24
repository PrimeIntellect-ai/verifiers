import verifiers.v1 as vf
from verifiers.v1.harnesses.prime_agent.harness import _autonomous_args


def _trace() -> vf.Trace:
    return vf.Trace(
        id="trace-1",
        agent=vf.AgentInfo(
            config=vf.AgentConfig(
                max_turns=7,
                max_total_tokens=12345,
                timeout={"rollout": 1.001},
            )
        ),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=0, prompt="test"),
        ),
    )


def test_autonomous_args_disabled() -> None:
    assert _autonomous_args(False, _trace()) == []


def test_autonomous_args_forward_rollout_budgets() -> None:
    assert _autonomous_args(True, _trace()) == [
        "--autonomous",
        "--autonomous-max-turns",
        "7",
        "--autonomous-max-tokens",
        "12345",
        "--autonomous-timeout-ms",
        "1001",
    ]
