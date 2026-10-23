"""A task that refuses the agent's submission at `finalize` (`SubmissionRefused`) scores
it 0: the rollout is ok and graded, the refusal is on the trace, and no verifier runs."""

from test_agent_resume import (  # noqa: F401 - fixtures
    NotesTask,
    ScriptHarness,
    fake_boxes,
)
from test_provider_retry import Provider

import verifiers.v1 as vf
from verifiers.v1.errors import SubmissionRefused
from verifiers.v1.utils.decorators import reward

SCORED: list[str] = []


class RefusingTask(NotesTask):
    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        raise SubmissionRefused("collect hook refused the submission: src/build.rs")

    @reward(weight=2.0)
    async def solved(self, trace: vf.Trace) -> float:
        SCORED.append(trace.id)
        return 1.0


async def test_a_refused_submission_scores_zero_without_grading():
    async with Provider() as provider:
        agent = vf.Agent(
            vf.AgentConfig(
                model="scripted",
                client={
                    "type": "eval",
                    "base_url": provider.url,
                    "api_key_var": "SCRIPTED_KEY",
                },
                runtime={"type": "subprocess"},
                harness={"id": "null"},
            )
        )
        agent.harness = ScriptHarness(agent.harness.config)
        trace = await agent.run(RefusingTask(vf.TaskData(idx=0, prompt="start")))
    assert trace.ok and not trace.errors
    assert "src/build.rs" in trace.info["submission_refused"]
    assert trace.rewards["solved"] == vf.Reward(score=0.0, weight=2.0)
    assert trace.reward == 0.0
    assert SCORED == []
