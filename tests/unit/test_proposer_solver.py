"""proposer_solver: the proposer's claimed answer is the solvers' ground truth only
while the solvers don't unanimously contradict it. A rejected label earns no
learnability, and the episode's solver rollouts leave the training set instead of
training toward an unconfirmed label. Solvers who disagree with EACH OTHER confirm
the label — their split is the difficulty signal `learnability` pays for."""

import json
import sys
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "environments" / "proposer_solver")
)
from proposer_solver.taskset import (
    ProposerSolverEnv,
    ProposerSolverTaskset,
    SolveTask,
)


def _trace(task, reply: str, agent: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(name=agent, config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        nodes=[
            MessageNode(
                parent=None,
                message=UserMessage(content=task.data.prompt),
                sampled=False,
            ),
            MessageNode(
                parent=0, message=AssistantMessage(content=reply), sampled=True
            ),
        ],
    )


async def _episode(problem: str, answer: int, replies: list[str]):
    seed = next(iter(ProposerSolverTaskset(vf.TasksetConfig())))
    contract = "I verified it.\n" + json.dumps({"problem": problem, "answer": answer})
    proposer = _trace(seed, contract, "proposer")
    solve = SolveTask.from_trace(proposer)
    solvers = [_trace(solve, reply, "solver") for reply in replies]
    for trace in solvers:
        await solve.score(trace, None)
    env = object.__new__(ProposerSolverEnv)  # finalize only reads the episode
    await env.finalize(
        seed,
        vf.Episode(
            task=vf.TraceTask(type="ProposeTask", data=seed.data),
            traces=[proposer, *solvers],
        ),
    )
    return proposer, solvers


async def test_unanimous_contradiction_rejects_the_label():
    """Issue #2947: "What is 1+1?" labelled 3 with solvers at 2, 2, 3, 3 earned
    learnability 1.0 — a false label paid like a genuinely hard problem."""
    proposer, solvers = await _episode("What is 1+1?", 3, ["2", "2", "3", "3"])
    assert proposer.rewards["learnability"].score == 0.0
    assert proposer.metrics["label_confirmed"] == 0.0
    assert all(not trace.agent.trainable for trace in solvers)


async def test_ambiguous_problem_with_one_dominant_misreading_is_rejected():
    proposer, solvers = await _episode(
        "If x^2 = 9, what is x?", -3, ["3", "3", "-3", "-3"]
    )
    assert proposer.rewards["learnability"].score == 0.0
    assert all(not trace.agent.trainable for trace in solvers)


async def test_all_solvers_wrong_together_reject_the_label():
    proposer, solvers = await _episode("What is 1+1?", 2, ["7", "7", "7", "7"])
    assert proposer.rewards["learnability"].score == 0.0
    assert all(not trace.agent.trainable for trace in solvers)


async def test_split_solvers_confirm_a_hard_true_label():
    """Replies 2, 2, 7, 11 against the true label 2: the non-matching solvers
    disagree with each other, so the label stands and learnability peaks."""
    proposer, solvers = await _episode("A hard problem", 2, ["2", "2", "7", "11"])
    assert proposer.rewards["learnability"].score == 1.0
    assert proposer.metrics["label_confirmed"] == 1.0
    assert all(trace.agent.trainable for trace in solvers)


async def test_easy_true_label_needs_no_gate_to_score_zero():
    proposer, solvers = await _episode("What is 1+1?", 2, ["2", "2", "2", "2"])
    assert proposer.rewards["learnability"].score == 0.0  # 4p(1-p) at p=1
    assert proposer.metrics["label_confirmed"] == 1.0
    assert all(trace.agent.trainable for trace in solvers)
