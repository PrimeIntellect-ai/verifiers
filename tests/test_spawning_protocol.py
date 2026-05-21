"""End-to-end test for SpawningProtocol via a toy proposer/solver env.

The proposer picks an integer N. The protocol then spawns k child solver
rollouts whose job is to double N. We assert that:

  - the parent's trajectory contains both proposer and child solver steps
  - child steps are tagged with agent_id="solver" and is_trainable
  - state["extras"]["spawns"] carries SpawnResult(s) with one State per child
  - each child's reward was computed by its own rubric (score was 1.0 only
    when the solver's completion equaled 2*N)
"""

from __future__ import annotations

import re

import pytest
from datasets import Dataset

import verifiers as vf
from verifiers import (
    Agent,
    MultiAgentEnv,
    Rubric,
    SingleTurnEnv,
    SpawningProtocol,
    SpawnSpec,
)
from verifiers.types import Messages, State


PROPOSER_NUMBER = 5  # what the proposer always picks in this test
NUM_CHILDREN = 3


# --------------------------------------------------------------------------- #
# Child env: simple single-turn doubling env.
# --------------------------------------------------------------------------- #


def _doubling_correct(completion, answer, **_) -> float:
    text = completion if isinstance(completion, str) else completion[-1]["content"]
    match = re.search(r"-?\d+", text)
    if match is None:
        return 0.0
    return 1.0 if int(match.group(0)) == int(answer) else 0.0


@pytest.fixture
def child_solver_env(mock_client):
    """SingleTurnEnv whose prompt asks for 2*N and rubric scores correctness."""
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": f"Double {PROPOSER_NUMBER}."}]],
            "answer": [str(2 * PROPOSER_NUMBER)],
            "example_id": [0],
        }
    )
    rubric = Rubric(funcs=[_doubling_correct])

    env = SingleTurnEnv(
        client=mock_client,
        model="test-model",
        dataset=dataset,
        parser=vf.Parser(),
        rubric=rubric,
    )

    # Pre-stage half of the children's responses to be correct, half wrong.
    # The mocked client returns the same default response unless overridden;
    # since the proposer ALSO uses this client we'll just set a single default
    # and use add_response for both turn types.
    mock_client.add_response(
        [{"role": "user", "content": f"Double {PROPOSER_NUMBER}."}],
        str(2 * PROPOSER_NUMBER),
    )
    return env


# --------------------------------------------------------------------------- #
# Parent env: proposer that emits a number, then spawns NUM_CHILDREN solvers.
# --------------------------------------------------------------------------- #


class _OneShotSpawnProtocol(SpawningProtocol):
    """Spawns child solvers exactly once, immediately after the proposer's turn."""

    def __init__(self, child_env, agent_id: str, num_children: int):
        self._child_env = child_env
        self._agent_id = agent_id
        self._num_children = num_children

    def get_initial_agent(self, state: State) -> str:
        return "proposer"

    def get_next_agent(self, state: State) -> str:
        # Single-turn protocol: never returns a "next" agent because
        # on_turn_complete sets state["is_completed"]=True.
        return "proposer"

    def should_spawn(self, state: State) -> bool:
        # Spawn once: only if the proposer just acted and we haven't spawned yet.
        already_spawned = bool(state["extras"].get("spawns"))
        last_step = state["trajectory"][-1] if state["trajectory"] else None
        last_agent = (last_step or {}).get("extras", {}).get("agent_id")
        return last_agent == "proposer" and not already_spawned

    def get_spawn_specs(self, state: State) -> list[SpawnSpec]:
        # The proposer's "answer" is the last word of its completion.
        text = state["trajectory"][-1]["completion"][-1]["content"]
        n = int(re.search(r"-?\d+", text).group(0))
        prompt = [{"role": "user", "content": f"Double {n}."}]
        inputs = [
            {"prompt": prompt, "answer": str(2 * n), "example_id": i}
            for i in range(self._num_children)
        ]
        return [
            SpawnSpec(
                agent_id=self._agent_id,
                child_env=self._child_env,
                inputs=inputs,
                is_trainable=True,
            )
        ]


class _ProposerEnv(MultiAgentEnv):
    """One-turn proposer that picks a number, registered as a trainable agent."""

    async def build_agent_prompt(self, agent_id: str, state: State) -> Messages:
        return [
            {
                "role": "user",
                "content": "Pick an integer and the solver will try to double it.",
            }
        ]

    @vf.stop
    async def proposer_done(self, state: State, **kwargs) -> bool:
        # End the rollout once the proposer's turn has been spawned out;
        # the spawn block in MultiAgentEnv.rollout() runs before the next
        # iteration's is_completed check, so children finish first.
        return bool(state.get("extras", {}).get("spawns"))


@pytest.fixture
def proposer_env(mock_client, child_solver_env):
    protocol = _OneShotSpawnProtocol(
        child_env=child_solver_env, agent_id="solver", num_children=NUM_CHILDREN
    )
    rubric = Rubric()
    env = _ProposerEnv(
        protocol=protocol,
        client=mock_client,
        model="test-model",
        dataset=Dataset.from_dict({"prompt": [[{"role": "user", "content": "go"}]], "example_id": [0]}),
        parser=vf.Parser(),
        rubric=rubric,
        max_turns=8,
    )
    env.register_agent(Agent(id="proposer", system_prompt="", is_trainable=True))
    env.register_agent(Agent(id="solver", system_prompt="", is_trainable=True))

    mock_client.add_response(
        [
            {
                "role": "user",
                "content": "Pick an integer and the solver will try to double it.",
            }
        ],
        str(PROPOSER_NUMBER),
    )
    return env


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_spawning_protocol_runs_children_and_records_spawns(
    proposer_env, mock_client
):
    """One proposer turn → NUM_CHILDREN solver children → all embedded + recorded."""
    state = await proposer_env.rollout(
        {"prompt": [{"role": "user", "content": "go"}], "example_id": 0},
        client=mock_client,
        model="test-model",
        sampling_args={"temperature": 1.0},
    )

    # 1. The parent trajectory contains the proposer's step plus one per child.
    agent_ids = [s["extras"].get("agent_id") for s in state["trajectory"]]
    assert agent_ids.count("proposer") == 1
    assert agent_ids.count("solver") == NUM_CHILDREN

    # 2. Spawns recorded.
    spawns = state["extras"]["spawns"]
    assert len(spawns) == 1
    spawn = spawns[0]
    assert spawn.spec.agent_id == "solver"
    assert len(spawn.states) == NUM_CHILDREN

    # 3. Children were scored by the child env's own rubric — the mocked
    #    solver always returns 2*N so each child's reward is 1.0.
    for child_state in spawn.states:
        assert child_state["reward"] == 1.0


@pytest.mark.asyncio
async def test_child_trajectory_steps_carry_is_trainable_tag(
    proposer_env, mock_client
):
    state = await proposer_env.rollout(
        {"prompt": [{"role": "user", "content": "go"}], "example_id": 0},
        client=mock_client,
        model="test-model",
        sampling_args={"temperature": 1.0},
    )
    child_steps = [s for s in state["trajectory"] if s["extras"].get("agent_id") == "solver"]
    assert child_steps, "expected child steps in parent trajectory"
    for step in child_steps:
        # is_trainable was set on the SpawnSpec; it must flow through to steps.
        assert step["extras"].get("is_trainable") is True
