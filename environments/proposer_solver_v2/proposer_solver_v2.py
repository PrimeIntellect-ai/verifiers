"""
Proposer-Solver v2: New decomposition.

Old style: ProposerSolverEnv(MultiAgentEnv) + SolverEnv(MultiAgentEnv) — two env
           subclasses, Protocol.spawn() for child rollouts, everything interleaved.

New style: Game logic in ProposerSolverTask(TaskSet). Two agents, 2 turns.
           No spawning, no child envs. Just proposer → solver → evaluate.

What this shows about the abstractions:
    - TaskSet handles asymmetric roles (proposer sees problem, solver sees problem + strategy)
    - Different roles could use different Harnesses (solver could use ToolHarness
      to run code, but here both use SingleTurnHarness for simplicity)
    - Rubric: proposer rewarded if solver succeeds, solver rewarded for correctness
    - Only 2 turns total — simplest possible multi-agent pipeline

The old spawning pattern (N solver copies in parallel) is a separate concern.
That needs env-level orchestration, not TaskSet.
"""

import re

from datasets import Dataset

from verifiers.agent import Agent
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.taskset import TaskSet
from verifiers.types import Messages, State
from verifiers.utils.client_utils import get_actor_client


# =============================================================================
# Model Configuration
# =============================================================================

PROPOSER_ENDPOINT = None
SOLVER_ENDPOINT = None

proposer_client, proposer_model = get_actor_client(PROPOSER_ENDPOINT)
solver_client, solver_model = get_actor_client(SOLVER_ENDPOINT)


# =============================================================================
# Rubric
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    """
    Proposer rewarded if solver succeeds (incentivizes clear strategies).
    Solver rewarded for correctness.
    """
    rubric = MultiAgentRubric()

    def proposer_reward(state, **kwargs) -> float:
        return 1.0 if state.get("extras", {}).get("correct", False) else 0.0

    def solver_reward(state, **kwargs) -> float:
        return 1.0 if state.get("extras", {}).get("correct", False) else 0.0

    rubric.add_actor_reward_func("proposer", proposer_reward, weight=1.0)
    rubric.add_actor_reward_func("solver", solver_reward, weight=1.0)
    return rubric


# =============================================================================
# TaskSet: ALL game logic lives here
# =============================================================================

class ProposerSolverTask(TaskSet):
    """
    2-turn pipeline: proposer strategizes, solver executes.

    Turn 1 (proposer): Sees a math problem, proposes a solution strategy.
    Turn 2 (solver):   Sees the problem + proposer's strategy, produces answer.
    Evaluate:          Check if solver's answer matches expected.

    Compare with old ProposerSolverEnv + SolverEnv:
    - BEFORE: Two env subclasses, Protocol.spawn() for N solver copies,
              parent-child state relationships
    - AFTER:  One TaskSet, two turns, no spawning. Clean and testable.
    """

    def __init__(self, num_examples: int = -1):
        dataset = self._create_dataset()
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        super().__init__(
            name="proposer_solver_v2",
            dataset=dataset,
            rubric=create_rubric(),
            roles=["proposer", "solver"],
        )

    @staticmethod
    def _create_dataset() -> Dataset:
        """Math problems with known answers."""
        problems = [
            ("What is 17 + 28?", "45"),
            ("What is 143 - 67?", "76"),
            ("What is 12 * 9?", "108"),
            ("What is 256 + 189?", "445"),
            ("What is 84 - 37?", "47"),
            ("What is 15 * 13?", "195"),
            ("What is 1024 - 768?", "256"),
            ("What is 33 + 77?", "110"),
            ("What is 19 * 7?", "133"),
            ("What is 500 - 234?", "266"),
        ]
        return Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": problem}],
                "answer": answer,
                "info": {"problem": problem},
                "example_id": i,
                "task": "proposer_solver_v2",
            }
            for i, (problem, answer) in enumerate(problems)
        ])

    # ---- State ----

    async def setup_state(self, state: State) -> State:
        state["extras"]["problem"] = state["input"].get("info", {}).get("problem", "")
        state["extras"]["expected_answer"] = state["input"].get("answer", "")
        state["extras"]["strategy"] = ""
        state["extras"]["solver_answer"] = ""
        state["extras"]["correct"] = False
        return state

    # ---- Prompts ----

    async def build_prompt(self, role: str, state: State) -> Messages:
        problem = state["extras"]["problem"]

        if role == "proposer":
            return [
                {"role": "system", "content": (
                    "You are a Math Strategist. Given a math problem, propose a clear "
                    "step-by-step strategy for solving it.\n\n"
                    "Be specific about the operations needed. "
                    "Output ONLY the strategy, nothing else."
                )},
                {"role": "user", "content": f"Problem: {problem}\n\nPropose a solution strategy:"},
            ]

        else:  # solver
            strategy = state["extras"]["strategy"]
            return [
                {"role": "system", "content": (
                    "You are a Math Solver. You are given a problem and a strategy.\n"
                    "Follow the strategy and compute the final answer.\n\n"
                    "Output ONLY the numeric answer. Nothing else."
                )},
                {"role": "user", "content": (
                    f"Problem: {problem}\n\n"
                    f"Strategy: {strategy}\n\n"
                    f"Compute the answer:"
                )},
            ]

    # ---- Game Logic ----

    async def on_turn_complete(self, state: State) -> None:
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        actor_id = last_step.get("extras", {}).get("actor_id", "")
        completion = last_step.get("completion", [])
        if not completion:
            return
        content = completion[-1].get("content", "")

        if actor_id == "proposer":
            # Store strategy for solver to use
            state["extras"]["strategy"] = content

        elif actor_id == "solver":
            # Check answer
            state["extras"]["solver_answer"] = content
            expected = state["extras"]["expected_answer"]
            numbers = re.findall(r'-?\d+', content)
            state["extras"]["correct"] = bool(numbers and numbers[0] == expected)

    async def should_stop(self, state: State) -> bool:
        # Stop after solver has gone (2 turns total)
        actor_history = state.get("extras", {}).get("actor_history", [])
        return len(actor_history) >= 2

    async def on_game_end(self, state: State) -> None:
        # Nothing extra needed — correct flag already set in on_turn_complete
        pass


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(num_examples: int = -1):
    """
    Composition:
        Task   = ProposerSolverTask (math problems + strategy→solve pipeline)
        Agents = {proposer: strategist, solver: calculator}
        Env    = MultiAgentEnv(task, agents)
    """
    task = ProposerSolverTask(num_examples=num_examples)

    proposer = Agent(
        id="proposer",
        max_tokens=200,
        is_trainable=True,
        model=proposer_model,
        client=proposer_client,
    )

    solver = Agent(
        id="solver",
        max_tokens=20,
        is_trainable=True,
        model=solver_model,
        client=solver_client,
    )

    return MultiAgentEnv(
        task=task,
        agents={"proposer": proposer, "solver": solver},
        max_turns=4,
    )
