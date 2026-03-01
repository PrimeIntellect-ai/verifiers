"""
Proposer-Solver: Two-agent math pipeline with conflicting incentives.

Proposer gives a hint (rewarded for correctness + brevity).
Solver uses the hint to answer (rewarded for correctness only).

Per-actor GRPO demo: optimal rewards differ at convergence.
"""

import re

from verifiers.envs.agent import Agent
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.envs.taskset import TaskSet
from verifiers.types import Messages, State
from verifiers.utils.client_utils import get_actor_client
from verifiers.utils.data_utils import load_example_dataset

from datasets import Dataset


# =============================================================================
# Model Configuration
# =============================================================================

PROPOSER_ENDPOINT = None
SOLVER_ENDPOINT = None

proposer_client, proposer_model = get_actor_client(PROPOSER_ENDPOINT)
solver_client, solver_model = get_actor_client(SOLVER_ENDPOINT)

MAX_HINT_LEN = 500  # chars — hints longer than this get 0 brevity bonus


# =============================================================================
# Rubric
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    """
    Proposer: correctness * (0.5 + 0.5 * brevity) — wants short hints.
    Solver:   1.0 if correct, 0.0 if not — wants good hints.
    """
    rubric = MultiAgentRubric()

    def proposer_reward(state, **kwargs) -> float:
        correct = state.get("extras", {}).get("correct", False)
        if not correct:
            return 0.0
        hint_len = state.get("extras", {}).get("hint_length", 0)
        brevity = max(0.0, 1.0 - hint_len / MAX_HINT_LEN)
        return 0.5 + 0.5 * brevity

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
    2-turn pipeline: proposer hints, solver executes.

    Turn 1 (proposer): Sees a math problem, writes a concise hint.
    Turn 2 (solver):   Sees the problem + hint, produces numeric answer.
    Evaluate:          Check if solver's answer matches expected.
    """

    def __init__(self, num_examples: int = -1):
        dataset = self._create_dataset(num_examples)

        super().__init__(
            name="proposer_solver_v2",
            dataset=dataset,
            rubric=create_rubric(),
            roles=["proposer", "solver"],
        )

    @staticmethod
    def _create_dataset(num_examples: int = -1) -> Dataset:
        """Load GSM8K and reformat for TaskSet."""
        gsm = load_example_dataset("gsm8k", split="train", n=50, seed=42)
        rows = []
        for i, row in enumerate(gsm):
            if num_examples > 0 and i >= num_examples:
                break
            rows.append({
                "prompt": [{"role": "user", "content": row["question"]}],
                "answer": row["answer"],
                "info": {"problem": row["question"]},
                "example_id": i,
                "task": "proposer_solver_v2",
            })
        return Dataset.from_list(rows)

    # ---- State ----

    async def setup_state(self, state: State) -> State:
        state["extras"]["problem"] = state["input"].get("info", {}).get("problem", "")
        state["extras"]["expected_answer"] = state["input"].get("answer", "")
        state["extras"]["hint"] = ""
        state["extras"]["hint_length"] = 0
        state["extras"]["solver_answer"] = ""
        state["extras"]["correct"] = False
        return state

    # ---- Prompts ----

    async def build_prompt(self, role: str, state: State) -> Messages:
        problem = state["extras"]["problem"]

        if role == "proposer":
            return [
                {"role": "system", "content": (
                    "You are a Math Hint Writer. Given a math problem, write a brief hint "
                    "to help someone solve it.\n\n"
                    "Be as concise as possible — shorter hints are better.\n"
                    "Output ONLY the hint, nothing else."
                )},
                {"role": "user", "content": f"Problem: {problem}\n\nWrite a concise hint:"},
            ]

        else:  # solver
            hint = state["extras"]["hint"]
            return [
                {"role": "system", "content": (
                    "You are a Math Solver. You are given a problem and a hint.\n"
                    "Follow the hint and compute the final answer.\n\n"
                    "Output ONLY the numeric answer. Nothing else."
                )},
                {"role": "user", "content": (
                    f"Problem: {problem}\n\n"
                    f"Hint: {hint}\n\n"
                    f"Compute the answer:"
                )},
            ]

    # ---- Game Logic ----

    async def on_turn_complete(self, state: State) -> None:
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        actor_id = last_step["extras"].get("actor_id", "")
        completion = last_step.get("completion", [])
        if not completion:
            return
        content = completion[-1].get("content", "")

        if actor_id == "proposer":
            state["extras"]["hint"] = content
            state["extras"]["hint_length"] = len(content)

        elif actor_id == "solver":
            state["extras"]["solver_answer"] = content
            expected = state["extras"]["expected_answer"]
            numbers = re.findall(r'-?\d+', content)
            state["extras"]["correct"] = bool(numbers and numbers[-1] == expected)

    async def should_stop(self, state: State) -> bool:
        actor_history = state.get("extras", {}).get("actor_history", [])
        return len(actor_history) >= 2

    async def on_game_end(self, state: State) -> None:
        pass


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(num_examples: int = -1):
    """
    Conflicting-incentives proposer-solver:
        Proposer: rewarded for correctness + brevity
        Solver:   rewarded for correctness only
    """
    task = ProposerSolverTask(num_examples=num_examples)

    proposer = Agent(
        id="proposer",
        max_tokens=150,
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
