"""
Proposer-Solver v2: New decomposition.

Old style: ProposerSolverEnv(MultiAgentEnv) + SolverEnv(MultiAgentEnv) + Protocol
           Direct subclasses, Actor objects, hierarchical spawning.

New style: Game logic in ProposerSolverTask(TaskSet). Agents are separate.
           MultiAgentEnv just runs the loop.

Two roles:
    proposer — generates a simple arithmetic problem
    solver   — solves the problem, outputs a numeric answer

What this demonstrates:
    - TaskSet owns game logic (prompts, turn order, scoring)
    - Simple 2-role sequential pipeline (proposer -> solver)
    - Proposer rewarded by solver success (collaborative incentive)
    - Both roles trainable with per-actor GRPO advantages

Compare with proposer_code_solver (same pipeline but with tools):
    - BEFORE: solver uses run_code() tool to compute answers
    - HERE:   solver answers directly (no tools)
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

PROPOSER_ENDPOINT = "qwen3-235b-i"
SOLVER_ENDPOINT = "qwen3-30b-i"

proposer_client, proposer_model = get_actor_client(PROPOSER_ENDPOINT)
solver_client, solver_model = get_actor_client(SOLVER_ENDPOINT)


# =============================================================================
# Rubric
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    """Both roles rewarded if solver gets the right answer."""
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
    Proposer generates math problems, Solver answers them.

    Compare with old ProposerSolverEnv + SolverEnv + Protocol:
    - BEFORE: Two MultiAgentEnv subclasses, Actor objects, Protocol for
              hierarchical spawning between parent and child environments
    - AFTER:  All game logic in this TaskSet. Single MultiAgentEnv.
              Proposer and solver are just roles taking turns.

    Turn 1 (proposer): Generate a simple arithmetic problem.
    Turn 2 (solver):   Solve the problem, output numeric answer.
    Evaluate:          Check if solver's answer matches expected.
    """

    def __init__(self, num_examples: int = -1):
        dataset = self._create_dataset()
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        super().__init__(
            name="proposer_solver",
            dataset=dataset,
            rubric=create_rubric(),
            roles=["proposer", "solver"],
        )

    @staticmethod
    def _create_dataset() -> Dataset:
        """Seed prompts for the proposer to generate problems."""
        return Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": "Generate a math problem."}],
                "answer": "",
                "info": {"seed": i},
                "example_id": i,
                "task": "proposer_solver",
            }
            for i in range(10)
        ])

    # ---- State ----

    async def setup_state(self, state: State) -> State:
        state["extras"]["problem"] = ""
        state["extras"]["expected_answer"] = ""
        state["extras"]["solver_answer"] = ""
        state["extras"]["correct"] = False
        return state

    # ---- Prompts ----

    async def build_prompt(self, role: str, state: State) -> Messages:
        if role == "proposer":
            return [
                {"role": "system", "content": (
                    "You are a Math Problem Proposer. Generate a simple arithmetic problem.\n\n"
                    "Rules:\n"
                    "1. Create a problem using +, -, or * with two numbers between 1-20\n"
                    "2. Format: Just state the problem, e.g., \"What is 7 + 5?\"\n"
                    "3. Make it solvable but not trivial\n\n"
                    "Output only the problem, nothing else."
                )},
                {"role": "user", "content": "Generate a math problem."},
            ]

        else:  # solver
            problem = state["extras"]["problem"]
            return [
                {"role": "system", "content": (
                    "You are a Math Solver. Solve the given problem.\n\n"
                    "Rules:\n"
                    "1. Calculate the answer\n"
                    "2. Output ONLY the numeric answer, nothing else\n"
                    "3. Example: If asked \"What is 3 + 4?\", output just: 7"
                )},
                {"role": "user", "content": problem},
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
        content = completion[-1].get("content", "") or ""

        if actor_id == "proposer":
            state["extras"]["problem"] = content
            answer = self._solve_problem(content)
            state["extras"]["expected_answer"] = str(answer) if answer is not None else ""

        elif actor_id == "solver":
            state["extras"]["solver_answer"] = content
            expected = state["extras"]["expected_answer"]
            numbers = re.findall(r'-?\d+', content)
            state["extras"]["correct"] = bool(numbers and numbers[0] == expected)

    async def should_stop(self, state: State) -> bool:
        actor_history = state.get("extras", {}).get("actor_history", [])
        proposer_turns = sum(1 for aid, _ in actor_history if aid == "proposer")
        solver_turns = sum(1 for aid, _ in actor_history if aid == "solver")
        return proposer_turns >= 1 and solver_turns >= 1

    async def on_game_end(self, state: State) -> None:
        pass

    # ---- Helpers ----

    @staticmethod
    def _solve_problem(problem: str) -> int | None:
        """Parse a simple arithmetic problem and compute the answer."""
        match = re.search(r'(\d+)\s*([+\-*])\s*(\d+)', problem)
        if not match:
            return None
        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        return None


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(num_examples: int = -1):
    """
    Composition:
        Task   = ProposerSolverTask (problem generation + solving pipeline)
        Agents = {proposer: generator, solver: answerer}
        Env    = MultiAgentEnv(task, agents)
    """
    task = ProposerSolverTask(num_examples=num_examples)

    proposer = Agent(
        id="proposer",
        max_tokens=50,
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
