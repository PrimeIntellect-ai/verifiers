"""
Proposer + Code Solver: Demo for MultiAgentStatefulToolEnv + TaskSet.

Shows multi-agent + tools + hidden state injection, using the TaskSet pattern.

Two roles:
    proposer — sees a math problem, proposes a strategy (no tools)
    solver   — sees problem + strategy, uses run_code(code) tool to compute answer

What this demonstrates:
    - TaskSet owns game logic (prompts, turn order, scoring)
    - MultiAgentStatefulToolEnv adds tool layer on top
    - Hidden args: run_code has a hidden "state" param injected via update_tool_args
    - Same-actor continuation: solver keeps calling run_code until it responds without tools
    - Asymmetric tool access: proposer has no tools, solver has tools


Compare with proposer_solver_v2 (no tools):
    - BEFORE: ProposerSolverTask + MultiAgentEnv — solver outputs answer directly
    - AFTER:  ProposerCodeSolverTask + ProposerCodeSolverEnv — solver runs code to compute
"""

import io
import re
import traceback
from contextlib import redirect_stdout

from datasets import Dataset

from verifiers.agent import Agent
from verifiers.envs.multiagent_stateful_tool_env import MultiAgentStatefulToolEnv
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
# TaskSet: game logic
# =============================================================================

class ProposerCodeSolverTask(TaskSet):
    """
    2-role pipeline with code execution.

    Turn 1 (proposer): Sees problem, proposes a step-by-step strategy.
    Turn 2+ (solver):  Sees problem + strategy, uses run_code() tool to
                       execute Python and compute the answer. May call
                       run_code multiple times (same-actor continuation).
    Evaluate:          Check if solver's final answer matches expected.
    """

    def __init__(self, num_examples: int = -1):
        dataset = self._create_dataset()
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        super().__init__(
            name="proposer_code_solver",
            dataset=dataset,
            rubric=create_rubric(),
            roles=["proposer", "solver"],
        )

    @staticmethod
    def _create_dataset() -> Dataset:
        """Math problems that benefit from code execution."""
        problems = [
            ("What is 17 * 28 + 53?", "529"),
            ("What is the sum of all integers from 1 to 100?", "5050"),
            ("What is 2 to the power of 10?", "1024"),
            ("What is 143 * 67 - 256?", "9325"),
            ("What is the factorial of 7?", "5040"),
            ("What is 999 * 999?", "998001"),
            ("What is 12345 + 67890?", "80235"),
            ("How many prime numbers are there below 50?", "15"),
            ("What is 256 divided by 16?", "16"),
            ("What is 3 * 3 * 3 * 3 * 3?", "243"),
        ]
        return Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": problem}],
                "answer": answer,
                "info": {"problem": problem},
                "example_id": i,
                "task": "proposer_code_solver",
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
        state["extras"]["code_history"] = []
        return state

    # ---- Prompts ----

    async def build_prompt(self, role: str, state: State) -> Messages:
        problem = state["extras"]["problem"]

        if role == "proposer":
            return [
                {"role": "system", "content": (
                    "You are a Math Strategist. Given a math problem, propose a clear "
                    "step-by-step strategy for solving it using Python code.\n\n"
                    "Be specific about the computation steps needed. "
                    "Output ONLY the strategy, nothing else."
                )},
                {"role": "user", "content": f"Problem: {problem}\n\nPropose a solution strategy:"},
            ]

        else:  # solver
            strategy = state["extras"]["strategy"]
            return [
                {"role": "system", "content": (
                    "You are a Math Solver with access to a Python REPL.\n"
                    "You are given a problem and a strategy.\n\n"
                    "Use the run_code tool to execute Python code and compute the answer.\n"
                    "When you have the answer, respond with ONLY the numeric answer.\n"
                    "Nothing else. Just the number."
                )},
                {"role": "user", "content": (
                    f"Problem: {problem}\n\n"
                    f"Strategy: {strategy}\n\n"
                    f"Use run_code to compute the answer, then respond with just the number."
                )},
            ]

    # ---- Game Logic ----

    async def on_turn_complete(self, state: State) -> None:
        """Called only on non-tool turns (MultiAgentStatefulToolEnv guards this)."""
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        actor_id = last_step.get("extras", {}).get("actor_id", "")
        completion = last_step.get("completion", [])
        if not completion:
            return
        content = completion[-1].get("content", "") or ""

        if actor_id == "proposer":
            state["extras"]["strategy"] = content

        elif actor_id == "solver":
            state["extras"]["solver_answer"] = content
            expected = state["extras"]["expected_answer"]
            numbers = re.findall(r'-?\d+', content)
            state["extras"]["correct"] = bool(numbers and numbers[0] == expected)

    async def should_stop(self, state: State) -> bool:
        # Stop after both proposer and solver have completed their turns.
        # Tool loops are handled by MultiAgentEnv.rollout() — should_stop
        # only runs after the tool loop exits (actor's final response).
        actor_history = state.get("extras", {}).get("actor_history", [])
        proposer_turns = sum(1 for aid, _ in actor_history if aid == "proposer")
        solver_turns = sum(1 for aid, _ in actor_history if aid == "solver")
        return proposer_turns >= 1 and solver_turns >= 1

    async def on_game_end(self, state: State) -> None:
        pass


# =============================================================================
# Environment: adds tool layer
# =============================================================================

class ProposerCodeSolverEnv(MultiAgentStatefulToolEnv):
    """
    MultiAgentStatefulToolEnv subclass that registers the run_code tool.

    All game logic lives in ProposerCodeSolverTask (the TaskSet).
    This class only adds:
    - Tool registration (run_code with hidden state arg)
    - Hidden arg injection (update_tool_args)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_tool(self.run_code, args_to_skip=["state"])

    # ---- Tool ----

    async def run_code(self, code: str, state: State) -> str:
        """
        Execute Python code and return the output.

        Use this to compute math results. Print the answer to see it.

        Args:
            code: Python code to execute
        """
        stdout_buf = io.StringIO()
        namespace = {"__builtins__": __builtins__}

        try:
            with redirect_stdout(stdout_buf):
                exec(code, namespace)
        except Exception:
            return f"Error:\n{traceback.format_exc()}"

        output = stdout_buf.getvalue()
        state["extras"].setdefault("code_history", []).append({
            "code": code,
            "output": output or "(no output)",
        })

        return output or "(no output)"

    # ---- Hidden Arg Injection ----

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: State,
    ) -> dict:
        if tool_name == "run_code":
            return {**tool_args, "state": state}
        return tool_args


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(num_examples: int = -1):
    """
    Composition:
        Task   = ProposerCodeSolverTask (math problems + strategy→code→solve pipeline)
        Agents = {proposer: strategist, solver: code executor}
        Env    = ProposerCodeSolverEnv(MultiAgentStatefulToolEnv)
    """
    task = ProposerCodeSolverTask(num_examples=num_examples)

    proposer = Agent(
        id="proposer",
        max_tokens=200,
        is_trainable=True,
        model=proposer_model,
        client=proposer_client,
    )

    solver = Agent(
        id="solver",
        max_tokens=200,
        is_trainable=True,

        model=solver_model,
        client=solver_client,
    )

    return ProposerCodeSolverEnv(
        task=task,
        agents={"proposer": proposer, "solver": solver},
        max_turns=10,  # proposer(1) + solver(up to ~4 tool rounds + final)
    )
