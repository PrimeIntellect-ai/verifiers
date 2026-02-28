"""
Prisoner's Dilemma: Asymmetric payoffs with masked actions.

Demonstrates per-actor GRPO advantages for asymmetric games.
Actions are masked with random strings to remove pretraining bias —
the model must learn purely from reward signal.

Asymmetric payoff matrix (default):
              P2: A    P2: B
    P1: A     3, 4     0, 2
    P1: B     5, 1     1, 0

Player1: B dominates (5>3, 1>0) → should converge to B
Player2: A dominates (4>1, 2>0) → should converge to A

With per-actor GRPO, each player gets its own baseline,
so asymmetric optimal strategies emerge cleanly.
"""

import random
import string

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

PLAYER1_ENDPOINT = None
PLAYER2_ENDPOINT = None

p1_client, p1_model = get_actor_client(PLAYER1_ENDPOINT)
p2_client, p2_model = get_actor_client(PLAYER2_ENDPOINT)

# Payoff matrices: payoffs[action_idx_p1][action_idx_p2] = (p1_reward, p2_reward)
# Action A = index 0, Action B = index 1
PAYOFF_MATRICES = {
    "asymmetric": [
        [(3, 4), (0, 2)],  # P1 picks A: (P2 picks A → 3,4), (P2 picks B → 0,2)
        [(5, 1), (1, 0)],  # P1 picks B: (P2 picks A → 5,1), (P2 picks B → 1,0)
    ],
    "standard": [
        [(3, 3), (0, 5)],  # P1 picks A: (P2 picks A → 3,3), (P2 picks B → 0,5)
        [(5, 0), (1, 1)],  # P1 picks B: (P2 picks A → 5,0), (P2 picks B → 1,1)
    ],
}

MAX_PAYOFF = 5  # For normalizing rewards to [0, 1]


# =============================================================================
# Rubric
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    rubric = MultiAgentRubric()

    def player1_reward(state, **kwargs) -> float:
        extras = state.get("extras", {})
        total = extras.get("round", 1)
        return extras.get("p1_score", 0) / (total * MAX_PAYOFF) if total > 0 else 0.0

    def player2_reward(state, **kwargs) -> float:
        extras = state.get("extras", {})
        total = extras.get("round", 1)
        return extras.get("p2_score", 0) / (total * MAX_PAYOFF) if total > 0 else 0.0

    rubric.add_actor_reward_func("player1", player1_reward, weight=1.0)
    rubric.add_actor_reward_func("player2", player2_reward, weight=1.0)
    return rubric


# =============================================================================
# TaskSet
# =============================================================================

def _random_label() -> str:
    """Generate a random 4-letter uppercase string."""
    return "".join(random.choices(string.ascii_uppercase, k=4))


class PDTask(TaskSet):
    """
    Prisoner's Dilemma with masked actions and configurable payoff matrix.

    Simultaneous moves via information hiding (same pattern as RPS):
    player1 goes first, player2 goes second, but build_prompt
    hides player1's current-round choice from player2.
    """

    def __init__(
        self,
        num_rounds: int = 5,
        payoff_type: str = "asymmetric",
        num_examples: int = -1,
    ):
        dataset = self._create_dataset()
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        super().__init__(
            name="prisoners_dilemma",
            dataset=dataset,
            rubric=create_rubric(),
            roles=["player1", "player2"],
        )
        self.num_rounds = num_rounds
        self.payoff_matrix = PAYOFF_MATRICES[payoff_type]

    @staticmethod
    def _create_dataset() -> Dataset:
        return Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": "play"}],
                "answer": "",
                "info": {"seed": i},
                "example_id": i,
                "task": "prisoners_dilemma",
            }
            for i in range(10)
        ])

    # ---- State ----

    async def setup_state(self, state: State) -> State:
        # Generate masked action names for this game
        label_a = _random_label()
        label_b = _random_label()
        while label_b == label_a:
            label_b = _random_label()

        state["extras"]["label_a"] = label_a
        state["extras"]["label_b"] = label_b
        state["extras"]["round"] = 0
        state["extras"]["p1_score"] = 0
        state["extras"]["p2_score"] = 0
        state["extras"]["history"] = []
        state["extras"]["p1_pending"] = None
        return state

    # ---- Prompts ----

    async def build_prompt(self, role: str, state: State) -> Messages:
        round_num = state["extras"]["round"] + 1
        history = state["extras"]["history"]
        label_a = state["extras"]["label_a"]
        label_b = state["extras"]["label_b"]

        system = (
            f"You are {role} in a two-player game. Best of {self.num_rounds} rounds.\n"
            f"Each round, choose one of two actions: {label_a} or {label_b}\n"
            f"Your goal is to maximize your own score.\n\n"
            f"Respond with EXACTLY one word: {label_a} or {label_b}\n"
            f"Nothing else. Just the action name."
        )

        if not history:
            history_str = "No rounds played yet."
        else:
            lines = []
            for i, entry in enumerate(history, 1):
                you = entry["p1"] if role == "player1" else entry["p2"]
                opp = entry["p2"] if role == "player1" else entry["p1"]
                your_pts = entry["p1_pts"] if role == "player1" else entry["p2_pts"]
                opp_pts = entry["p2_pts"] if role == "player1" else entry["p1_pts"]
                lines.append(
                    f"Round {i}: You={you}, Opponent={opp} → You got {your_pts}, Opponent got {opp_pts}"
                )
            history_str = "\n".join(lines)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"Round {round_num} of {self.num_rounds}.\n\n"
                f"History:\n{history_str}\n\n"
                f"Choose your action:"
            )},
        ]

    # ---- Game Logic ----

    async def on_turn_complete(self, state: State) -> None:
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        actor_id = last_step["extras"].get("actor_id", "")
        choice = self._extract_choice(last_step, state)

        if actor_id == "player1":
            state["extras"]["p1_pending"] = choice

        elif actor_id == "player2":
            p1_action = state["extras"]["p1_pending"]
            p2_action = choice
            state["extras"]["p1_pending"] = None

            p1_pts, p2_pts = self.payoff_matrix[p1_action][p2_action]

            state["extras"]["p1_score"] += p1_pts
            state["extras"]["p2_score"] += p2_pts

            label_a = state["extras"]["label_a"]
            label_b = state["extras"]["label_b"]
            p1_label = label_a if p1_action == 0 else label_b
            p2_label = label_a if p2_action == 0 else label_b

            state["extras"]["history"].append({
                "p1": p1_label, "p2": p2_label,
                "p1_pts": p1_pts, "p2_pts": p2_pts,
            })
            state["extras"]["round"] += 1

    async def should_stop(self, state: State) -> bool:
        return state["extras"].get("round", 0) >= self.num_rounds

    async def on_game_end(self, state: State) -> None:
        p1 = state["extras"]["p1_score"]
        p2 = state["extras"]["p2_score"]
        if p1 > p2:
            state["extras"]["winner"] = "player1"
        elif p2 > p1:
            state["extras"]["winner"] = "player2"
        else:
            state["extras"]["winner"] = "tie"

    # ---- Helpers ----

    def _extract_choice(self, step, state: State) -> int:
        """Extract action index (0=A, 1=B) from model response."""
        completion = step.get("completion", [])
        if not completion:
            return random.randint(0, 1)
        text = completion[-1].get("content", "").strip().upper()
        label_a = state["extras"]["label_a"]
        label_b = state["extras"]["label_b"]
        if label_a in text and label_b not in text:
            return 0
        if label_b in text and label_a not in text:
            return 1
        return random.randint(0, 1)


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    num_rounds: int = 5,
    payoff_type: str = "asymmetric",
    num_examples: int = -1,
):
    task = PDTask(
        num_rounds=num_rounds,
        payoff_type=payoff_type,
        num_examples=num_examples,
    )

    player1 = Agent(
        id="player1",
        max_tokens=20,
        is_trainable=True,
        model=p1_model,
        client=p1_client,
    )

    player2 = Agent(
        id="player2",
        max_tokens=20,
        is_trainable=True,
        model=p2_model,
        client=p2_client,
    )

    return MultiAgentEnv(
        task=task,
        agents={"player1": player1, "player2": player2},
        max_turns=num_rounds * 2 + 2,
    )
