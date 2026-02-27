"""
Rock Paper Scissors v2: New decomposition.

Old style: RockPaperScissorsEnv(MultiAgentEnv) — game logic, prompts, state,
           rubric, actors, dataset all in one class.

New style: Game logic in RPSTask(TaskSet). Agents are separate.
           MultiAgentEnv just runs the loop.

What this shows about the abstractions:
    - TaskSet controls information flow (build_prompt hides opponent's choice)
    - Both players are simple Agents — one model call per turn
    - Simple Agents — one model call per turn, no tools needed
    - Turn order: alternating, but task creates simultaneous feel via hidden info
"""

import random

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

BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


# =============================================================================
# Rubric
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    rubric = MultiAgentRubric()

    def player1_reward(state, **kwargs) -> float:
        extras = state.get("extras", {})
        total = extras.get("round", 1)
        return extras.get("p1_score", 0) / total if total > 0 else 0.0

    def player2_reward(state, **kwargs) -> float:
        extras = state.get("extras", {})
        total = extras.get("round", 1)
        return extras.get("p2_score", 0) / total if total > 0 else 0.0

    rubric.add_actor_reward_func("player1", player1_reward, weight=1.0)
    rubric.add_actor_reward_func("player2", player2_reward, weight=1.0)
    return rubric


# =============================================================================
# TaskSet: ALL game logic lives here
# =============================================================================

class RPSTask(TaskSet):
    """
    Rock Paper Scissors — best of N rounds.

    Simultaneous moves via information hiding:
    player1 goes first, player2 goes second, but build_prompt
    hides player1's current-round choice from player2.

    Compare with old RockPaperScissorsEnv:
    - BEFORE: get_active_actors, build_actor_prompt, on_turn_complete,
              setup_state, game_over stop — all in the env subclass
    - AFTER:  All of that is here in the TaskSet. Env is generic.
    """

    def __init__(self, num_rounds: int = 3, num_examples: int = -1):
        dataset = self._create_dataset()
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        super().__init__(
            name="rock_paper_scissors",
            dataset=dataset,
            rubric=create_rubric(),
            roles=["player1", "player2"],
        )
        self.num_rounds = num_rounds

    @staticmethod
    def _create_dataset() -> Dataset:
        return Dataset.from_list([
            {
                "prompt": [{"role": "user", "content": "play"}],
                "answer": "",
                "info": {"seed": i},
                "example_id": i,
                "task": "rock_paper_scissors",
            }
            for i in range(10)
        ])

    # ---- State ----

    async def setup_state(self, state: State) -> State:
        state["extras"]["round"] = 0
        state["extras"]["p1_score"] = 0
        state["extras"]["p2_score"] = 0
        state["extras"]["history"] = []        # [(p1_choice, p2_choice, result), ...]
        state["extras"]["p1_pending"] = None   # player1's choice waiting for player2
        return state

    # ---- Prompts ----

    async def build_prompt(self, role: str, state: State) -> Messages:
        round_num = state["extras"]["round"] + 1
        history = state["extras"]["history"]

        if role == "player1":
            strategy_hint = "You like to play aggressively and switch moves often to keep your opponent guessing."
        else:
            strategy_hint = "You like to play defensively and try to predict what your opponent will do next."

        system = (
            f"You are {role} in Rock Paper Scissors. Best of {self.num_rounds} rounds.\n"
            f"{strategy_hint}\n\n"
            "Respond with EXACTLY one word: rock, paper, or scissors\n"
            "Nothing else. Just the word."
        )

        if not history:
            history_str = "No rounds played yet."
        else:
            lines = []
            for i, (p1, p2, result) in enumerate(history, 1):
                you = p1 if role == "player1" else p2
                opp = p2 if role == "player1" else p1
                lines.append(f"Round {i}: You={you}, Opponent={opp} → {result}")
            history_str = "\n".join(lines)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Round {round_num} of {self.num_rounds}.\n\nHistory:\n{history_str}\n\nMake your choice:"},
        ]

    # ---- Turn Management ----
    # Default alternating (player1 → player2 → player1 → ...) works perfectly.
    # No need to override get_initial_role or get_next_role.

    # ---- Game Logic ----

    async def on_turn_complete(self, state: State) -> None:
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        actor_id = last_step.get("extras", {}).get("actor_id", "")
        choice = self._extract_choice(last_step)

        if actor_id == "player1":
            # Stash choice, wait for player2
            state["extras"]["p1_pending"] = choice

        elif actor_id == "player2":
            # Both moved — resolve
            p1_choice = state["extras"]["p1_pending"]
            p2_choice = choice
            state["extras"]["p1_pending"] = None

            if p1_choice == p2_choice:
                result = "draw"
            elif BEATS.get(p1_choice) == p2_choice:
                result = "player1 wins"
                state["extras"]["p1_score"] += 1
            else:
                result = "player2 wins"
                state["extras"]["p2_score"] += 1

            state["extras"]["history"].append((p1_choice, p2_choice, result))
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

    def _extract_choice(self, step) -> str:
        completion = step.get("completion", [])
        if not completion:
            return "rock"
        text = completion[-1].get("content", "").lower().strip()
        for c in ["rock", "paper", "scissors"]:
            if c in text:
                return c
        return random.choice(["rock", "paper", "scissors"])


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(num_rounds: int = 3, num_examples: int = -1):
    """
    Composition:
        Task   = RPSTask (rules + prompts + scoring)
        Agents = {player1, player2} with SingleTurnHarness
        Env    = MultiAgentEnv(task, agents)
    """
    task = RPSTask(num_rounds=num_rounds, num_examples=num_examples)

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
