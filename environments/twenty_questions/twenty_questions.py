"""
20 Questions: A simple multi-agent guessing game.

This environment demonstrates:
- Alternating turns via get_next_actor() (standard turn-based flow)
- Asymmetric actors (one trainable, one frozen)
- Multiple stop conditions (win or max questions)
- Fresh prompts per actor with different context

Game flow:
1. Guesser receives category hint and asks first question
2. Thinker (with secret word) answers yes/no
3. Alternate until guesser wins or runs out of questions
4. Only guesser is trained - rewarded for winning quickly
"""

import re
from datasets import Dataset

from verifiers import Actor, MultiAgentEnv, MultiAgentRubric, Protocol
from verifiers.types import Messages, State
import verifiers as vf


# =============================================================================
# Actors
# =============================================================================

THINKER = Actor(
    id="thinker",
    system_prompt="""You are the Thinker in 20 Questions. You have a SECRET WORD.

Rules:
1. Answer questions with ONLY "Yes" or "No"
2. Be honest and consistent
3. If asked to guess, confirm with "Correct!" or "No, try again"

Format your response as exactly one of:
- Yes
- No
- Correct!
- No, try again""",
    max_tokens=20,
    is_trainable=False,
)

GUESSER = Actor(
    id="guesser",
    system_prompt="""You are the Guesser in 20 Questions. Try to figure out the secret word.

Rules:
1. Ask yes/no questions to narrow down possibilities
2. When ready to guess, say "Is it [your guess]?"
3. You have 20 questions maximum

Good strategy: Start broad (Is it alive? Is it man-made?) then narrow down.

Format: Just ask your question directly.""",
    max_tokens=50,
    is_trainable=True,
)


# =============================================================================
# Environment
# =============================================================================

class TwentyQuestionsEnv(MultiAgentEnv):
    """20 Questions game environment."""

    name = "twenty_questions"
    actors = ["thinker", "guesser"]

    def __init__(self, max_questions: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.max_questions = max_questions

    # -------------------------------------------------------------------------
    # Turn Management
    # -------------------------------------------------------------------------

    def get_initial_actor(self, state: State) -> str:
        return "guesser"

    def get_next_actor(self, state: State) -> str:
        current = state["extras"]["current_actor_id"]
        return "thinker" if current == "guesser" else "guesser"

    # -------------------------------------------------------------------------
    # Stop Conditions
    # -------------------------------------------------------------------------

    @vf.stop
    async def game_won(self, state: State) -> bool:
        return state.get("extras", {}).get("won", False)

    @vf.stop
    async def max_questions_reached(self, state: State) -> bool:
        return state.get("extras", {}).get("question_count", 0) >= self.max_questions

    # -------------------------------------------------------------------------
    # State Setup
    # -------------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        secret_word = state["input"].get("answer", "dog")
        state["extras"]["secret_word"] = secret_word.lower()
        state["extras"]["question_count"] = 0
        state["extras"]["won"] = False
        state["extras"]["questions"] = []
        return state

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        """Build fresh prompt for current actor."""
        secret = state["extras"]["secret_word"]
        category = state["input"].get("info", {}).get("category", "thing")

        # Guesser's turn
        if actor_id == "guesser":
            # First turn: initial prompt
            if len(state["trajectory"]) == 0:
                return [
                    {"role": "system", "content": GUESSER.system_prompt},
                    {"role": "user", "content": f"I'm thinking of a {category}. You have {self.max_questions} questions to guess what it is. Ask your first question!"}
                ]

            # Subsequent turns: show Q&A history
            remaining = self.max_questions - state["extras"]["question_count"]
            history_str = self._build_qa_history(state)

            return [
                {"role": "system", "content": GUESSER.system_prompt},
                {"role": "user", "content": f"I'm thinking of a {category}.\n\nConversation so far:\n{history_str}\n\nYou have {remaining} questions left. Ask another question or make a guess!"}
            ]

        # Thinker's turn: show question with secret word
        else:
            last_question = self._get_last_guesser_response(state)
            question_num = state["extras"]["question_count"] + 1  # Will be incremented after this turn
            return [
                {"role": "system", "content": THINKER.system_prompt + f"\n\nYour secret word is: {secret}"},
                {"role": "user", "content": f"Question {question_num}: {last_question}"}
            ]

    def _build_qa_history(self, state: State) -> str:
        """Build Q&A history string from trajectory."""
        history_lines = []
        current_question = None

        for step in state["trajectory"]:
            actor_id = step.get("extras", {}).get("actor_id")
            completion = step.get("completion", [])
            content = completion[-1].get("content", "") if completion else ""

            if actor_id == "guesser":
                current_question = content
            elif actor_id == "thinker" and current_question:
                history_lines.append(f"Q: {current_question}")
                history_lines.append(f"A: {content}")
                current_question = None

        return "\n".join(history_lines)

    def _get_last_guesser_response(self, state: State) -> str:
        """Get the most recent guesser response from trajectory."""
        for step in reversed(state["trajectory"]):
            if step.get("extras", {}).get("actor_id") == "guesser":
                completion = step.get("completion", [])
                if completion:
                    return completion[-1].get("content", "")
        return ""

    # -------------------------------------------------------------------------
    # Game Logic
    # -------------------------------------------------------------------------

    async def on_turn_complete(self, state: State) -> None:
        """Process game logic after each turn."""
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        last_completion = last_step.get("completion", [])
        if not last_completion:
            return

        content = last_completion[-1].get("content", "") if isinstance(last_completion[-1], dict) else str(last_completion[-1])
        content_lower = content.lower().strip()
        secret = state["extras"]["secret_word"]
        last_actor = last_step.get("extras", {}).get("actor_id", "")

        if last_actor == "guesser":
            # Guesser just asked a question
            state["extras"]["question_count"] += 1
            state["extras"]["questions"].append(content)

            # Check if it's a correct guess
            guess_match = re.search(r"is it (?:a |an )?([a-zA-Z]+)\s*\??$", content_lower)
            if guess_match and guess_match.group(1).lower() == secret:
                state["extras"]["won"] = True
                state["final_env_response"] = [{"role": "user", "content": "Correct! You win!"}]

        else:
            # Thinker just answered
            if "correct" in content_lower:
                state["extras"]["won"] = True
                state["final_env_response"] = [{"role": "user", "content": "Correct! You win!"}]
            elif state["extras"]["question_count"] >= self.max_questions:
                state["final_env_response"] = [{"role": "user", "content": f"Game over! The word was: {secret}"}]


# =============================================================================
# Rubric (Scoring)
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    """Create rubric - guesser rewarded for winning fast."""
    rubric = MultiAgentRubric()

    def guesser_reward(state, **kwargs) -> float:
        extras = state.get("extras", {})
        won = extras.get("won", False)
        questions = extras.get("question_count", 20)
        if won:
            return 1.0 - 0.9 * (questions - 1) / 19
        return 0.0

    def game_length_metric(state, **kwargs) -> float:
        return float(state.get("extras", {}).get("question_count", 0))

    def win_rate_metric(state, **kwargs) -> float:
        return 1.0 if state.get("extras", {}).get("won", False) else 0.0

    rubric.add_actor_reward_func("guesser", guesser_reward, weight=1.0)
    rubric.add_reward_func(game_length_metric, weight=0.0)
    rubric.add_reward_func(win_rate_metric, weight=0.0)

    return rubric


# =============================================================================
# Dataset
# =============================================================================

def create_dataset() -> Dataset:
    """Create dataset of secret words."""
    def make_prompt(category: str) -> list:
        return [
            {"role": "system", "content": GUESSER.system_prompt},
            {"role": "user", "content": f"I'm thinking of a {category}. You have 20 questions to guess what it is. Ask your first question!"}
        ]

    items = [
        {"prompt": make_prompt("animal"), "answer": "dog", "info": {"category": "animal"}, "example_id": 0, "task": "twenty_questions"},
        {"prompt": make_prompt("animal"), "answer": "cat", "info": {"category": "animal"}, "example_id": 1, "task": "twenty_questions"},
        {"prompt": make_prompt("animal"), "answer": "elephant", "info": {"category": "animal"}, "example_id": 2, "task": "twenty_questions"},
        {"prompt": make_prompt("animal"), "answer": "penguin", "info": {"category": "animal"}, "example_id": 3, "task": "twenty_questions"},
        {"prompt": make_prompt("object"), "answer": "chair", "info": {"category": "object"}, "example_id": 4, "task": "twenty_questions"},
        {"prompt": make_prompt("object"), "answer": "book", "info": {"category": "object"}, "example_id": 5, "task": "twenty_questions"},
        {"prompt": make_prompt("object"), "answer": "computer", "info": {"category": "object"}, "example_id": 6, "task": "twenty_questions"},
        {"prompt": make_prompt("object"), "answer": "bicycle", "info": {"category": "object"}, "example_id": 7, "task": "twenty_questions"},
        {"prompt": make_prompt("food"), "answer": "pizza", "info": {"category": "food"}, "example_id": 8, "task": "twenty_questions"},
        {"prompt": make_prompt("food"), "answer": "apple", "info": {"category": "food"}, "example_id": 9, "task": "twenty_questions"},
    ]
    return Dataset.from_list(items)


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    max_questions: int = 20,
    num_examples: int = -1,
) -> TwentyQuestionsEnv:
    """Factory function to create a fully configured 20 Questions environment."""
    dataset = create_dataset()
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    env = TwentyQuestionsEnv(
        max_questions=max_questions,
        rubric=create_rubric(),
        max_turns=max_questions * 2 + 2,
        dataset=dataset,
    )

    Protocol(actors=[THINKER, GUESSER], envs=[env])

    return env
