"""
20 Questions: A simple multi-agent guessing game.

This environment demonstrates:
- Alternating turns via TaskSet.get_next_role() (standard turn-based flow)
- Asymmetric agents (one trainable, one frozen)
- Multiple stop conditions (win or max questions)
- Fresh prompts per role with different context
- Different models per agent (small guesser vs large thinker)

Game flow:
1. Guesser receives category hint and asks first question
2. Thinker (with secret word) answers yes/no
3. Alternate until guesser wins or runs out of questions
4. Only guesser is trained - rewarded for winning quickly
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
# Change these to use different models for each agent.
# Set to None to use the default model from the eval command.
#
# Small models: "olmo3-7b-i", "trinity-mini", "haiku", "gemini-3-flash"
# Large models: "sonnet", "opus", "qwen3-235b-i", "gemini-3-pro"
# =============================================================================

THINKER_ENDPOINT = "qwen3-235b-i"  # Large model answers questions
GUESSER_ENDPOINT = "olmo3-7b-i"   # Small model asks questions

thinker_client, thinker_model = get_actor_client(THINKER_ENDPOINT)
guesser_client, guesser_model = get_actor_client(GUESSER_ENDPOINT)


# =============================================================================
# Rubric (Scoring)
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    """Create rubric - guesser rewarded for winning fast."""
    rubric = MultiAgentRubric()

    def guesser_reward(state, **kwargs) -> float:
        """Read efficiency from extras (computed in on_game_end)."""
        return state.get("extras", {}).get("efficiency", 0.0)

    def game_length_metric(state, **kwargs) -> float:
        return float(state.get("extras", {}).get("question_count", 0))

    def win_rate_metric(state, **kwargs) -> float:
        return 1.0 if state.get("extras", {}).get("won", False) else 0.0

    rubric.add_actor_reward_func("guesser", guesser_reward, weight=1.0)
    rubric.add_reward_func(game_length_metric, weight=0.0)
    rubric.add_reward_func(win_rate_metric, weight=0.0)

    return rubric


# =============================================================================
# TaskSet: the game (dataset + rubric + rules + prompts)
# =============================================================================

class TwentyQuestionsTask(TaskSet):
    """
    20 Questions game definition.

    This TaskSet owns ALL game-specific logic:
    - Dataset of secret words
    - Rubric for scoring
    - Game state (secret word, question count, win flag)
    - Prompts for each role
    - Turn management (guesser → thinker → guesser → ...)
    - Stop conditions (win or max questions)
    """

    def __init__(self, max_questions: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.max_questions = max_questions

    # ---- State Setup ----

    async def setup_state(self, state: State) -> State:
        secret_word = state["input"].get("answer", "dog")
        state["extras"]["secret_word"] = secret_word.lower()
        state["extras"]["question_count"] = 0
        state["extras"]["won"] = False
        state["extras"]["questions"] = []
        return state

    # ---- Prompts ----

    async def build_prompt(self, role: str, state: State) -> Messages:
        secret = state["extras"]["secret_word"]
        category = state["input"].get("info", {}).get("category", "thing")

        if role == "guesser":
            if len(state["trajectory"]) == 0:
                return [
                    {"role": "user", "content": f"I'm thinking of a {category}. You have {self.max_questions} questions to guess what it is. Ask your first question!"}
                ]

            remaining = self.max_questions - state["extras"]["question_count"]
            history_str = self._build_qa_history(state)

            return [
                {"role": "user", "content": f"I'm thinking of a {category}.\n\nConversation so far:\n{history_str}\n\nYou have {remaining} questions left. Ask another question or make a guess!"}
            ]

        else:  # thinker
            last_question = self._get_last_guesser_response(state)
            question_num = state["extras"]["question_count"] + 1
            return [
                {"role": "user", "content": f"Your secret word is: {secret}\n\nQuestion {question_num}: {last_question}"}
            ]

    def _build_qa_history(self, state: State) -> str:
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
        for step in reversed(state["trajectory"]):
            if step.get("extras", {}).get("actor_id") == "guesser":
                completion = step.get("completion", [])
                if completion:
                    return completion[-1].get("content", "")
        return ""

    # ---- Game Logic ----

    async def on_turn_complete(self, state: State) -> None:
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
            state["extras"]["question_count"] += 1
            state["extras"]["questions"].append(content)

            guess_match = re.search(r"is it (?:a |an )?([a-zA-Z]+)\s*\??$", content_lower)
            if guess_match and guess_match.group(1).lower() == secret:
                state["extras"]["won"] = True

        else:  # thinker
            if "correct" in content_lower:
                state["extras"]["won"] = True

    async def should_stop(self, state: State) -> bool:
        if state.get("extras", {}).get("won", False):
            return True
        if state.get("extras", {}).get("question_count", 0) >= self.max_questions:
            return True
        return False

    async def on_game_end(self, state: State) -> None:
        won = state["extras"]["won"]
        questions = state["extras"]["question_count"]

        if won:
            state["extras"]["efficiency"] = 1.0 - 0.9 * (questions - 1) / 19
        else:
            state["extras"]["efficiency"] = 0.0


# =============================================================================
# Dataset
# =============================================================================

def create_dataset() -> Dataset:
    """Create dataset of secret words."""
    items = [
        {"prompt": [{"role": "user", "content": "play"}], "answer": "dog", "info": {"category": "animal"}, "example_id": 0, "task": "twenty_questions"},
        {"prompt": [{"role": "user", "content": "play"}], "answer": "cat", "info": {"category": "animal"}, "example_id": 1, "task": "twenty_questions"},
        {"prompt": [{"role": "user", "content": "play"}], "answer": "elephant", "info": {"category": "animal"}, "example_id": 2, "task": "twenty_questions"},
        {"prompt": [{"role": "user", "content": "play"}], "answer": "penguin", "info": {"category": "animal"}, "example_id": 3, "task": "twenty_questions"},
        {"prompt": [{"role": "user", "content": "play"}], "answer": "chair", "info": {"category": "object"}, "example_id": 4, "task": "twenty_questions"},
        {"prompt": [{"role": "user", "content": "play"}], "answer": "book", "info": {"category": "object"}, "example_id": 5, "task": "twenty_questions"},
        {"prompt": [{"role": "user", "content": "play"}], "answer": "computer", "info": {"category": "object"}, "example_id": 6, "task": "twenty_questions"},
        {"prompt": [{"role": "user", "content": "play"}], "answer": "bicycle", "info": {"category": "object"}, "example_id": 7, "task": "twenty_questions"},
        {"prompt": [{"role": "user", "content": "play"}], "answer": "pizza", "info": {"category": "food"}, "example_id": 8, "task": "twenty_questions"},
        {"prompt": [{"role": "user", "content": "play"}], "answer": "apple", "info": {"category": "food"}, "example_id": 9, "task": "twenty_questions"},
    ]
    return Dataset.from_list(items)


# =============================================================================
# Agents
# =============================================================================

THINKER = Agent(
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
    model=thinker_model,
    client=thinker_client,
)

GUESSER = Agent(
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
    model=guesser_model,
    client=guesser_client,
)


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    max_questions: int = 20,
    num_examples: int = -1,
) -> MultiAgentEnv:
    """Factory function to create a fully configured 20 Questions environment."""
    dataset = create_dataset()
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    task = TwentyQuestionsTask(
        name="twenty_questions",
        dataset=dataset,
        rubric=create_rubric(),
        roles=["guesser", "thinker"],
        max_questions=max_questions,
    )

    agents = {"guesser": GUESSER, "thinker": THINKER}

    env = MultiAgentEnv(
        task=task,
        agents=agents,
        max_turns=max_questions * 2 + 2,
    )

    return env
