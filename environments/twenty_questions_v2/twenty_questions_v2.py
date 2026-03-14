"""
20 Questions v2: Clean decomposition.

Three abstractions:
    TwentyQuestionsTask (TaskSet)   — the game: dataset, rubric, rules, prompts
    SingleTurnHarness               — how agents respond: one model call per turn
    Agent                           — who responds: model config + capabilities

These feed into MultiAgentEnv, which runs the game loop.

    Model + Harness = Agent
    MultiAgentEnv(task, agents) = runnable environment

Compare with original twenty_questions.py:
- BEFORE: TwentyQuestionsEnv(MultiAgentEnv) has game logic, prompt building,
          stop conditions, rubric, dataset — everything in one class
- AFTER:  Game logic lives in TwentyQuestionsTask (the TaskSet).
          Agents are self-contained (model + config).
          MultiAgentEnv just runs the loop.
"""

import re

from datasets import Dataset

from verifiers.envs.agent import Agent
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.envs.taskset import TaskSet
from verifiers.types import Messages, State
from verifiers.utils.client_utils import get_actor_client


# =============================================================================
# Model Configuration
# =============================================================================

THINKER_ENDPOINT = None
GUESSER_ENDPOINT = None

thinker_client, thinker_model = get_actor_client(THINKER_ENDPOINT)
guesser_client, guesser_model = get_actor_client(GUESSER_ENDPOINT)


# =============================================================================
# Rubric
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    """Guesser rewarded for winning fast."""
    rubric = MultiAgentRubric()

    def guesser_reward(state, **kwargs) -> float:
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
    - Prompt building (what each role sees)
    - Turn logic (track questions, check guesses)
    - Stop conditions (win or max questions)
    - Final metrics (efficiency score)
    """

    def __init__(self, max_questions: int = 20, num_examples: int = -1):
        dataset = self._create_dataset()
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        super().__init__(
            name="twenty_questions_v2",
            dataset=dataset,
            rubric=create_rubric(),
            roles=["guesser", "thinker"],
        )
        self.max_questions = max_questions

    @staticmethod
    def _create_dataset() -> Dataset:
        def make_prompt(category: str) -> list:
            return [{"role": "user", "content": f"I'm thinking of a {category}. You have 20 questions to guess what it is. Ask your first question!"}]

        items = [
            {"prompt": make_prompt("animal"), "answer": "dog", "info": {"category": "animal"}, "example_id": 0, "task": "twenty_questions_v2"},
            {"prompt": make_prompt("animal"), "answer": "cat", "info": {"category": "animal"}, "example_id": 1, "task": "twenty_questions_v2"},
            {"prompt": make_prompt("animal"), "answer": "elephant", "info": {"category": "animal"}, "example_id": 2, "task": "twenty_questions_v2"},
            {"prompt": make_prompt("animal"), "answer": "penguin", "info": {"category": "animal"}, "example_id": 3, "task": "twenty_questions_v2"},
            {"prompt": make_prompt("object"), "answer": "chair", "info": {"category": "object"}, "example_id": 4, "task": "twenty_questions_v2"},
            {"prompt": make_prompt("object"), "answer": "book", "info": {"category": "object"}, "example_id": 5, "task": "twenty_questions_v2"},
            {"prompt": make_prompt("object"), "answer": "computer", "info": {"category": "object"}, "example_id": 6, "task": "twenty_questions_v2"},
            {"prompt": make_prompt("object"), "answer": "bicycle", "info": {"category": "object"}, "example_id": 7, "task": "twenty_questions_v2"},
            {"prompt": make_prompt("food"), "answer": "pizza", "info": {"category": "food"}, "example_id": 8, "task": "twenty_questions_v2"},
            {"prompt": make_prompt("food"), "answer": "apple", "info": {"category": "food"}, "example_id": 9, "task": "twenty_questions_v2"},
        ]
        return Dataset.from_list(items)

    # -------------------------------------------------------------------------
    # Game State
    # -------------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        secret_word = state["input"].get("answer", "dog")
        state["extras"]["secret_word"] = secret_word.lower()
        state["extras"]["question_count"] = 0
        state["extras"]["won"] = False
        state["extras"]["questions"] = []
        return state

    # -------------------------------------------------------------------------
    # Prompts: what each role sees
    # -------------------------------------------------------------------------

    async def build_prompt(self, role: str, state: State) -> Messages:
        secret = state["extras"]["secret_word"]
        category = state["input"].get("info", {}).get("category", "thing")

        if role == "guesser":
            return self._build_guesser_prompt(state, category)
        else:
            return self._build_thinker_prompt(state, secret)

    def _build_guesser_prompt(self, state: State, category: str) -> Messages:
        guesser_system = (
            "You are the Guesser in 20 Questions. Try to figure out the secret word.\n\n"
            "Rules:\n"
            "1. Ask yes/no questions to narrow down possibilities\n"
            "2. When ready to guess, say \"Is it [your guess]?\"\n"
            "3. You have 20 questions maximum\n\n"
            "Good strategy: Start broad (Is it alive? Is it man-made?) then narrow down.\n\n"
            "Format: Just ask your question directly."
        )

        if len(state["trajectory"]) == 0:
            return [
                {"role": "system", "content": guesser_system},
                {"role": "user", "content": f"/no_think I'm thinking of a {category}. You have {self.max_questions} questions to guess what it is. Ask your first question!"},
            ]

        remaining = self.max_questions - state["extras"]["question_count"]
        history_str = self._build_qa_history(state)
        return [
            {"role": "system", "content": guesser_system},
            {"role": "user", "content": f"/no_think I'm thinking of a {category}.\n\nConversation so far:\n{history_str}\n\nYou have {remaining} questions left. Ask another question or make a guess!"},
        ]

    def _build_thinker_prompt(self, state: State, secret: str) -> Messages:
        thinker_system = (
            "You are the Thinker in 20 Questions. You have a SECRET WORD.\n\n"
            "Rules:\n"
            "1. Answer questions with ONLY \"Yes\" or \"No\"\n"
            "2. Be honest and consistent\n"
            "3. If asked to guess, confirm with \"Correct!\" or \"No, try again\"\n\n"
            "Format your response as exactly one of:\n"
            "- Yes\n"
            "- No\n"
            "- Correct!\n"
            "- No, try again"
        )

        last_question = self._get_last_response(state, "guesser")
        question_num = state["extras"]["question_count"] + 1
        return [
            {"role": "system", "content": thinker_system + f"\n\nYour secret word is: {secret}"},
            {"role": "user", "content": f"/no_think Question {question_num}: {last_question}"},
        ]

    # -------------------------------------------------------------------------
    # Game Logic
    # -------------------------------------------------------------------------

    async def on_turn_complete(self, state: State) -> None:
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        last_completion = last_step.get("completion", [])
        if not last_completion:
            return

        content = last_completion[-1].get("content", "")
        content_lower = content.lower().strip()
        secret = state["extras"]["secret_word"]
        last_actor = last_step.get("extras", {}).get("actor_id", "")

        if last_actor == "guesser":
            state["extras"]["question_count"] += 1
            state["extras"]["questions"].append(content)

            guess_match = re.search(r"is it (?:a |an )?([a-zA-Z]+)\s*\??$", content_lower)
            if guess_match and guess_match.group(1).lower() == secret:
                state["extras"]["won"] = True
                state["final_env_response"] = [{"role": "user", "content": "Correct! You win!"}]

        else:  # thinker
            if "correct" in content_lower:
                state["extras"]["won"] = True
                state["final_env_response"] = [{"role": "user", "content": "Correct! You win!"}]
            elif state["extras"]["question_count"] >= self.max_questions:
                state["final_env_response"] = [{"role": "user", "content": f"Game over! The word was: {secret}"}]

    async def should_stop(self, state: State) -> bool:
        extras = state.get("extras", {})
        return extras.get("won", False) or extras.get("question_count", 0) >= self.max_questions

    async def on_game_end(self, state: State) -> None:
        won = state["extras"]["won"]
        questions = state["extras"]["question_count"]
        if won:
            state["extras"]["efficiency"] = 1.0 - 0.9 * (questions - 1) / 19
        else:
            state["extras"]["efficiency"] = 0.0

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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

    def _get_last_response(self, state: State, role: str) -> str:
        for step in reversed(state["trajectory"]):
            if step.get("extras", {}).get("actor_id") == role:
                completion = step.get("completion", [])
                if completion:
                    return completion[-1].get("content", "")
        return ""


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    max_questions: int = 20,
    num_examples: int = -1,
):
    """
    Factory function for vf-eval.

    Composition:
        Task   = TwentyQuestionsTask (dataset + rubric + game rules)
        Agents = {guesser: Agent(SingleTurnHarness), thinker: Agent(SingleTurnHarness)}
        Env    = MultiAgentEnv(task, agents)
    """

    task = TwentyQuestionsTask(
        max_questions=max_questions,
        num_examples=num_examples,
    )

    guesser = Agent(
        id="guesser",
        system_prompt="",
        max_tokens=128,
        is_trainable=True,
        model=guesser_model,
        client=guesser_client,
    )

    thinker = Agent(
        id="thinker",
        system_prompt="",
        max_tokens=32,
        is_trainable=False,
        model=thinker_model,
        client=thinker_client,
    )

    return MultiAgentEnv(
        task=task,
        agents={"guesser": guesser, "thinker": thinker},
        max_turns=max_questions * 2 + 2,
    )
