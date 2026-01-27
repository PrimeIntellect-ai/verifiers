"""
20 Questions: A simple multi-agent guessing game.

This environment demonstrates:
- Alternating turns via get_next_actor() (standard turn-based flow)
- Asymmetric actors (one trainable, one frozen)
- Multiple stop conditions (win or max questions)
- Conversation relay between actors via env_response()

Game flow:
1. Guesser receives category hint and asks first question
2. Thinker (with secret word) answers yes/no
3. Alternate until guesser wins or runs out of questions
4. Only guesser is trained - rewarded for winning quickly
"""

import re
from datasets import Dataset

import verifiers as vf
from verifiers import Actor, MultiAgentEnv, MultiAgentRubric, Protocol
from verifiers.types import Messages, State


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
    is_trainable=False,  # Frozen - just follows rules, not trained
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
    is_trainable=True,  # This is the agent we're training
)


# =============================================================================
# Environment
# =============================================================================

class TwentyQuestionsEnv(MultiAgentEnv):
    """
    20 Questions game environment.

    Uses standard alternating turn flow (unlike RPS which uses simultaneous moves).
    The Guesser asks questions, Thinker answers, until win or question limit.
    """

    name = "twenty_questions"
    actors = ["thinker", "guesser"]

    def __init__(self, max_questions: int = 20, **kwargs):
        """
        Initialize environment.

        Args:
            max_questions: Maximum questions before game ends (default 20)
            **kwargs: Passed to parent (rubric, dataset, max_turns, etc.)
        """
        super().__init__(**kwargs)
        self.max_questions = max_questions

    # -------------------------------------------------------------------------
    # Turn Management
    # Uses standard alternating flow - no custom rollout needed
    # -------------------------------------------------------------------------

    def get_initial_actor(self, state: State) -> str:
        """Guesser asks first question."""
        return "guesser"

    def get_next_actor(self, state: State) -> str:
        """Alternate between guesser and thinker each turn."""
        current = state["extras"]["current_actor_id"]
        return "thinker" if current == "guesser" else "guesser"

    # -------------------------------------------------------------------------
    # Stop Conditions
    # Two ways to end: win or run out of questions
    # -------------------------------------------------------------------------

    @vf.stop
    async def game_won(self, state: State) -> bool:
        """Stop if guesser guessed correctly."""
        return state.get("extras", {}).get("won", False)

    @vf.stop
    async def max_questions_reached(self, state: State) -> bool:
        """Stop after max questions (guesser loses)."""
        return state.get("extras", {}).get("question_count", 0) >= self.max_questions

    # -------------------------------------------------------------------------
    # State Setup
    # -------------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        """
        Initialize game state with secret word from dataset.

        Sets up:
        - secret_word: The word guesser must discover
        - question_count: Progress toward limit
        - won: Victory flag (checked by game_won stop condition)
        - questions: History for debugging/analysis
        """
        state = await super().setup_state(state)

        # Get secret word from dataset's "answer" field
        secret_word = state["input"].get("answer", "dog")
        state["extras"]["secret_word"] = secret_word.lower()
        state["extras"]["question_count"] = 0
        state["extras"]["won"] = False
        state["extras"]["questions"] = []

        return state

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    async def get_initial_messages(self, state: State) -> Messages:
        """
        Create opening prompt for guesser.

        Includes category hint (animal/object/food) from dataset
        to make the game tractable.
        """
        category = state["input"].get("info", {}).get("category", "thing")
        return [
            {"role": "system", "content": GUESSER.system_prompt},
            {"role": "user", "content": f"I'm thinking of a {category}. Ask your first question!"}
        ]

    # -------------------------------------------------------------------------
    # Environment Response (Game Logic)
    # -------------------------------------------------------------------------

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        """
        Process last response and build prompt for next actor.

        Called after each actor speaks:
        - After Guesser: Check for correct guess, prompt Thinker to answer
        - After Thinker: Relay answer to Guesser, check for game over

        Returns the messages to show the next actor.
        """
        if not state["trajectory"]:
            return []

        # Extract what was just said
        last_step = state["trajectory"][-1]
        last_completion = last_step.get("completion", [])
        if not last_completion:
            return []

        content = last_completion[-1].get("content", "") if isinstance(last_completion[-1], dict) else str(last_completion[-1])
        content_lower = content.lower().strip()
        secret = state["extras"]["secret_word"]

        # Get who just spoke (from trajectory, not current_actor which is NEXT speaker)
        last_actor = last_step.get("extras", {}).get("actor_id", "")

        if last_actor == "guesser":
            # -----------------------------------------------------------------
            # Guesser just asked a question
            # -----------------------------------------------------------------
            state["extras"]["question_count"] += 1
            state["extras"]["questions"].append(content)

            # Check if it's a guess - look for "is it [something]?" patterns
            guess_match = re.search(r"is it .*?([a-zA-Z]+)\s*\??$", content_lower)
            if guess_match:
                guess = guess_match.group(1).lower()
                # Check exact match or if secret appears anywhere in question
                if guess == secret or secret in content_lower:
                    state["extras"]["won"] = True
                    state["final_env_response"] = [{"role": "user", "content": "Correct! You win!"}]
                    return []

            # Build prompt for Thinker (inject secret word so it can answer)
            return [
                {"role": "system", "content": THINKER.system_prompt + f"\n\nYour secret word is: {secret}"},
                {"role": "user", "content": f"Question {state['extras']['question_count']}: {content}"}
            ]

        else:
            # -----------------------------------------------------------------
            # Thinker just answered
            # -----------------------------------------------------------------
            # Safety check if already won
            if state["extras"].get("won", False):
                return []

            # Check if Thinker confirmed a correct guess
            if "correct" in content_lower:
                state["extras"]["won"] = True
                state["final_env_response"] = [{"role": "user", "content": "Correct! You win!"}]
                return []

            # Check if max questions reached (game over, guesser loses)
            if state["extras"]["question_count"] >= self.max_questions:
                state["final_env_response"] = [
                    {"role": "user", "content": f"Game over! The word was: {secret}"}
                ]
                return []

            # Relay answer to Guesser with remaining count
            remaining = self.max_questions - state["extras"]["question_count"]
            return [
                {"role": "user", "content": f"Answer: {content}\n\nYou have {remaining} questions left. Ask another question or make a guess!"}
            ]

    # -------------------------------------------------------------------------
    # Rollout
    # -------------------------------------------------------------------------

    async def rollout(self, input, client, model, sampling_args=None) -> State:
        """
        Run the game and split into per-actor states.

        Uses parent's standard alternating rollout (no custom loop needed).
        Just adds per-actor state splitting at the end for scoring.
        """
        # Use parent rollout (handles turn alternation, stop conditions)
        state = await super().rollout(input, client, model, sampling_args)

        # Split into per-actor states for proper per-actor scoring
        state["child_states"] = self.create_actor_states(state)

        return state


# =============================================================================
# Rubric (Scoring)
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    """
    Create rubric - only guesser is trained, rewarded for winning fast.

    Reward structure:
    - Win in 1 question: 1.0 (maximum)
    - Win in 20 questions: 0.1 (minimum win reward)
    - Lose: 0.0

    Thinker has is_trainable=False, so no reward needed for it.
    """
    rubric = MultiAgentRubric()

    def guesser_reward(state, **kwargs) -> float:
        """Reward guesser for winning quickly. Faster = higher reward."""
        extras = state.get("extras", {})
        won = extras.get("won", False)
        questions = extras.get("question_count", 20)
        max_q = 20

        if won:
            # Linear scale from 1.0 (1 question) to 0.1 (20 questions)
            return 1.0 - 0.9 * (questions - 1) / (max_q - 1)
        else:
            return 0.0

    def game_length_metric(state, **kwargs) -> float:
        """Track how many questions were asked (metric only, not trained on)."""
        return float(state.get("extras", {}).get("question_count", 0))

    def win_rate_metric(state, **kwargs) -> float:
        """Track win rate (metric only, not trained on)."""
        return 1.0 if state.get("extras", {}).get("won", False) else 0.0

    # Guesser reward - the only trainable actor
    rubric.add_actor_reward_func("guesser", guesser_reward, weight=1.0)
    # Thinker is frozen (is_trainable=False), no reward function needed

    # Metrics for logging (weight=0.0 means not used in training)
    rubric.add_reward_func(game_length_metric, weight=0.0)
    rubric.add_reward_func(win_rate_metric, weight=0.0)

    return rubric


# =============================================================================
# Dataset
# =============================================================================

def create_dataset() -> Dataset:
    """
    Create dataset of secret words across categories.

    Each row contains:
    - prompt: Initial messages for Guesser
    - answer: The secret word (used in setup_state)
    - info.category: Hint category (animal/object/food)
    - example_id: Unique identifier
    - task: Environment name for routing
    """
    def make_prompt(category: str) -> list:
        return [
            {"role": "system", "content": GUESSER.system_prompt},
            {"role": "user", "content": f"I'm thinking of a {category}. You have 20 questions to guess what it is. Ask your first question!"}
        ]

    items = [
        # Animals (4)
        {"prompt": make_prompt("animal"), "answer": "dog", "info": {"category": "animal"}, "example_id": 0, "task": "twenty_questions"},
        {"prompt": make_prompt("animal"), "answer": "cat", "info": {"category": "animal"}, "example_id": 1, "task": "twenty_questions"},
        {"prompt": make_prompt("animal"), "answer": "elephant", "info": {"category": "animal"}, "example_id": 2, "task": "twenty_questions"},
        {"prompt": make_prompt("animal"), "answer": "penguin", "info": {"category": "animal"}, "example_id": 3, "task": "twenty_questions"},
        # Objects (4)
        {"prompt": make_prompt("object"), "answer": "chair", "info": {"category": "object"}, "example_id": 4, "task": "twenty_questions"},
        {"prompt": make_prompt("object"), "answer": "book", "info": {"category": "object"}, "example_id": 5, "task": "twenty_questions"},
        {"prompt": make_prompt("object"), "answer": "computer", "info": {"category": "object"}, "example_id": 6, "task": "twenty_questions"},
        {"prompt": make_prompt("object"), "answer": "bicycle", "info": {"category": "object"}, "example_id": 7, "task": "twenty_questions"},
        # Food (2)
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
    """
    Factory function to create a fully configured 20 Questions environment.

    Args:
        max_questions: Questions before game ends (default 20)
        num_examples: Number of games to run (-1 = all 10)

    Returns:
        Ready-to-use TwentyQuestionsEnv
    """
    dataset = create_dataset()
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    rubric = create_rubric()

    env = TwentyQuestionsEnv(
        max_questions=max_questions,
        rubric=rubric,
        max_turns=max_questions * 2 + 2,  # 2 turns per question + buffer
        dataset=dataset,
    )

    # Wire actors to environment via Protocol
    # Protocol constructor registers itself with env (env.protocol = protocol)
    # This enables env.get_actor() and env.protocol.spawn()
    Protocol(
        actors=[THINKER, GUESSER],
        envs=[env],
    )

    return env
