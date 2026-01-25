"""
Rock-Paper-Scissors: Multi-agent environment with simultaneous moves.

This environment demonstrates:
- Simultaneous moves via get_active_actors() returning both players
- Custom rollout loop (both players act each round, not alternating)
- Per-actor reward functions (competitive scoring)
- Round-based game with history tracking

Game flow:
1. Both players see the round number and previous results
2. Both make their choice (simultaneously from game perspective)
3. Round is resolved, scores updated
4. Repeat for num_rounds
5. Split into per-actor states for scoring
"""

from datasets import Dataset

import verifiers as vf
from verifiers import Actor, MultiAgentEnv, MultiAgentRubric, Protocol
from verifiers.types import Messages, State


# =============================================================================
# Actors
# =============================================================================

PLAYER1 = Actor(
    id="player1",
    system_prompt="""You are Player 1 in Rock-Paper-Scissors.

Choose ONE of: rock, paper, or scissors

Output ONLY your choice (one word, lowercase). Nothing else.""",
    max_tokens=10,
    is_trainable=True,
)

PLAYER2 = Actor(
    id="player2",
    system_prompt="""You are Player 2 in Rock-Paper-Scissors.

Choose ONE of: rock, paper, or scissors

Output ONLY your choice (one word, lowercase). Nothing else.""",
    max_tokens=10,
    is_trainable=True,
)


# =============================================================================
# Environment
# =============================================================================

class RockPaperScissorsEnv(MultiAgentEnv):
    """Rock-Paper-Scissors with simultaneous moves."""

    name = "rock_paper_scissors"
    actors = ["player1", "player2"]

    def __init__(self, num_rounds: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds

    # -------------------------------------------------------------------------
    # Turn Management
    # Required by MultiAgentEnv but not really used here - we override rollout()
    # to use get_active_actors() for simultaneous play instead.
    # -------------------------------------------------------------------------

    def get_initial_actor(self, state: State) -> str:
        return "player1"

    def get_next_actor(self, state: State) -> str:
        return "player1"

    def get_active_actors(self, state: State) -> list[str]:
        """Both players act simultaneously each round."""
        return ["player1", "player2"]

    # -------------------------------------------------------------------------
    # Stop Condition
    # -------------------------------------------------------------------------

    @vf.stop
    async def game_over(self, state: State) -> bool:
        """Stop after all rounds played."""
        return state.get("extras", {}).get("round", 0) >= self.num_rounds

    # -------------------------------------------------------------------------
    # State Setup
    # -------------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        """Initialize RPS-specific game state."""
        state = await super().setup_state(state)
        state["extras"]["round"] = 0           # Current round (0-indexed during play)
        state["extras"]["p1_score"] = 0        # Player 1 win count
        state["extras"]["p2_score"] = 0        # Player 2 win count
        state["extras"]["history"] = []        # List of (p1_choice, p2_choice, result)
        state["extras"]["p1_choice"] = None    # Temp storage for current round
        state["extras"]["p2_choice"] = None    # Temp storage for current round
        return state

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    async def get_prompt_messages(self, state: State) -> Messages:
        """
        Build fresh prompt for current actor each round.

        Overrides base class because RPS needs clean, summarized prompts
        rather than accumulated raw conversation.
        """
        current_actor_id = state["extras"]["current_actor_id"]
        actor = self.get_actor(current_actor_id)
        round_num = state["extras"]["round"] + 1

        # Build history from this player's perspective ("You" vs "Opponent")
        history = state["extras"]["history"]
        history_str = ""
        if history:
            history_str = "\n\nPrevious rounds:\n"
            for i, (p1, p2, result) in enumerate(history, 1):
                you = p1 if current_actor_id == "player1" else p2
                opponent = p2 if current_actor_id == "player1" else p1
                history_str += f"  Round {i}: You={you}, Opponent={opponent} → {result}\n"

        return [
            {"role": "system", "content": actor.system_prompt},
            {"role": "user", "content": f"Round {round_num} of {self.num_rounds}. Make your choice!{history_str}"}
        ]

    # -------------------------------------------------------------------------
    # Environment Response (Game Logic)
    # -------------------------------------------------------------------------

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        """
        Process each actor's choice and resolve round when both have played.

        Called after each actor's turn:
        - First call: stores player1's choice
        - Second call: stores player2's choice, resolves round

        Returns [] because we build fresh prompts in get_prompt_messages().
        """
        if not state["trajectory"]:
            return []

        # Get the last completion
        last_step = state["trajectory"][-1]
        last_completion = last_step.get("completion", [])
        if not last_completion:
            return []

        # Parse the choice from model output
        content = last_completion[-1].get("content", "").lower().strip() if isinstance(last_completion[-1], dict) else str(last_completion[-1]).lower().strip()
        choice = self._parse_choice(content)

        # Store choice for the actor who just played
        actor_id = last_step.get("extras", {}).get("actor_id", state["extras"]["current_actor_id"])
        if actor_id == "player1":
            state["extras"]["p1_choice"] = choice
        else:
            state["extras"]["p2_choice"] = choice

        # If both have chosen, resolve the round
        p1_choice = state["extras"]["p1_choice"]
        p2_choice = state["extras"]["p2_choice"]

        if p1_choice and p2_choice:
            winner = self._determine_winner(p1_choice, p2_choice)

            if winner == "player1":
                state["extras"]["p1_score"] += 1
                result = "Player 1 wins"
            elif winner == "player2":
                state["extras"]["p2_score"] += 1
                result = "Player 2 wins"
            else:
                result = "Tie"

            # Record result and reset for next round
            state["extras"]["history"].append((p1_choice, p2_choice, result))
            state["extras"]["round"] += 1
            state["extras"]["p1_choice"] = None
            state["extras"]["p2_choice"] = None

            print(f"  [Round {state['extras']['round']}] {p1_choice} vs {p2_choice} → {result}")

        return []

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def _parse_choice(self, text: str) -> str:
        """Extract rock/paper/scissors from model output. Defaults to rock."""
        text = text.lower()
        if "rock" in text:
            return "rock"
        elif "paper" in text:
            return "paper"
        elif "scissors" in text:
            return "scissors"
        return "rock"

    def _determine_winner(self, p1: str, p2: str) -> str | None:
        """Determine winner using standard RPS rules. Returns None for tie."""
        if p1 == p2:
            return None
        wins = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
        if wins.get(p1) == p2:
            return "player1"
        return "player2"

    # -------------------------------------------------------------------------
    # Custom Rollout (Simultaneous Moves)
    # -------------------------------------------------------------------------

    async def rollout(
        self,
        input,
        client,
        model,
        sampling_args=None,
    ) -> State:
        """
        Custom rollout with simultaneous moves.

        Overrides base class because RPS has both players act each round,
        rather than strict alternation. Uses get_active_actors() to get
        both players, then loops through them each round.
        """
        state = await self.init_state(input, client, model, sampling_args)
        state = await self.setup_state(state)

        while not await self.is_completed(state):
            active_actors = self.get_active_actors(state)

            for actor_id in active_actors:
                # Set who's currently playing
                state["extras"]["current_actor_id"] = actor_id

                try:
                    # Build prompt for this actor
                    prompt_messages = await self.get_prompt_messages(state)

                    # Get actor's sampling settings
                    actor = self.get_actor(actor_id)
                    merged_args = actor.merge_sampling_args(sampling_args or {})

                    # Call the model
                    response = await self.get_model_response(state, prompt_messages, sampling_args=merged_args)

                    # Store in trajectory (tagged with actor_id)
                    await self.add_model_response(state, prompt_messages, response)

                    # Process choice and maybe resolve round
                    await self.env_response([], state)

                except vf.Error as e:
                    state["error"] = e
                    break

        await self.render_completion(state)

        # Split into per-actor states for scoring
        state["child_states"] = self.create_actor_states(state)

        return state


# =============================================================================
# Rubric (Scoring)
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    """
    Create competitive rubric with per-actor rewards.

    Each player's reward = win rate (wins / total_rounds).
    Creates competitive dynamic: one player's gain ≈ other's loss.
    """
    rubric = MultiAgentRubric()

    def player1_reward(state: State, **kwargs) -> float:
        """Player 1 reward = win rate."""
        extras = state.get("extras", {})
        p1_score = extras.get("p1_score", 0)
        total_rounds = extras.get("round", 1)
        return p1_score / total_rounds if total_rounds > 0 else 0.0

    def player2_reward(state: State, **kwargs) -> float:
        """Player 2 reward = win rate."""
        extras = state.get("extras", {})
        p2_score = extras.get("p2_score", 0)
        total_rounds = extras.get("round", 1)
        return p2_score / total_rounds if total_rounds > 0 else 0.0

    def rounds_played_metric(state: State, **kwargs) -> float:
        """Track rounds played (metric only, weight=0)."""
        return float(state.get("extras", {}).get("round", 0))

    rubric.add_actor_reward_func("player1", player1_reward, weight=1.0)
    rubric.add_actor_reward_func("player2", player2_reward, weight=1.0)
    rubric.add_reward_func(rounds_played_metric, weight=0.0)

    return rubric


# =============================================================================
# Dataset
# =============================================================================

def create_dataset() -> Dataset:
    """
    Create dataset for RPS games.

    The prompt is a placeholder - RPS builds its own prompts via
    get_prompt_messages(). Each row represents one game to play.
    """
    return Dataset.from_list([
        {
            "example_id": i,
            "prompt": [{"role": "user", "content": "play"}],
            "answer": "",
            "task": "rock_paper_scissors"
        }
        for i in range(10)
    ])


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    num_rounds: int = 3,
    num_examples: int = -1,
) -> RockPaperScissorsEnv:
    """
    Factory function to create a fully configured RPS environment.

    Args:
        num_rounds: Rounds per game (default 3)
        num_examples: Number of games to run (-1 = all 10)

    Returns:
        Ready-to-use RockPaperScissorsEnv
    """
    dataset = create_dataset()
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    env = RockPaperScissorsEnv(
        num_rounds=num_rounds,
        rubric=create_rubric(),
        max_turns=num_rounds * 2 + 2,  # 2 turns per round + buffer
        dataset=dataset,
    )

    # Wire actors to environment via Protocol
    Protocol(actors=[PLAYER1, PLAYER2], envs=[env])

    return env
