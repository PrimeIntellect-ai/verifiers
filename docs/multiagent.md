# Multi-Agent Environments

This guide covers building multi-agent environments where multiple actors interact. See [Environments](environments.md) for single-agent basics and [API Reference](reference.md) for type definitions.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
  - [Actor](#actor)
  - [Protocol](#protocol)
  - [MultiAgentEnv](#multiagentenv)
  - [MultiAgentRubric](#multiagentubric)
- [Building a Multi-Agent Environment](#building-a-multi-agent-environment)
  - [Turn Management](#turn-management)
  - [Building Prompts](#building-prompts)
  - [Game Logic](#game-logic)
  - [Ending the Game](#ending-the-game)
- [Per-Actor Rewards](#per-actor-rewards)
  - [State Splitting](#state-splitting)
  - [Per-Actor GRPO](#per-actor-grpo)
  - [Frozen Actors](#frozen-actors)
- [Hierarchical Spawning](#hierarchical-spawning)
- [Examples](#examples)
  - [Alternating Turns (Twenty Questions)](#alternating-turns-twenty-questions)
  - [Simultaneous Moves (Rock Paper Scissors)](#simultaneous-moves-rock-paper-scissors)
  - [Hierarchical (Proposer-Solver)](#hierarchical-proposer-solver)
  - [Complex Game (Multi-Player Poker)](#complex-game-multi-player-poker)

## Overview

Multi-agent environments enable training multiple actors that interact with each other. Key capabilities:

- **Multiple actors** with distinct system prompts and configurations
- **Turn management** for alternating or simultaneous moves
- **Per-actor rewards** for credit assignment
- **Per-actor GRPO** advantages computed within actor groups
- **Hierarchical spawning** for complex multi-level games

The architecture separates concerns:

| Component | Responsibility |
|-----------|----------------|
| `Actor` | Configuration (system prompt, model, trainability) |
| `Protocol` | Registry (wires actors to environments, enables spawning) |
| `MultiAgentEnv` | Game logic (turn order, prompts, win conditions) |
| `MultiAgentRubric` | Scoring (per-actor rewards, per-actor GRPO) |

## Core Components

### Actor

An actor is a trainable entity with a distinct identity:

```python
from verifiers.envs.actor import Actor

player1 = Actor(
    id="player1",
    system_prompt="You are Player 1 in a game...",
    max_tokens=512,
    is_trainable=True,
)

judge = Actor(
    id="judge",
    system_prompt="You are a fair judge...",
    is_trainable=False,  # Frozen - no gradient updates
    model="gpt-4o-mini",  # Can use different model
)
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier (e.g., `"player1"`, `"guesser"`) |
| `system_prompt` | `str` | The actor's persona/instructions |
| `max_tokens` | `int` | Max response length (default: 4096) |
| `is_trainable` | `bool` | Whether to compute GRPO advantages (default: True) |
| `sampling_args` | `dict` | Per-actor sampling settings |
| `model` | `str \| None` | Model override (None = use default) |
| `client` | `AsyncOpenAI \| None` | Client override (None = use default) |

### Protocol

Protocol wires actors to environments and enables cross-environment spawning:

```python
from verifiers.envs.protocol import Protocol
from verifiers.envs.actor import Actor

# Define actors
player1 = Actor("player1", system_prompt="...")
player2 = Actor("player2", system_prompt="...")

# Define environment
env = MyGameEnv(rubric=rubric, max_turns=10)

# Wire them together
protocol = Protocol(
    actors=[player1, player2],
    envs=[env],
)
# Now env.protocol is set, and env.get_actor("player1") works
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_actor(actor_id)` | `Actor` | Look up actor by ID |
| `get_env(name)` | `Environment` | Look up environment by name |
| `spawn(inputs, client, model, ...)` | `list[State]` | Run child rollouts in target environments |

### MultiAgentEnv

`MultiAgentEnv` extends `MultiTurnEnv` with multi-actor support:

```python
class MyGameEnv(vf.MultiAgentEnv):
    name = "MyGame"  # For protocol registration
    actors = ["player1", "player2"]  # Actor IDs this env uses

    # Required: implement these four methods
    def get_initial_actor(self, state) -> str: ...
    def get_next_actor(self, state) -> str: ...
    async def build_actor_prompt(self, actor_id, state) -> Messages: ...
    async def on_turn_complete(self, state) -> None: ...

    # Optional: override for simultaneous moves
    def get_active_actors(self, state) -> list[str]: ...

    # Optional: final metrics before scoring
    async def on_game_end(self, state) -> None: ...
```

**Key differences from MultiTurnEnv:**

| MultiTurnEnv | MultiAgentEnv |
|--------------|---------------|
| Single actor | Multiple actors |
| `env_response()` after each turn | `on_turn_complete()` for game logic |
| Single prompt throughout | `build_actor_prompt()` per actor |
| — | Turn order via `get_initial_actor()` / `get_next_actor()` |

### MultiAgentRubric

`MultiAgentRubric` extends `Rubric` with per-actor rewards:

```python
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric

rubric = MultiAgentRubric()

# Global reward (applies to all actors)
rubric.add_reward_func(game_completed)

# Per-actor rewards
rubric.add_actor_reward_func("player1", player1_win_bonus)
rubric.add_actor_reward_func("player2", player2_win_bonus)

# Per-actor metrics (weight=0)
rubric.add_actor_metric("guesser", questions_asked)
```

**Key differences from Rubric:**

| Rubric | MultiAgentRubric |
|--------|------------------|
| Single GRPO across all states | Per-actor GRPO (solver vs solver) |
| Global reward functions only | Per-actor reward functions |
| — | Children scored before parents |

## Building a Multi-Agent Environment

### Turn Management

Implement turn order with two methods:

```python
class TwentyQuestionsEnv(vf.MultiAgentEnv):
    actors = ["guesser", "thinker"]

    def get_initial_actor(self, state) -> str:
        """Who goes first."""
        return "guesser"

    def get_next_actor(self, state) -> str:
        """Who goes next (alternating)."""
        current = state["extras"]["current_actor_id"]
        return "thinker" if current == "guesser" else "guesser"
```

For **simultaneous moves**, override `get_active_actors`:

```python
class RPSEnv(vf.MultiAgentEnv):
    actors = ["player1", "player2"]

    def get_active_actors(self, state) -> list[str]:
        """Both players move each round."""
        return ["player1", "player2"]
```

### Building Prompts

`build_actor_prompt` is called before each model response:

```python
async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
    """Build the prompt for this actor's turn."""
    actor = self.get_actor(actor_id)  # Get Actor config from Protocol

    # Build context from game state
    history = self.format_game_history(state)

    return [
        {"role": "system", "content": actor.system_prompt},
        {"role": "user", "content": f"Game history:\n{history}\n\nYour turn:"},
    ]
```

The actor's `model`, `client`, and `sampling_args` are automatically used when generating.

### Game Logic

`on_turn_complete` is called after each model response:

```python
async def on_turn_complete(self, state: State) -> None:
    """Update game state after a turn."""
    # Get the response that was just added
    last_step = state["trajectory"][-1]
    response_text = last_step["completion"][-1]["content"]
    actor_id = last_step["extras"]["actor_id"]

    # Parse and process
    move = self.parse_move(response_text)
    state["extras"]["moves"].append((actor_id, move))

    # Check win condition
    if self.check_winner(state):
        state["extras"]["winner"] = actor_id
```

### Ending the Game

To end a game, set `state["final_env_response"]`:

```python
async def on_turn_complete(self, state: State) -> None:
    if self.check_game_over(state):
        winner = state["extras"]["winner"]
        state["final_env_response"] = [
            {"role": "assistant", "content": f"Game over! {winner} wins!"}
        ]
```

This triggers the `has_final_env_response` stop condition. You can also add custom stop conditions:

```python
@vf.stop
async def game_won(self, state: State) -> bool:
    return state["extras"].get("won", False)
```

## Per-Actor Rewards

### State Splitting

After a game completes, `MultiAgentEnv.run_group()` splits each game state into per-actor states:

```
Game State (6 turns):
trajectory: [p1, p2, p1, p2, p1, p2]
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Player1 State           Player2 State
  traj: [p1, p1, p1]      traj: [p2, p2, p2]
  prompt: "You are P1"    prompt: "You are P2"
```

This is handled internally by `create_actor_states()`. Each actor state contains:

- Only that actor's trajectory steps
- The actor's prompt (from their first turn)
- Shared references to `client`, `model`, `trajectory_id`
- Fresh `reward`, `advantage`, `metrics` fields for scoring

### Per-Actor GRPO

`MultiAgentRubric.score_group()` computes advantages within actor groups:

```
Without per-actor grouping (bad):
  Solver reward=0.8, Proposer reward=0.2
  Mean = 0.5
  Solver advantage = +0.3, Proposer advantage = -0.3
  Unfair comparison across different roles

With per-actor grouping (good):
  Solvers compared to other solvers (mean = 0.75)
  Proposers compared to other proposers (mean = 0.25)
  Fair comparison within same role
```

### Frozen Actors

Actors with `is_trainable=False` are scored but don't receive gradients:

```python
thinker = Actor(
    "thinker",
    system_prompt="Answer yes/no about the secret word.",
    is_trainable=False,  # Frozen - just answers questions
)

guesser = Actor(
    "guesser",
    system_prompt="Guess the word in 20 questions.",
    is_trainable=True,  # Learning to ask good questions
)
```

Frozen actors get `advantage=0` during GRPO computation.

## Hierarchical Spawning

Protocol enables spawning child rollouts in other environments:

```python
class ProposerEnv(vf.MultiAgentEnv):
    async def on_turn_complete(self, state: State) -> None:
        if self.proposer_submitted_problem(state):
            problem = self.extract_problem(state)

            # Spawn solver attempts in SolverEnv
            child_states = await self.protocol.spawn(
                inputs=[
                    {"task": "SolverEnv", "prompt": problem},
                    {"task": "SolverEnv", "prompt": problem},
                    {"task": "SolverEnv", "prompt": problem},
                ],
                client=state["client"],
                model=state["model"],
            )

            # Store for later access
            state["child_states"] = child_states

            # Score proposer based on solver success
            solver_rewards = [s["reward"] for s in child_states]
            state["extras"]["solver_success_rate"] = sum(solver_rewards) / len(solver_rewards)
```

Child states are automatically included in `run_group()` output for training.

## Examples

### Alternating Turns (Twenty Questions)

```python
class TwentyQuestionsEnv(vf.MultiAgentEnv):
    name = "TwentyQuestions"
    actors = ["guesser", "thinker"]

    def get_initial_actor(self, state) -> str:
        return "guesser"

    def get_next_actor(self, state) -> str:
        current = state["extras"]["current_actor_id"]
        return "thinker" if current == "guesser" else "guesser"

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        actor = self.get_actor(actor_id)
        secret = state["info"]["secret_word"]
        history = self.format_qa_history(state)

        if actor_id == "guesser":
            content = f"History:\n{history}\n\nAsk a yes/no question or make a final guess."
        else:
            content = f"The secret word is: {secret}\n\nHistory:\n{history}\n\nAnswer yes or no."

        return [
            {"role": "system", "content": actor.system_prompt},
            {"role": "user", "content": content},
        ]

    async def on_turn_complete(self, state: State) -> None:
        actor_id = state["extras"]["current_actor_id"]
        response = state["trajectory"][-1]["completion"][-1]["content"]

        if actor_id == "guesser" and "FINAL GUESS:" in response:
            guess = self.extract_guess(response)
            secret = state["info"]["secret_word"]
            state["extras"]["won"] = (guess.lower() == secret.lower())
            state["final_env_response"] = [
                {"role": "assistant", "content": f"{'Correct!' if state['extras']['won'] else 'Wrong!'}"}
            ]
```

### Simultaneous Moves (Rock Paper Scissors)

```python
class RPSEnv(vf.MultiAgentEnv):
    name = "RPS"
    actors = ["player1", "player2"]

    def get_active_actors(self, state) -> list[str]:
        return ["player1", "player2"]  # Both move each round

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        actor = self.get_actor(actor_id)
        history = self.format_history_for(state, actor_id)  # Hide opponent's pending move

        return [
            {"role": "system", "content": actor.system_prompt},
            {"role": "user", "content": f"Previous rounds:\n{history}\n\nChoose: rock, paper, or scissors"},
        ]

    async def on_turn_complete(self, state: State) -> None:
        # Check if both players have moved this round
        recent = state["trajectory"][-2:]
        actors_moved = {step["extras"]["actor_id"] for step in recent}

        if actors_moved == {"player1", "player2"}:
            # Resolve the round
            p1_move = self.parse_move(recent[0]["completion"][-1]["content"])
            p2_move = self.parse_move(recent[1]["completion"][-1]["content"])
            winner = self.determine_winner(p1_move, p2_move)

            state["extras"]["rounds"].append({
                "p1": p1_move, "p2": p2_move, "winner": winner
            })

            if len(state["extras"]["rounds"]) >= 3:
                state["final_env_response"] = [
                    {"role": "assistant", "content": "Best of 3 complete!"}
                ]
```

### Hierarchical (Proposer-Solver)

```python
class ProposerSolverEnv(vf.MultiAgentEnv):
    name = "ProposerSolver"
    actors = ["proposer"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SolverEnv must also be registered with Protocol

    async def on_turn_complete(self, state: State) -> None:
        response = state["trajectory"][-1]["completion"][-1]["content"]
        problem = self.extract_problem(response)

        # Spawn solvers
        child_states = await self.protocol.spawn(
            inputs=[
                {"task": "SolverEnv", "prompt": self.format_problem(problem)}
                for _ in range(4)  # 4 solver attempts
            ],
            client=state["client"],
            model=state["model"],
        )

        state["child_states"] = child_states
        state["extras"]["num_solved"] = sum(1 for s in child_states if s["reward"] > 0)
        state["final_env_response"] = [{"role": "assistant", "content": "Solvers finished."}]


# Reward proposer based on solver success
async def proposer_reward(state: State) -> float:
    num_solved = state["extras"].get("num_solved", 0)
    return num_solved / 4.0  # Fraction of solvers that succeeded
```

Wire with Protocol:

```python
proposer = Actor("proposer", system_prompt="Generate a challenging math problem.")
solver = Actor("solver", system_prompt="Solve the given problem.")

proposer_env = ProposerSolverEnv(rubric=proposer_rubric)
solver_env = SolverEnv(rubric=solver_rubric)

protocol = Protocol(
    actors=[proposer, solver],
    envs=[proposer_env, solver_env],
)
```

### Complex Game (Multi-Player Poker)

See [`environments/poker_multi/poker_multi.py`](../environments/poker_multi/poker_multi.py) for a full example demonstrating:

- **Dynamic player count** - 2-9 players via dynamic `self.actors` list
- **Position-based turns** - UTG acts first preflop, dealer button rotates
- **Per-player model configs** - Different models/strategies per player
- **Multiple stop conditions** - `one_player_left`, `hand_complete`, `max_actions_hit`
- **Game phases** - Preflop → flop → turn → river → showdown with betting rounds
- **Per-actor rewards** - Chip profit/loss as fraction of starting stack

Key patterns from the implementation:

```python
class PokerMultiEnv(vf.MultiAgentEnv):
    name = "poker_multi"

    def __init__(self, num_players: int = 6, **kwargs):
        super().__init__(**kwargs)
        # Dynamic actor list based on player count
        self.actors = [f"player{i}" for i in range(1, num_players + 1)]

    def get_initial_actor(self, state) -> str:
        """Position-based: UTG (dealer+3) acts first preflop."""
        dealer_idx = state["extras"]["dealer_idx"]
        if self.num_players == 2:
            return self.actors[dealer_idx]  # Heads-up: dealer acts first
        return self.actors[(dealer_idx + 3) % self.num_players]  # UTG

    def get_next_actor(self, state) -> str:
        """Next player clockwise who hasn't folded and isn't all-in."""
        current_idx = self.actors.index(state["extras"]["current_actor_id"])
        for i in range(1, self.num_players + 1):
            candidate = self.actors[(current_idx + i) % self.num_players]
            if candidate not in state["extras"]["folded"]:
                return candidate
        return state["extras"]["current_actor_id"]

    @vf.stop
    async def one_player_left(self, state) -> bool:
        """End when all others fold."""
        active = [p for p in self.actors if p not in state["extras"]["folded"]]
        return len(active) == 1
```

```python
# Per-player model/strategy configuration
PLAYER_CONFIGS = [
    {"endpoint": "model-a", "strategy": "aggressive", "is_trainable": True},
    {"endpoint": "model-b", "strategy": "conservative", "is_trainable": False},
]

# Per-actor reward based on chip profit
def player_reward(actor_id: str):
    def reward_func(state: State) -> float:
        starting = state["extras"]["starting_chips"]
        final = state["extras"]["chips"].get(actor_id, starting)
        return (final - starting) / starting
    return reward_func
```
