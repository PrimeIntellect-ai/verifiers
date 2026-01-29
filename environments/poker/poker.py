"""
Poker: Heads-Up No-Limit Texas Hold'em.

This environment demonstrates:
- Hidden information (each player sees only their own hole cards)
- Complex state management (chips, pot, betting rounds)
- Multi-phase gameplay (preflop, flop, turn, river, showdown)
- JSON-structured action parsing
- Different models per player (small vs large)

Game flow:
1. Post blinds (dealer=small blind, other=big blind)
2. Deal 2 hole cards each (hidden)
3. Preflop betting (dealer acts first in heads-up)
4. Deal flop (3 community cards)
5. Postflop betting (non-dealer acts first)
6. Deal turn (1 card)
7. Turn betting
8. Deal river (1 card)
9. River betting
10. Showdown - compare hands, award pot
11. Repeat for num_hands (dealer rotates)
"""

import json
import random
import re
from collections import Counter
from itertools import combinations
from datasets import Dataset

from verifiers import Actor, MultiAgentEnv, MultiAgentRubric, Protocol
from verifiers.types import Messages, State
from verifiers.utils.client_utils import get_actor_client
import verifiers as vf

# =============================================================================
# Model Configuration
# =============================================================================
# Change these to use different models for each player.
# Set to None to use the default model from the eval command.
#
# Small models: "olmo3-7b-i", "trinity-mini", "haiku", "gemini-3-flash"
# Large models: "sonnet", "opus", "qwen3-235b-i", "gemini-3-pro"
# =============================================================================

PLAYER1_ENDPOINT = "olmo3-7b-i"   # Small model
PLAYER2_ENDPOINT = "qwen3-235b-i" # Large model

player1_client, player1_model = get_actor_client(PLAYER1_ENDPOINT)
player2_client, player2_model = get_actor_client(PLAYER2_ENDPOINT)


# =============================================================================
# Card Utilities
# =============================================================================

RANKS = "23456789TJQKA"
SUITS = "hdcs"  # hearts, diamonds, clubs, spades
RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}  # 2=2, 3=3, ..., A=14


def create_deck() -> list[str]:
    """Create a shuffled 52-card deck."""
    deck = [f"{r}{s}" for r in RANKS for s in SUITS]
    random.shuffle(deck)
    return deck


def card_rank(card: str) -> int:
    """Get numeric rank value (2-14)."""
    return RANK_VALUES[card[0]]


def card_suit(card: str) -> str:
    """Get suit character."""
    return card[1]


def format_cards(cards: list[str]) -> str:
    """Format cards for display."""
    return ", ".join(cards) if cards else "None"


# =============================================================================
# Hand Evaluation (Simple built-in evaluator)
# =============================================================================

# Hand rankings (higher = better)
HAND_RANKS = {
    "high_card": 1,
    "pair": 2,
    "two_pair": 3,
    "three_of_a_kind": 4,
    "straight": 5,
    "flush": 6,
    "full_house": 7,
    "four_of_a_kind": 8,
    "straight_flush": 9,
    "royal_flush": 10,
}


def evaluate_five_cards(cards: list[str]) -> tuple[int, list[int]]:
    """
    Evaluate a 5-card hand.
    Returns (hand_rank, tiebreakers) where higher is better.
    """
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]
    rank_counts = Counter(ranks)

    is_flush = len(set(suits)) == 1

    # Check for straight (including A-2-3-4-5 wheel)
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0

    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        # Wheel: A-2-3-4-5
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5  # 5-high straight

    # Get counts for pair/trips/etc detection
    counts = sorted(rank_counts.values(), reverse=True)

    # Determine hand type
    if is_straight and is_flush:
        if straight_high == 14 and 13 in ranks:  # Royal flush
            return (HAND_RANKS["royal_flush"], [14])
        return (HAND_RANKS["straight_flush"], [straight_high])

    if counts == [4, 1]:
        quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
        kicker = [r for r, c in rank_counts.items() if c == 1][0]
        return (HAND_RANKS["four_of_a_kind"], [quad_rank, kicker])

    if counts == [3, 2]:
        trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        return (HAND_RANKS["full_house"], [trip_rank, pair_rank])

    if is_flush:
        return (HAND_RANKS["flush"], ranks)

    if is_straight:
        return (HAND_RANKS["straight"], [straight_high])

    if counts == [3, 1, 1]:
        trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HAND_RANKS["three_of_a_kind"], [trip_rank] + kickers)

    if counts == [2, 2, 1]:
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = [r for r, c in rank_counts.items() if c == 1][0]
        return (HAND_RANKS["two_pair"], pairs + [kicker])

    if counts == [2, 1, 1, 1]:
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HAND_RANKS["pair"], [pair_rank] + kickers)

    return (HAND_RANKS["high_card"], ranks)


def evaluate_hand(hole_cards: list[str], community: list[str]) -> tuple[int, list[int], str]:
    """
    Find best 5-card hand from 7 cards.
    Returns (hand_rank, tiebreakers, hand_name).
    """
    all_cards = hole_cards + community
    best_score = (0, [])
    best_name = "high_card"

    # Try all 21 combinations of 5 cards from 7
    for five_cards in combinations(all_cards, 5):
        score = evaluate_five_cards(list(five_cards))
        if score > best_score:
            best_score = score
            # Find hand name
            for name, rank in HAND_RANKS.items():
                if rank == score[0]:
                    best_name = name
                    break

    return (best_score[0], best_score[1], best_name)


def compare_hands(
    hole1: list[str], hole2: list[str], community: list[str]
) -> tuple[str, str, str]:
    """
    Compare two hands.
    Returns (winner, hand1_name, hand2_name) where winner is "player1", "player2", or "tie".
    """
    eval1 = evaluate_hand(hole1, community)
    eval2 = evaluate_hand(hole2, community)

    score1 = (eval1[0], eval1[1])
    score2 = (eval2[0], eval2[1])

    if score1 > score2:
        return ("player1", eval1[2], eval2[2])
    elif score2 > score1:
        return ("player2", eval1[2], eval2[2])
    else:
        return ("tie", eval1[2], eval2[2])


# =============================================================================
# Actors
# =============================================================================

PLAYER_SYSTEM_PROMPT = """You are playing Heads-Up No-Limit Texas Hold'em Poker.

Rules:
- You and your opponent each have 2 hole cards (hidden from each other)
- 5 community cards are dealt face-up over multiple rounds
- Best 5-card hand from your 7 cards wins

On your turn, output ONLY a JSON object with your action:
- Fold: {"action": "fold"}
- Check (if no bet to call): {"action": "check"}
- Call (match current bet): {"action": "call"}
- Raise to amount: {"action": "raise", "amount": 100}
- All-in: {"action": "allin"}

Output ONLY the JSON, nothing else."""

PLAYER1 = Actor(
    id="player1",
    system_prompt=PLAYER_SYSTEM_PROMPT,
    max_tokens=50,
    is_trainable=True,
    model=player1_model,
    client=player1_client,
)

PLAYER2 = Actor(
    id="player2",
    system_prompt=PLAYER_SYSTEM_PROMPT,
    max_tokens=50,
    is_trainable=True,
    model=player2_model,
    client=player2_client,
)


# =============================================================================
# Environment
# =============================================================================

class PokerEnv(MultiAgentEnv):
    """Heads-Up No-Limit Texas Hold'em."""

    name = "poker"
    actors = ["player1", "player2"]

    def __init__(
        self,
        num_hands: int = 1,
        max_actions_per_hand: int = 20,
        starting_chips: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hands = num_hands
        self.max_actions_per_hand = max_actions_per_hand
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = big_blind

    # -------------------------------------------------------------------------
    # Turn Management
    # -------------------------------------------------------------------------

    def get_initial_actor(self, state: State) -> str:
        """Dealer (small blind) acts first preflop in heads-up."""
        return state["extras"]["dealer"]

    def get_next_actor(self, state: State) -> str:
        """Alternate between players, skip folded."""
        current = state["extras"]["current_actor_id"]
        next_actor = "player2" if current == "player1" else "player1"

        # Skip if folded
        if next_actor in state["extras"]["folded"]:
            return current
        return next_actor

    # -------------------------------------------------------------------------
    # Stop Conditions
    # -------------------------------------------------------------------------

    @vf.stop
    async def player_folded(self, state: State) -> bool:
        """One player folded - hand over."""
        return len(state["extras"]["folded"]) > 0

    @vf.stop
    async def hand_complete(self, state: State) -> bool:
        """Showdown complete or all hands played."""
        return state["extras"]["phase"] == "complete"

    @vf.stop
    async def max_actions_hit(self, state: State) -> bool:
        """Safety limit - force showdown."""
        if state["extras"]["actions_this_hand"] >= self.max_actions_per_hand:
            await self._force_showdown(state)
            return True
        return False

    # -------------------------------------------------------------------------
    # State Setup
    # -------------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        """Initialize poker game state."""
        state = await super().setup_state(state)

        # Session tracking
        state["extras"]["hands_played"] = 0
        state["extras"]["starting_chips"] = self.starting_chips
        state["extras"]["chips"] = {
            "player1": self.starting_chips,
            "player2": self.starting_chips,
        }

        # Start first hand
        state["extras"]["dealer"] = "player1"  # Will rotate each hand
        await self._start_new_hand(state)

        return state

    async def _start_new_hand(self, state: State) -> None:
        """Initialize state for a new hand."""
        extras = state["extras"]

        # Create and shuffle deck
        extras["deck"] = create_deck()

        # Deal hole cards
        extras["hole_cards"] = {
            "player1": [extras["deck"].pop(), extras["deck"].pop()],
            "player2": [extras["deck"].pop(), extras["deck"].pop()],
        }

        # Reset hand state
        extras["community_cards"] = []
        extras["pot"] = 0
        extras["current_bet"] = 0
        extras["bets_this_round"] = {"player1": 0, "player2": 0}
        extras["phase"] = "preflop"
        extras["folded"] = []
        extras["actions_this_hand"] = 0
        extras["actions_this_round"] = {"player1": 0, "player2": 0}
        extras["last_aggressor"] = None

        # Post blinds
        dealer = extras["dealer"]
        non_dealer = "player2" if dealer == "player1" else "player1"

        # Dealer posts small blind
        sb_amount = min(self.small_blind, extras["chips"][dealer])
        extras["chips"][dealer] -= sb_amount
        extras["bets_this_round"][dealer] = sb_amount
        extras["pot"] += sb_amount

        # Non-dealer posts big blind
        bb_amount = min(self.big_blind, extras["chips"][non_dealer])
        extras["chips"][non_dealer] -= bb_amount
        extras["bets_this_round"][non_dealer] = bb_amount
        extras["pot"] += bb_amount
        extras["current_bet"] = bb_amount

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        """Build prompt showing only this player's hole cards."""
        extras = state["extras"]

        my_cards = extras["hole_cards"][actor_id]
        community = extras["community_cards"]
        pot = extras["pot"]
        my_chips = extras["chips"][actor_id]
        opponent_id = "player2" if actor_id == "player1" else "player1"
        opponent_chips = extras["chips"][opponent_id]

        to_call = extras["current_bet"] - extras["bets_this_round"][actor_id]

        # Build situation description
        phase = extras["phase"].upper()
        hand_num = extras["hands_played"] + 1

        position = "Dealer (Small Blind)" if actor_id == extras["dealer"] else "Big Blind"

        situation = f"""=== HAND {hand_num} - {phase} ===
Position: {position}

Your hole cards: {format_cards(my_cards)}
Community cards: {format_cards(community)}

Pot: {pot}
Your chips: {my_chips}
Opponent chips: {opponent_chips}

Current bet: {extras['current_bet']}
Your bet this round: {extras['bets_this_round'][actor_id]}
To call: {to_call}

Your action?"""

        return [
            {"role": "system", "content": PLAYER_SYSTEM_PROMPT},
            {"role": "user", "content": situation},
        ]

    # -------------------------------------------------------------------------
    # Action Parsing
    # -------------------------------------------------------------------------

    def _parse_action(self, text: str, state: State, actor_id: str) -> dict:
        """Parse JSON action from model output."""
        try:
            # Strip markdown code blocks if present
            clean = text.strip()
            if "```" in clean:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
                if match:
                    clean = match.group(1)

            # Try to find JSON object in text
            json_match = re.search(r"\{[^}]+\}", clean)
            if json_match:
                clean = json_match.group(0)

            action = json.loads(clean)

            # Validate required field
            if "action" not in action:
                raise ValueError("Missing 'action' key")

            action_type = action["action"].lower()

            # Validate action type
            valid_actions = ["fold", "check", "call", "raise", "allin"]
            if action_type not in valid_actions:
                raise ValueError(f"Invalid action: {action_type}")

            # Raise requires amount
            if action_type == "raise":
                if "amount" not in action:
                    raise ValueError("Raise requires 'amount'")
                action["amount"] = int(action["amount"])

            action["action"] = action_type
            return action

        except (json.JSONDecodeError, ValueError, KeyError):
            # FALLBACK: Safe default action
            return self._get_fallback_action(state, actor_id)

    def _get_fallback_action(self, state: State, actor_id: str) -> dict:
        """When parsing fails, do the safest legal action."""
        to_call = state["extras"]["current_bet"] - state["extras"]["bets_this_round"][actor_id]
        if to_call == 0:
            return {"action": "check"}
        else:
            return {"action": "fold"}

    def _validate_and_adjust_action(self, action: dict, state: State, actor_id: str) -> dict:
        """Validate action and adjust if needed (clamp raises, etc.)."""
        extras = state["extras"]
        my_chips = extras["chips"][actor_id]
        to_call = extras["current_bet"] - extras["bets_this_round"][actor_id]

        action_type = action["action"]

        # Can't check if there's a bet to call
        if action_type == "check" and to_call > 0:
            # Convert to call or fold based on having chips
            if my_chips >= to_call:
                return {"action": "call"}
            else:
                return {"action": "allin"}

        # Can't call more than we have
        if action_type == "call":
            if my_chips <= to_call:
                return {"action": "allin"}
            return action

        # Handle raise
        if action_type == "raise":
            amount = action.get("amount", 0)
            min_raise = extras["current_bet"] + self.big_blind  # Minimum raise
            max_raise = my_chips + extras["bets_this_round"][actor_id]  # All-in

            # Clamp to valid range
            if amount >= max_raise:
                return {"action": "allin"}
            if amount < min_raise:
                amount = min_raise
            if amount > max_raise:
                amount = max_raise

            return {"action": "raise", "amount": amount}

        # All-in is always valid
        if action_type == "allin":
            return action

        # Fold is always valid
        return action

    # -------------------------------------------------------------------------
    # Game Logic
    # -------------------------------------------------------------------------

    async def on_turn_complete(self, state: State) -> None:
        """Process action after each turn."""
        if not state["trajectory"]:
            return

        last_step = state["trajectory"][-1]
        last_completion = last_step.get("completion", [])
        if not last_completion:
            return

        # Get actor who just played
        actor_id = last_step.get("extras", {}).get("actor_id")
        if not actor_id:
            return

        # Parse action
        content = last_completion[-1].get("content", "") if isinstance(last_completion[-1], dict) else str(last_completion[-1])
        action = self._parse_action(content, state, actor_id)
        action = self._validate_and_adjust_action(action, state, actor_id)

        extras = state["extras"]
        extras["actions_this_hand"] += 1
        extras["actions_this_round"][actor_id] += 1

        # Process action
        await self._process_action(action, state, actor_id)

        # Check if betting round is complete
        if self._is_betting_round_complete(state):
            await self._advance_phase(state)

    async def _process_action(self, action: dict, state: State, actor_id: str) -> None:
        """Process a validated action."""
        extras = state["extras"]
        action_type = action["action"]

        if action_type == "fold":
            extras["folded"].append(actor_id)
            # Award pot to opponent
            opponent = "player2" if actor_id == "player1" else "player1"
            extras["chips"][opponent] += extras["pot"]
            extras["pot"] = 0
            extras["winner"] = opponent
            extras["win_reason"] = "fold"
            extras["phase"] = "complete"

        elif action_type == "check":
            pass  # No chip movement

        elif action_type == "call":
            to_call = extras["current_bet"] - extras["bets_this_round"][actor_id]
            call_amount = min(to_call, extras["chips"][actor_id])
            extras["chips"][actor_id] -= call_amount
            extras["bets_this_round"][actor_id] += call_amount
            extras["pot"] += call_amount

        elif action_type == "raise":
            amount = action["amount"]
            # Amount is total bet, not additional
            current_bet = extras["bets_this_round"][actor_id]
            additional = amount - current_bet
            additional = min(additional, extras["chips"][actor_id])

            extras["chips"][actor_id] -= additional
            extras["bets_this_round"][actor_id] += additional
            extras["pot"] += additional
            extras["current_bet"] = extras["bets_this_round"][actor_id]
            extras["last_aggressor"] = actor_id

            # Reset opponent's action count (they need to respond to raise)
            opponent = "player2" if actor_id == "player1" else "player1"
            extras["actions_this_round"][opponent] = 0

        elif action_type == "allin":
            all_in_amount = extras["chips"][actor_id]
            extras["bets_this_round"][actor_id] += all_in_amount
            extras["pot"] += all_in_amount
            extras["chips"][actor_id] = 0

            if extras["bets_this_round"][actor_id] > extras["current_bet"]:
                extras["current_bet"] = extras["bets_this_round"][actor_id]
                extras["last_aggressor"] = actor_id
                # Reset opponent's action count
                opponent = "player2" if actor_id == "player1" else "player1"
                extras["actions_this_round"][opponent] = 0

    def _is_betting_round_complete(self, state: State) -> bool:
        """Check if current betting round is complete."""
        extras = state["extras"]

        # Hand already over
        if extras["phase"] == "complete" or extras["folded"]:
            return False

        # Both players must have acted at least once this round
        p1_actions = extras["actions_this_round"]["player1"]
        p2_actions = extras["actions_this_round"]["player2"]

        if p1_actions == 0 or p2_actions == 0:
            return False

        # Bets must be equal (or someone is all-in)
        p1_bet = extras["bets_this_round"]["player1"]
        p2_bet = extras["bets_this_round"]["player2"]
        p1_chips = extras["chips"]["player1"]
        p2_chips = extras["chips"]["player2"]

        bets_equal = p1_bet == p2_bet
        p1_allin = p1_chips == 0
        p2_allin = p2_chips == 0

        return bets_equal or p1_allin or p2_allin

    async def _advance_phase(self, state: State) -> None:
        """Move to next phase of the hand."""
        extras = state["extras"]
        current_phase = extras["phase"]

        # Reset for new betting round
        extras["bets_this_round"] = {"player1": 0, "player2": 0}
        extras["current_bet"] = 0
        extras["actions_this_round"] = {"player1": 0, "player2": 0}
        extras["last_aggressor"] = None

        # Determine next phase
        phase_order = ["preflop", "flop", "turn", "river", "showdown"]
        current_idx = phase_order.index(current_phase)
        next_phase = phase_order[current_idx + 1]
        extras["phase"] = next_phase

        # Deal community cards
        if next_phase == "flop":
            # Burn and deal 3
            extras["deck"].pop()  # Burn
            extras["community_cards"].extend([
                extras["deck"].pop(),
                extras["deck"].pop(),
                extras["deck"].pop(),
            ])
        elif next_phase == "turn":
            # Burn and deal 1
            extras["deck"].pop()  # Burn
            extras["community_cards"].append(extras["deck"].pop())
        elif next_phase == "river":
            # Burn and deal 1
            extras["deck"].pop()  # Burn
            extras["community_cards"].append(extras["deck"].pop())
        elif next_phase == "showdown":
            await self._resolve_showdown(state)

        # Update current actor for post-flop (non-dealer acts first)
        if next_phase in ["flop", "turn", "river"]:
            dealer = extras["dealer"]
            non_dealer = "player2" if dealer == "player1" else "player1"
            # Only change if non-dealer hasn't folded
            if non_dealer not in extras["folded"]:
                extras["current_actor_id"] = non_dealer

    async def _force_showdown(self, state: State) -> None:
        """Force showdown when max actions reached."""
        extras = state["extras"]

        # Deal remaining community cards
        while len(extras["community_cards"]) < 5:
            if extras["deck"]:
                extras["deck"].pop()  # Burn
            if extras["deck"]:
                extras["community_cards"].append(extras["deck"].pop())

        extras["phase"] = "showdown"
        await self._resolve_showdown(state)

    async def _resolve_showdown(self, state: State) -> None:
        """Compare hands and award pot."""
        extras = state["extras"]

        p1_hole = extras["hole_cards"]["player1"]
        p2_hole = extras["hole_cards"]["player2"]
        community = extras["community_cards"]

        winner, p1_hand, p2_hand = compare_hands(p1_hole, p2_hole, community)

        extras["player1_hand"] = {
            "hole_cards": p1_hole,
            "hand_name": p1_hand.replace("_", " ").title(),
        }
        extras["player2_hand"] = {
            "hole_cards": p2_hole,
            "hand_name": p2_hand.replace("_", " ").title(),
        }

        if winner == "tie":
            # Split pot
            half = extras["pot"] // 2
            extras["chips"]["player1"] += half
            extras["chips"]["player2"] += extras["pot"] - half
            extras["winner"] = "tie"
            extras["win_reason"] = "split pot"
        else:
            extras["chips"][winner] += extras["pot"]
            extras["winner"] = winner
            loser = "player2" if winner == "player1" else "player1"
            winner_hand = extras[f"{winner}_hand"]["hand_name"]
            loser_hand = extras[f"{loser}_hand"]["hand_name"]
            extras["win_reason"] = f"{winner_hand} beats {loser_hand}"

        extras["pot"] = 0
        extras["hands_played"] += 1

        # Check if session continues
        if extras["hands_played"] < self.num_hands:
            # Check if both players have chips
            if extras["chips"]["player1"] > 0 and extras["chips"]["player2"] > 0:
                # Rotate dealer and start new hand
                extras["dealer"] = "player2" if extras["dealer"] == "player1" else "player1"
                await self._start_new_hand(state)
                return

        # Session complete
        extras["phase"] = "complete"

    async def on_game_end(self, state: State) -> None:
        """Compute final session results."""
        extras = state["extras"]

        # Determine overall winner
        p1_chips = extras["chips"]["player1"]
        p2_chips = extras["chips"]["player2"]

        if p1_chips > p2_chips:
            extras["session_winner"] = "player1"
        elif p2_chips > p1_chips:
            extras["session_winner"] = "player2"
        else:
            extras["session_winner"] = "tie"

        # Calculate profit/loss
        starting = extras["starting_chips"]
        extras["player1_profit"] = p1_chips - starting
        extras["player2_profit"] = p2_chips - starting


# =============================================================================
# Rubric (Scoring)
# =============================================================================

def create_rubric() -> MultiAgentRubric:
    """Create rubric based on chip profit/loss."""
    rubric = MultiAgentRubric()

    def player_reward(actor_id: str):
        def reward_func(state: State, **kwargs) -> float:
            extras = state.get("extras", {})
            starting = extras.get("starting_chips", 1000)
            final = extras.get("chips", {}).get(actor_id, starting)
            # Normalize to [-1, 1] range (losing all = -1, doubling up = +1)
            return (final - starting) / starting
        return reward_func

    def hands_played_metric(state: State, **kwargs) -> float:
        return float(state.get("extras", {}).get("hands_played", 0))

    def showdowns_metric(state: State, **kwargs) -> float:
        # Count non-fold endings
        winner = state.get("extras", {}).get("winner")
        reason = state.get("extras", {}).get("win_reason", "")
        return 0.0 if "fold" in reason else 1.0

    rubric.add_actor_reward_func("player1", player_reward("player1"), weight=1.0)
    rubric.add_actor_reward_func("player2", player_reward("player2"), weight=1.0)
    rubric.add_reward_func(hands_played_metric, weight=0.0)
    rubric.add_reward_func(showdowns_metric, weight=0.0)

    return rubric


# =============================================================================
# Dataset
# =============================================================================

def create_dataset(num_games: int = 10) -> Dataset:
    """Create dataset for poker games."""
    return Dataset.from_list([
        {
            "example_id": i,
            "prompt": [{"role": "user", "content": "play"}],
            "answer": "",
            "task": "poker",
        }
        for i in range(num_games)
    ])


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    num_hands: int = 5,
    max_actions_per_hand: int = 20,
    starting_chips: int = 1000,
    small_blind: int = 5,
    big_blind: int = 10,
    num_examples: int = -1,
) -> PokerEnv:
    """Factory function to create a fully configured Poker environment."""
    dataset = create_dataset()
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    env = PokerEnv(
        num_hands=num_hands,
        max_actions_per_hand=max_actions_per_hand,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        rubric=create_rubric(),
        max_turns=num_hands * max_actions_per_hand + 10,
        dataset=dataset,
    )

    Protocol(actors=[PLAYER1, PLAYER2], envs=[env])

    return env
