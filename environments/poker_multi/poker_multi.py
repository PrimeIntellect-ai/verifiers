"""
Poker Multi: Multi-player No-Limit Texas Hold'em (2-9 players).

This environment extends the heads-up poker to support:
- Configurable number of players (2-9)
- Full side pot tracking for all-in situations
- Proper dealer button rotation with SB/BB posting
- Turn rotation through active (non-folded) players

Game flow:
1. Post blinds (SB = dealer+1, BB = dealer+2)
2. Deal 2 hole cards each (hidden)
3. Preflop betting (UTG = dealer+3 acts first, or SB in heads-up)
4. Deal flop (3 community cards)
5. Postflop betting (first active player left of dealer)
6. Deal turn (1 card)
7. Turn betting
8. Deal river (1 card)
9. River betting
10. Showdown - compare hands, award pots (including side pots)
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
# Player Configuration
# =============================================================================
# Configure each player's model and strategy.
# Players without a config use the default model from the -m flag.
#
# Available endpoints: See configs/endpoints.py (e.g., "olmo3-7b-i", "sonnet")
# Available strategies: "aggressive", "conservative", "balanced"
# =============================================================================

PLAYER_CONFIGS = [
    {"endpoint": "olmo3-7b-i", "strategy": "aggressive", "is_trainable": True},
    {"endpoint": "qwen3-235b-i", "strategy": "conservative", "is_trainable": False},
    {"endpoint": "qwen3-30b-i", "strategy": "balanced", "is_trainable": False},
    {"endpoint": "qwen3-235b-i", "strategy": "aggressive", "is_trainable": False},
    {"endpoint": "olmo3-7b-i", "strategy": "conservative", "is_trainable": False},
    # Player 6+ use default model from -m flag with "balanced" strategy
]

STRATEGY_PROMPTS = {
    "aggressive": """
YOUR STRATEGY - Play aggressively:
- RAISE frequently, especially preflop with any decent hand
- BLUFF often - bet and raise even with mediocre hands to pressure opponents
- NEVER fold preflop unless you have absolute garbage (like 2-7 offsuit)
- When in doubt, RAISE rather than call or check
- Put maximum pressure on your opponents""",

    "conservative": """
YOUR STRATEGY - Play solid poker:
- CALL or RAISE with good hands (pairs, high cards like A/K/Q, suited connectors)
- CHECK when you can to see free cards
- Only FOLD when facing a big bet with a truly weak hand
- If you have a strong hand (pair or better), RAISE to build the pot
- Go to SHOWDOWN when possible to see who wins""",

    "balanced": """
YOUR STRATEGY - Play balanced poker:
- Mix up your play - sometimes raise, sometimes call, occasionally bluff
- RAISE with strong hands (pairs, AK, AQ) to build pots
- CALL with speculative hands (suited connectors, small pairs) to see flops
- FOLD weak hands when facing significant bets
- Pay attention to pot odds and position""",
}


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
# Hand Evaluation
# =============================================================================

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

    counts = sorted(rank_counts.values(), reverse=True)

    # Determine hand type
    if is_straight and is_flush:
        if straight_high == 14 and 13 in ranks:
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

    for five_cards in combinations(all_cards, 5):
        score = evaluate_five_cards(list(five_cards))
        if score > best_score:
            best_score = score
            for name, rank in HAND_RANKS.items():
                if rank == score[0]:
                    best_name = name
                    break

    return (best_score[0], best_score[1], best_name)


def find_best_hand(
    players: list[str],
    hole_cards: dict[str, list[str]],
    community: list[str],
) -> tuple[list[str], dict[str, tuple]]:
    """
    Find the best hand(s) among a list of players.
    Returns (winners, evaluations) where winners may be multiple (split pot).
    """
    evaluations = {}
    for player in players:
        eval_result = evaluate_hand(hole_cards[player], community)
        evaluations[player] = eval_result

    # Find best score
    best_score = max((evaluations[p][0], evaluations[p][1]) for p in players)

    # Find all players with that score (for split pots)
    winners = [
        p for p in players
        if (evaluations[p][0], evaluations[p][1]) == best_score
    ]

    return winners, evaluations


# =============================================================================
# Actors
# =============================================================================

def create_player_system_prompt(num_players: int, strategy: str = "balanced") -> str:
    """Create system prompt with strategy instructions."""
    strategy_text = STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["balanced"])

    return f"""You are playing {num_players}-player No-Limit Texas Hold'em Poker.

Rules:
- Each player has 2 hole cards (hidden from others)
- 5 community cards are dealt face-up over multiple rounds
- Best 5-card hand from your 7 cards wins
{strategy_text}

On your turn, output ONLY a JSON object with your action:
- Fold: {{"action": "fold"}}
- Check (if no bet to call): {{"action": "check"}}
- Call (match current bet): {{"action": "call"}}
- Raise to amount: {{"action": "raise", "amount": 100}}
- All-in: {{"action": "allin"}}

Output ONLY the JSON, nothing else."""


def create_actors(num_players: int) -> list[Actor]:
    """
    Create player actors with per-player model and strategy configs.

    Players with a config in PLAYER_CONFIGS get that model/strategy.
    Players without a config use the default model from -m flag with "balanced" strategy.
    """
    actors = []
    for i in range(num_players):
        # Get config if exists, otherwise use defaults
        config = PLAYER_CONFIGS[i] if i < len(PLAYER_CONFIGS) else None

        if config:
            client, model = get_actor_client(config.get("endpoint"))
            strategy = config.get("strategy", "balanced")
            is_trainable = config.get("is_trainable", False)
        else:
            # No config = use default model from -m flag
            client, model = None, None
            strategy = "balanced"
            is_trainable = False

        system_prompt = create_player_system_prompt(num_players, strategy)

        actors.append(Actor(
            id=f"player{i+1}",
            system_prompt=system_prompt,
            max_tokens=50,
            is_trainable=is_trainable,
            model=model,
            client=client,
        ))
    return actors


# =============================================================================
# Side Pot Management
# =============================================================================

class SidePotManager:
    """
    Manages side pots for all-in situations.

    When a player goes all-in for less than the current bet,
    we split the pot so they can only win from those who matched their bet.
    """

    def __init__(self, players: list[str]):
        self.players = players
        # Track total amount each player has put in across all rounds
        self.contributions: dict[str, int] = {p: 0 for p in players}
        # Players still in the hand (not folded)
        self.active: set[str] = set(players)

    def add_contribution(self, player: str, amount: int) -> None:
        """Record chips put into pot by player."""
        self.contributions[player] += amount

    def fold(self, player: str) -> None:
        """Mark player as folded (can't win, contributions stay)."""
        self.active.discard(player)

    def calculate_pots(self) -> list[dict]:
        """
        Calculate main pot and side pots based on contributions.

        Returns list of pots, each with:
        - amount: chips in this pot
        - eligible: players who can win this pot
        """
        if not self.active:
            return []

        # Get unique contribution levels from active players
        active_contributions = {
            p: self.contributions[p] for p in self.active
        }

        # All contribution levels (including folded players who added chips)
        all_contributions = list(self.contributions.values())

        # Unique levels sorted ascending
        levels = sorted(set(c for c in all_contributions if c > 0))

        if not levels:
            return []

        pots = []
        prev_level = 0

        for level in levels:
            # How much each player contributes to this tier
            tier_amount = level - prev_level

            # Players who contributed at least this much
            contributors = [
                p for p in self.players
                if self.contributions[p] >= level
            ]

            # Eligible winners = active players who contributed
            eligible = [p for p in contributors if p in self.active]

            if eligible and tier_amount > 0:
                pot_amount = tier_amount * len(contributors)
                pots.append({
                    "amount": pot_amount,
                    "eligible": eligible,
                    "level": level,
                })

            prev_level = level

        return pots

    def total_pot(self) -> int:
        """Get total chips in all pots."""
        return sum(self.contributions.values())


# =============================================================================
# Environment
# =============================================================================

class PokerMultiEnv(MultiAgentEnv):
    """Multi-player No-Limit Texas Hold'em (2-9 players)."""

    name = "poker_multi"

    def __init__(
        self,
        num_players: int = 6,
        num_hands: int = 1,
        max_actions_per_hand: int = 50,
        starting_chips: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        **kwargs,
    ):
        # Validate player count
        if num_players < 2 or num_players > 9:
            raise ValueError("num_players must be between 2 and 9")

        super().__init__(**kwargs)
        self.num_players = num_players
        self.num_hands = num_hands
        self.max_actions_per_hand = max_actions_per_hand
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = big_blind

        # Dynamic actor list
        self.actors = [f"player{i}" for i in range(1, num_players + 1)]

    # -------------------------------------------------------------------------
    # Turn Management
    # -------------------------------------------------------------------------

    def _get_seat_order(self, state: State) -> list[str]:
        """Get players in seat order starting from dealer."""
        extras = state["extras"]
        dealer_idx = extras["dealer_idx"]
        n = self.num_players
        return [self.actors[(dealer_idx + i) % n] for i in range(n)]

    def _get_active_players(self, state: State) -> list[str]:
        """Get non-folded players in seat order."""
        extras = state["extras"]
        return [p for p in self._get_seat_order(state) if p not in extras["folded"]]

    def _get_players_who_can_act(self, state: State) -> list[str]:
        """Get players who can still act (not folded, not all-in)."""
        extras = state["extras"]
        return [
            p for p in self._get_seat_order(state)
            if p not in extras["folded"] and extras["chips"][p] > 0
        ]

    def get_initial_actor(self, state: State) -> str:
        """UTG acts first preflop (dealer+3), or dealer+1 for heads-up."""
        extras = state["extras"]
        dealer_idx = extras["dealer_idx"]
        n = self.num_players

        if n == 2:
            # Heads-up: dealer (SB) acts first preflop
            start_idx = dealer_idx
        else:
            # UTG = dealer + 3 (after SB and BB)
            start_idx = (dealer_idx + 3) % n

        # Find first player who can act starting from this position
        for i in range(n):
            candidate_idx = (start_idx + i) % n
            candidate = self.actors[candidate_idx]
            if candidate not in extras["folded"] and extras["chips"][candidate] > 0:
                return candidate

        # Fallback - shouldn't happen if game state is valid
        return self.actors[start_idx]

    def get_next_actor(self, state: State) -> str:
        """Get next player to act in clockwise order."""
        extras = state["extras"]
        current = extras["current_actor_id"]
        current_idx = self.actors.index(current)
        n = self.num_players

        # Find next player who can act
        for i in range(1, n + 1):
            next_idx = (current_idx + i) % n
            next_player = self.actors[next_idx]

            # Skip folded players
            if next_player in extras["folded"]:
                continue
            # Skip all-in players (can't act but still in hand)
            if extras["chips"][next_player] == 0:
                continue

            return next_player

        # No one can act (everyone folded or all-in)
        return current

    # -------------------------------------------------------------------------
    # Stop Conditions
    # -------------------------------------------------------------------------

    @vf.stop
    async def one_player_left(self, state: State) -> bool:
        """Only one player remains (all others folded)."""
        active = self._get_active_players(state)
        if len(active) == 1:
            # Award pot to remaining player
            winner = active[0]
            extras = state["extras"]
            extras["chips"][winner] += extras["pot_manager"].total_pot()
            extras["winner"] = winner
            extras["win_reason"] = "all others folded"
            extras["hands_played"] += 1

            # Check for next hand
            if extras["hands_played"] < self.num_hands:
                players_with_chips = [p for p in self.actors if extras["chips"][p] > 0]
                if len(players_with_chips) >= 2:
                    # Rotate dealer to next active player
                    extras["dealer_idx"] = (extras["dealer_idx"] + 1) % self.num_players
                    extras["dealer_idx"] = self._next_active_seat(state, extras["dealer_idx"])
                    await self._start_new_hand(state)
                    return False

            extras["phase"] = "complete"
            return True
        return False

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

        extras = state["extras"]
        extras["hands_played"] = 0
        extras["starting_chips"] = self.starting_chips
        extras["chips"] = {p: self.starting_chips for p in self.actors}

        # Start first hand with player1 as dealer
        extras["dealer_idx"] = 0
        await self._start_new_hand(state)

        return state

    def _next_active_seat(self, state: State, start_idx: int) -> int:
        """Find next seat index with an active (non-busted) player."""
        extras = state["extras"]
        n = self.num_players
        for i in range(n):
            idx = (start_idx + i) % n
            if extras["chips"][self.actors[idx]] > 0:
                return idx
        return start_idx  # Fallback

    async def _start_new_hand(self, state: State) -> None:
        """Initialize state for a new hand."""
        extras = state["extras"]
        n = self.num_players

        # Ensure dealer is on an active player
        extras["dealer_idx"] = self._next_active_seat(state, extras["dealer_idx"])
        dealer_idx = extras["dealer_idx"]

        # Create and shuffle deck
        extras["deck"] = create_deck()

        # Deal hole cards to all players
        extras["hole_cards"] = {}
        for player in self.actors:
            if extras["chips"][player] > 0:
                extras["hole_cards"][player] = [
                    extras["deck"].pop(),
                    extras["deck"].pop(),
                ]
            else:
                extras["hole_cards"][player] = []  # Busted player

        # Reset hand state
        extras["community_cards"] = []
        extras["current_bet"] = 0
        extras["bets_this_round"] = {p: 0 for p in self.actors}
        extras["phase"] = "preflop"
        extras["folded"] = [p for p in self.actors if extras["chips"][p] == 0]
        extras["actions_this_hand"] = 0
        extras["actions_this_round"] = {p: 0 for p in self.actors}
        extras["last_aggressor"] = None

        # Initialize pot manager
        active_players = [p for p in self.actors if extras["chips"][p] > 0]
        extras["pot_manager"] = SidePotManager(active_players)

        # Find SB and BB seats (skip busted players)
        num_active = len(active_players)
        if num_active == 2:
            # Heads-up: dealer is SB
            sb_idx = dealer_idx
            bb_idx = self._next_active_seat(state, (dealer_idx + 1) % n)
        else:
            # Regular: SB is left of dealer, BB is left of SB
            sb_idx = self._next_active_seat(state, (dealer_idx + 1) % n)
            bb_idx = self._next_active_seat(state, (sb_idx + 1) % n)

        sb_player = self.actors[sb_idx]
        bb_player = self.actors[bb_idx]

        # Post small blind
        sb_amount = min(self.small_blind, extras["chips"][sb_player])
        extras["chips"][sb_player] -= sb_amount
        extras["bets_this_round"][sb_player] = sb_amount
        extras["pot_manager"].add_contribution(sb_player, sb_amount)

        # Post big blind
        bb_amount = min(self.big_blind, extras["chips"][bb_player])
        extras["chips"][bb_player] -= bb_amount
        extras["bets_this_round"][bb_player] = bb_amount
        extras["current_bet"] = bb_amount
        extras["pot_manager"].add_contribution(bb_player, bb_amount)

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    async def build_actor_prompt(self, actor_id: str, state: State) -> Messages:
        """Build prompt showing only this player's hole cards."""
        extras = state["extras"]

        my_cards = extras["hole_cards"].get(actor_id, [])
        community = extras["community_cards"]
        my_chips = extras["chips"][actor_id]
        to_call = extras["current_bet"] - extras["bets_this_round"][actor_id]

        phase = extras["phase"].upper()
        hand_num = extras["hands_played"] + 1
        total_pot = extras["pot_manager"].total_pot()

        # Build opponent info
        opponent_info = []
        for p in self.actors:
            if p != actor_id:
                status = "folded" if p in extras["folded"] else (
                    "all-in" if extras["chips"][p] == 0 else "active"
                )
                opponent_info.append(
                    f"  {p}: {extras['chips'][p]} chips ({status})"
                )

        # Determine position
        dealer_idx = extras["dealer_idx"]
        my_idx = self.actors.index(actor_id)
        relative_pos = (my_idx - dealer_idx) % self.num_players

        if self.num_players == 2:
            position = "Dealer/SB" if relative_pos == 0 else "BB"
        else:
            positions = ["Dealer", "SB", "BB"] + [f"UTG+{i}" for i in range(6)]
            position = positions[relative_pos] if relative_pos < len(positions) else f"Seat {relative_pos}"

        situation = f"""=== HAND {hand_num} - {phase} ===
Players: {self.num_players}
Position: {position}

Your hole cards: {format_cards(my_cards)}
Community cards: {format_cards(community)}

Pot: {total_pot}
Your chips: {my_chips}
Opponents:
{chr(10).join(opponent_info)}

Current bet: {extras['current_bet']}
Your bet this round: {extras['bets_this_round'][actor_id]}
To call: {to_call}

Your action?"""

        actor = self.get_actor(actor_id)
        return [
            {"role": "system", "content": actor.system_prompt},
            {"role": "user", "content": situation},
        ]

    # -------------------------------------------------------------------------
    # Action Parsing
    # -------------------------------------------------------------------------

    def _parse_action(self, text: str, state: State, actor_id: str) -> dict:
        """Parse JSON action from model output."""
        try:
            clean = text.strip()
            if "```" in clean:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
                if match:
                    clean = match.group(1)

            json_match = re.search(r"\{[^}]+\}", clean)
            if json_match:
                clean = json_match.group(0)

            action = json.loads(clean)

            if "action" not in action:
                raise ValueError("Missing 'action' key")

            action_type = action["action"].lower()
            valid_actions = ["fold", "check", "call", "raise", "allin"]
            if action_type not in valid_actions:
                raise ValueError(f"Invalid action: {action_type}")

            if action_type == "raise":
                if "amount" not in action:
                    raise ValueError("Raise requires 'amount'")
                action["amount"] = int(action["amount"])

            action["action"] = action_type
            return action

        except (json.JSONDecodeError, ValueError, KeyError):
            return self._get_fallback_action(state, actor_id)

    def _get_fallback_action(self, state: State, actor_id: str) -> dict:
        """When parsing fails, do the safest legal action."""
        to_call = state["extras"]["current_bet"] - state["extras"]["bets_this_round"][actor_id]
        if to_call == 0:
            return {"action": "check"}
        else:
            return {"action": "fold"}

    def _validate_and_adjust_action(self, action: dict, state: State, actor_id: str) -> dict:
        """Validate action and adjust if needed."""
        extras = state["extras"]
        my_chips = extras["chips"][actor_id]
        to_call = extras["current_bet"] - extras["bets_this_round"][actor_id]

        action_type = action["action"]

        if action_type == "check" and to_call > 0:
            if my_chips >= to_call:
                return {"action": "call"}
            else:
                return {"action": "allin"}

        if action_type == "call":
            if my_chips <= to_call:
                return {"action": "allin"}
            return action

        if action_type == "raise":
            amount = action.get("amount", 0)
            min_raise = extras["current_bet"] + self.big_blind
            max_raise = my_chips + extras["bets_this_round"][actor_id]

            if amount >= max_raise:
                return {"action": "allin"}
            if amount < min_raise:
                amount = min_raise
            if amount > max_raise:
                amount = max_raise

            return {"action": "raise", "amount": amount}

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

        actor_id = last_step.get("extras", {}).get("actor_id")
        if not actor_id:
            return

        content = last_completion[-1].get("content", "") if isinstance(last_completion[-1], dict) else str(last_completion[-1])
        action = self._parse_action(content, state, actor_id)
        action = self._validate_and_adjust_action(action, state, actor_id)

        extras = state["extras"]
        extras["actions_this_hand"] += 1
        extras["actions_this_round"][actor_id] += 1

        await self._process_action(action, state, actor_id)

        if self._is_betting_round_complete(state):
            await self._advance_phase(state)

    async def _process_action(self, action: dict, state: State, actor_id: str) -> None:
        """Process a validated action."""
        extras = state["extras"]
        action_type = action["action"]
        pot_mgr = extras["pot_manager"]

        if action_type == "fold":
            extras["folded"].append(actor_id)
            pot_mgr.fold(actor_id)

        elif action_type == "check":
            pass

        elif action_type == "call":
            to_call = extras["current_bet"] - extras["bets_this_round"][actor_id]
            call_amount = min(to_call, extras["chips"][actor_id])
            extras["chips"][actor_id] -= call_amount
            extras["bets_this_round"][actor_id] += call_amount
            pot_mgr.add_contribution(actor_id, call_amount)

        elif action_type == "raise":
            amount = action["amount"]
            current_bet = extras["bets_this_round"][actor_id]
            additional = amount - current_bet
            additional = min(additional, extras["chips"][actor_id])

            extras["chips"][actor_id] -= additional
            extras["bets_this_round"][actor_id] += additional
            extras["current_bet"] = extras["bets_this_round"][actor_id]
            extras["last_aggressor"] = actor_id
            pot_mgr.add_contribution(actor_id, additional)

            # Reset action counts for players who need to respond
            for p in self.actors:
                if p != actor_id and p not in extras["folded"] and extras["chips"][p] > 0:
                    extras["actions_this_round"][p] = 0

        elif action_type == "allin":
            all_in_amount = extras["chips"][actor_id]
            extras["bets_this_round"][actor_id] += all_in_amount
            extras["chips"][actor_id] = 0
            pot_mgr.add_contribution(actor_id, all_in_amount)

            if extras["bets_this_round"][actor_id] > extras["current_bet"]:
                extras["current_bet"] = extras["bets_this_round"][actor_id]
                extras["last_aggressor"] = actor_id
                # Reset action counts for players who need to respond
                for p in self.actors:
                    if p != actor_id and p not in extras["folded"] and extras["chips"][p] > 0:
                        extras["actions_this_round"][p] = 0

    def _is_betting_round_complete(self, state: State) -> bool:
        """Check if current betting round is complete."""
        extras = state["extras"]

        if extras["phase"] == "complete":
            return False

        active = self._get_active_players(state)
        if len(active) <= 1:
            return False  # Will be handled by one_player_left stop

        # Players who can still act
        can_act = self._get_players_who_can_act(state)

        # If no one can act, round is complete
        if not can_act:
            return True

        # Check if all can-act players have acted and bets are equal
        for player in can_act:
            # Player hasn't acted
            if extras["actions_this_round"][player] == 0:
                return False
            # Player's bet doesn't match (and they're not all-in)
            if extras["bets_this_round"][player] < extras["current_bet"]:
                return False

        return True

    async def _advance_phase(self, state: State) -> None:
        """Move to next phase of the hand."""
        extras = state["extras"]
        current_phase = extras["phase"]

        # Reset for new betting round
        extras["bets_this_round"] = {p: 0 for p in self.actors}
        extras["current_bet"] = 0
        extras["actions_this_round"] = {p: 0 for p in self.actors}
        extras["last_aggressor"] = None

        phase_order = ["preflop", "flop", "turn", "river", "showdown"]
        current_idx = phase_order.index(current_phase)
        next_phase = phase_order[current_idx + 1]
        extras["phase"] = next_phase

        if next_phase == "flop":
            extras["deck"].pop()  # Burn
            extras["community_cards"].extend([
                extras["deck"].pop(),
                extras["deck"].pop(),
                extras["deck"].pop(),
            ])
        elif next_phase == "turn":
            extras["deck"].pop()
            extras["community_cards"].append(extras["deck"].pop())
        elif next_phase == "river":
            extras["deck"].pop()
            extras["community_cards"].append(extras["deck"].pop())
        elif next_phase == "showdown":
            await self._resolve_showdown(state)
            return

        # Set first actor for post-flop (first active player left of dealer)
        if next_phase in ["flop", "turn", "river"]:
            dealer_idx = extras["dealer_idx"]
            for i in range(1, self.num_players + 1):
                candidate_idx = (dealer_idx + i) % self.num_players
                candidate = self.actors[candidate_idx]
                if candidate not in extras["folded"] and extras["chips"][candidate] > 0:
                    extras["current_actor_id"] = candidate
                    break

    async def _force_showdown(self, state: State) -> None:
        """Force showdown when max actions reached."""
        extras = state["extras"]

        while len(extras["community_cards"]) < 5:
            if extras["deck"]:
                extras["deck"].pop()
            if extras["deck"]:
                extras["community_cards"].append(extras["deck"].pop())

        extras["phase"] = "showdown"
        await self._resolve_showdown(state)

    async def _resolve_showdown(self, state: State) -> None:
        """Compare hands and award pots (including side pots)."""
        extras = state["extras"]
        pot_mgr = extras["pot_manager"]
        community = extras["community_cards"]

        # Calculate pots
        pots = pot_mgr.calculate_pots()

        # Track winnings
        winnings = {p: 0 for p in self.actors}
        pot_results = []

        for pot in pots:
            eligible = pot["eligible"]
            amount = pot["amount"]

            if len(eligible) == 1:
                # Only one player eligible
                winner = eligible[0]
                winnings[winner] += amount
                pot_results.append({
                    "amount": amount,
                    "winners": [winner],
                    "reason": "sole eligible"
                })
            else:
                # Compare hands
                winners, evaluations = find_best_hand(
                    eligible,
                    extras["hole_cards"],
                    community,
                )

                # Split pot among winners
                share = amount // len(winners)
                remainder = amount % len(winners)

                for i, w in enumerate(winners):
                    winnings[w] += share
                    if i < remainder:
                        winnings[w] += 1  # Odd chips go to first winners

                hand_name = evaluations[winners[0]][2].replace("_", " ").title()
                pot_results.append({
                    "amount": amount,
                    "winners": winners,
                    "hand": hand_name,
                    "reason": "best hand" if len(winners) == 1 else "split pot"
                })

        # Award chips
        for player, amount in winnings.items():
            extras["chips"][player] += amount

        # Store results
        extras["pot_results"] = pot_results

        # Determine overall winner (most chips won this hand)
        if winnings:
            max_won = max(winnings.values())
            winners = [p for p, w in winnings.items() if w == max_won and w > 0]
            if len(winners) == 1:
                extras["winner"] = winners[0]
            else:
                extras["winner"] = "split"
        else:
            extras["winner"] = None

        extras["win_reason"] = "showdown"

        # Store hand evaluations for all active players
        active = self._get_active_players(state)
        extras["final_hands"] = {}
        for p in active:
            ev = evaluate_hand(extras["hole_cards"][p], community)
            extras["final_hands"][p] = {
                "hole_cards": extras["hole_cards"][p],
                "hand_name": ev[2].replace("_", " ").title(),
            }

        extras["hands_played"] += 1

        # Check if session continues
        if extras["hands_played"] < self.num_hands:
            players_with_chips = [p for p in self.actors if extras["chips"][p] > 0]
            if len(players_with_chips) >= 2:
                # Rotate dealer to next active player
                next_dealer = (extras["dealer_idx"] + 1) % self.num_players
                extras["dealer_idx"] = self._next_active_seat(state, next_dealer)
                await self._start_new_hand(state)
                return

        extras["phase"] = "complete"

    async def on_game_end(self, state: State) -> None:
        """Compute final session results."""
        extras = state["extras"]

        # Rank players by final chips
        chip_ranking = sorted(
            self.actors,
            key=lambda p: extras["chips"][p],
            reverse=True,
        )

        extras["final_ranking"] = chip_ranking
        extras["session_winner"] = chip_ranking[0]

        # Calculate profits
        starting = extras["starting_chips"]
        extras["profits"] = {
            p: extras["chips"][p] - starting
            for p in self.actors
        }


# =============================================================================
# Rubric (Scoring)
# =============================================================================

def create_rubric(num_players: int) -> MultiAgentRubric:
    """Create rubric based on chip profit/loss."""
    rubric = MultiAgentRubric()

    def player_reward(actor_id: str):
        def reward_func(state: State, **kwargs) -> float:
            extras = state.get("extras", {})
            starting = extras.get("starting_chips", 1000)
            final = extras.get("chips", {}).get(actor_id, starting)
            return (final - starting) / starting
        return reward_func

    def hands_played_metric(state: State, **kwargs) -> float:
        return float(state.get("extras", {}).get("hands_played", 0))

    def showdowns_metric(state: State, **kwargs) -> float:
        reason = state.get("extras", {}).get("win_reason", "")
        return 0.0 if "fold" in reason else 1.0

    for i in range(1, num_players + 1):
        player_id = f"player{i}"
        rubric.add_actor_reward_func(player_id, player_reward(player_id), weight=1.0)

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
            "task": "poker_multi",
        }
        for i in range(num_games)
    ])


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    num_players: int = 6,
    num_hands: int = 1,
    max_actions_per_hand: int = 50,
    starting_chips: int = 1000,
    small_blind: int = 5,
    big_blind: int = 10,
    num_examples: int = -1,
) -> PokerMultiEnv:
    """Factory function to create a fully configured multi-player Poker environment."""
    dataset = create_dataset()
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    env = PokerMultiEnv(
        num_players=num_players,
        num_hands=num_hands,
        max_actions_per_hand=max_actions_per_hand,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        rubric=create_rubric(num_players),
        max_turns=num_hands * max_actions_per_hand * num_players + 10,
        dataset=dataset,
    )

    actors = create_actors(num_players)
    Protocol(actors=actors, envs=[env])

    return env
