"""
Poker Multi v2: New decomposition.

Old style: PokerMultiEnv(MultiAgentEnv) — 1100+ lines, game logic, hand evaluation,
           side pots, prompts, turn management, rubric, actors, dataset all in one class.

New style: Game logic in PokerTask(TaskSet). Agents are separate.
           Card utils and hand evaluation are module-level helpers.

What this shows about the abstractions:
    - TaskSet handles complex turn management (get_next_role depends on game state:
      who's folded, who's all-in, what betting phase we're in)
    - TaskSet handles hidden information (each player only sees their own cards)
    - Variable number of agents (2-9 players configured at load time)
    - Per-player Agents with different models and strategies
    - All players output a JSON action
    - Simple Agents — one model call per turn, no tools needed
"""

import json
import random
import re
from collections import Counter
from itertools import combinations

from datasets import Dataset

from verifiers.agent import Agent
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.taskset import TaskSet
from verifiers.types import Messages, State
from verifiers.utils.client_utils import get_actor_client


# =============================================================================
# Player Configuration
# =============================================================================

PLAYER_CONFIGS = [
    {"endpoint": "olmo3-7b-i", "strategy": "aggressive", "is_trainable": True},
    {"endpoint": "qwen3-235b-i", "strategy": "conservative", "is_trainable": False},
    {"endpoint": "qwen3-30b-i", "strategy": "balanced", "is_trainable": False},
    {"endpoint": "qwen3-235b-i", "strategy": "aggressive", "is_trainable": False},
    {"endpoint": "olmo3-7b-i", "strategy": "conservative", "is_trainable": False},
]

STRATEGY_PROMPTS = {
    "aggressive": (
        "YOUR STRATEGY - Play aggressively:\n"
        "- RAISE frequently, especially preflop with any decent hand\n"
        "- BLUFF often - bet and raise even with mediocre hands\n"
        "- NEVER fold preflop unless absolute garbage\n"
        "- When in doubt, RAISE"
    ),
    "conservative": (
        "YOUR STRATEGY - Play solid poker:\n"
        "- CALL or RAISE with good hands (pairs, high cards)\n"
        "- CHECK when you can to see free cards\n"
        "- Only FOLD when facing a big bet with a weak hand\n"
        "- Go to SHOWDOWN when possible"
    ),
    "balanced": (
        "YOUR STRATEGY - Play balanced poker:\n"
        "- Mix up your play - sometimes raise, sometimes call\n"
        "- RAISE with strong hands to build pots\n"
        "- CALL with speculative hands to see flops\n"
        "- FOLD weak hands when facing big bets"
    ),
}


# =============================================================================
# Card Utilities (module-level, not in TaskSet)
# =============================================================================

RANKS = "23456789TJQKA"
SUITS = "hdcs"
RANK_VALUES = {r: i for i, r in enumerate(RANKS, 2)}


def create_deck() -> list[str]:
    deck = [f"{r}{s}" for r in RANKS for s in SUITS]
    random.shuffle(deck)
    return deck


def format_cards(cards: list[str]) -> str:
    return ", ".join(cards) if cards else "None"


def evaluate_five_cards(cards: list[str]) -> tuple[int, list[int]]:
    ranks = sorted([RANK_VALUES[c[0]] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    rank_counts = Counter(ranks)
    is_flush = len(set(suits)) == 1
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0

    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5

    counts = sorted(rank_counts.values(), reverse=True)
    HR = {"high_card": 1, "pair": 2, "two_pair": 3, "three_kind": 4, "straight": 5,
          "flush": 6, "full_house": 7, "four_kind": 8, "straight_flush": 9, "royal_flush": 10}

    if is_straight and is_flush:
        if straight_high == 14 and 13 in ranks:
            return (HR["royal_flush"], [14])
        return (HR["straight_flush"], [straight_high])
    if counts == [4, 1]:
        q = [r for r, c in rank_counts.items() if c == 4][0]
        k = [r for r, c in rank_counts.items() if c == 1][0]
        return (HR["four_kind"], [q, k])
    if counts == [3, 2]:
        t = [r for r, c in rank_counts.items() if c == 3][0]
        p = [r for r, c in rank_counts.items() if c == 2][0]
        return (HR["full_house"], [t, p])
    if is_flush:
        return (HR["flush"], ranks)
    if is_straight:
        return (HR["straight"], [straight_high])
    if counts == [3, 1, 1]:
        t = [r for r, c in rank_counts.items() if c == 3][0]
        ks = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HR["three_kind"], [t] + ks)
    if counts == [2, 2, 1]:
        ps = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        k = [r for r, c in rank_counts.items() if c == 1][0]
        return (HR["two_pair"], ps + [k])
    if counts == [2, 1, 1, 1]:
        p = [r for r, c in rank_counts.items() if c == 2][0]
        ks = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HR["pair"], [p] + ks)
    return (HR["high_card"], ranks)


def best_hand(hole_cards: list[str], community: list[str]) -> tuple[int, list[int], str]:
    all_cards = hole_cards + community
    best = (0, [])
    best_name = "high_card"
    names = {1: "High Card", 2: "Pair", 3: "Two Pair", 4: "Three of a Kind",
             5: "Straight", 6: "Flush", 7: "Full House", 8: "Four of a Kind",
             9: "Straight Flush", 10: "Royal Flush"}
    for five in combinations(all_cards, 5):
        score = evaluate_five_cards(list(five))
        if score > best:
            best = score
            best_name = names.get(score[0], "High Card")
    return (best[0], best[1], best_name)


def find_winners(players, hole_cards, community):
    evals = {p: best_hand(hole_cards[p], community) for p in players}
    best_score = max((evals[p][0], evals[p][1]) for p in players)
    winners = [p for p in players if (evals[p][0], evals[p][1]) == best_score]
    return winners, evals


# =============================================================================
# Rubric
# =============================================================================

def create_rubric(num_players: int) -> MultiAgentRubric:
    rubric = MultiAgentRubric()
    for i in range(1, num_players + 1):
        pid = f"player{i}"
        def make_reward(actor_id):
            def reward(state, **kwargs) -> float:
                extras = state.get("extras", {})
                starting = extras.get("starting_chips", 1000)
                final = extras.get("chips", {}).get(actor_id, starting)
                return (final - starting) / starting
            return reward
        rubric.add_actor_reward_func(pid, make_reward(pid), weight=1.0)
    return rubric


# =============================================================================
# TaskSet: ALL game logic lives here
# =============================================================================

class PokerTask(TaskSet):
    """
    Multi-player No-Limit Texas Hold'em.

    Compare with old PokerMultiEnv (1100+ lines in one class):
    - BEFORE: Turn management, betting phases, side pots, prompts,
              hand evaluation, rubric — all in one MultiAgentEnv subclass
    - AFTER:  All game logic in this TaskSet. Card utils are module-level.
              MultiAgentEnv just calls build_prompt / on_turn_complete / should_stop.

    Complex turn management:
    - get_next_role() checks who's folded, who's all-in, betting phase
    - Not simple alternation — depends entirely on game state
    """

    def __init__(
        self,
        num_players: int = 6,
        num_hands: int = 1,
        max_actions: int = 50,
        starting_chips: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        num_examples: int = -1,
    ):
        if num_players < 2 or num_players > 9:
            raise ValueError("num_players must be between 2 and 9")

        self.num_players = num_players
        self.num_hands = num_hands
        self.max_actions = max_actions
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.player_ids = [f"player{i}" for i in range(1, num_players + 1)]

        dataset = self._create_dataset()
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))

        super().__init__(
            name="poker_multi",
            dataset=dataset,
            rubric=create_rubric(num_players),
            roles=self.player_ids,
        )

    def _create_dataset(self) -> Dataset:
        return Dataset.from_list([
            {"prompt": [{"role": "user", "content": "play"}], "answer": "",
             "info": {}, "example_id": i, "task": "poker_multi"}
            for i in range(10)
        ])

    # ---- State ----

    async def setup_state(self, state: State) -> State:
        e = state["extras"]
        e["hands_played"] = 0
        e["starting_chips"] = self.starting_chips
        e["chips"] = {p: self.starting_chips for p in self.player_ids}
        e["dealer_idx"] = 0
        self._start_hand(state)
        return state

    def _start_hand(self, state: State) -> None:
        e = state["extras"]
        n = self.num_players
        e["deck"] = create_deck()
        e["hole_cards"] = {}
        for p in self.player_ids:
            e["hole_cards"][p] = [e["deck"].pop(), e["deck"].pop()] if e["chips"][p] > 0 else []

        e["community"] = []
        e["current_bet"] = 0
        e["bets_round"] = {p: 0 for p in self.player_ids}
        e["phase"] = "preflop"
        e["folded"] = [p for p in self.player_ids if e["chips"][p] == 0]
        e["actions_hand"] = 0
        e["actions_round"] = {p: 0 for p in self.player_ids}
        e["contributions"] = {p: 0 for p in self.player_ids}

        # Post blinds
        di = e["dealer_idx"]
        active = [p for p in self.player_ids if e["chips"][p] > 0]
        if len(active) < 2:
            e["phase"] = "complete"
            return

        if len(active) == 2:
            sb_p = self.player_ids[di % n]
            while sb_p not in active:
                di += 1
                sb_p = self.player_ids[di % n]
            bb_p = next(p for p in active if p != sb_p)
        else:
            sb_idx = self._next_active_seat(e, (di + 1) % n)
            bb_idx = self._next_active_seat(e, (sb_idx + 1) % n)
            sb_p = self.player_ids[sb_idx]
            bb_p = self.player_ids[bb_idx]

        sb_amt = min(self.small_blind, e["chips"][sb_p])
        e["chips"][sb_p] -= sb_amt
        e["bets_round"][sb_p] = sb_amt
        e["contributions"][sb_p] += sb_amt

        bb_amt = min(self.big_blind, e["chips"][bb_p])
        e["chips"][bb_p] -= bb_amt
        e["bets_round"][bb_p] = bb_amt
        e["current_bet"] = bb_amt
        e["contributions"][bb_p] += bb_amt

    def _next_active_seat(self, extras, start_idx):
        n = self.num_players
        for i in range(n):
            idx = (start_idx + i) % n
            if extras["chips"][self.player_ids[idx]] > 0:
                return idx
        return start_idx

    # ---- Turn Management (the complex part) ----

    def get_initial_role(self, state: State) -> str:
        e = state["extras"]
        di = e["dealer_idx"]
        n = self.num_players
        active = [p for p in self.player_ids if p not in e["folded"] and e["chips"][p] > 0]

        if n == 2:
            start = di
        else:
            start = (di + 3) % n  # UTG

        for i in range(n):
            p = self.player_ids[(start + i) % n]
            if p in active:
                return p
        return self.player_ids[0]

    def get_next_role(self, state: State) -> str:
        e = state["extras"]
        current = e.get("current_actor_id", self.player_ids[0])
        idx = self.player_ids.index(current)
        n = self.num_players

        for i in range(1, n + 1):
            p = self.player_ids[(idx + i) % n]
            if p not in e["folded"] and e["chips"][p] > 0:
                return p
        return current

    # ---- Prompts ----

    async def build_prompt(self, role: str, state: State) -> Messages:
        e = state["extras"]
        to_call = e["current_bet"] - e["bets_round"][role]
        hand_num = e["hands_played"] + 1
        total_pot = sum(e["contributions"].values())

        opp_lines = []
        for p in self.player_ids:
            if p != role:
                status = "folded" if p in e["folded"] else ("all-in" if e["chips"][p] == 0 else "active")
                opp_lines.append(f"  {p}: {e['chips'][p]} chips ({status})")

        situation = (
            f"=== HAND {hand_num} - {e['phase'].upper()} ===\n"
            f"Your hole cards: {format_cards(e['hole_cards'].get(role, []))}\n"
            f"Community cards: {format_cards(e['community'])}\n\n"
            f"Pot: {total_pot} | Your chips: {e['chips'][role]}\n"
            f"Current bet: {e['current_bet']} | To call: {to_call}\n\n"
            f"Opponents:\n" + "\n".join(opp_lines) + "\n\nYour action?"
        )

        # System prompt comes from agent config — but task provides the game context
        return [
            {"role": "user", "content": situation},
        ]

    # ---- Game Logic ----

    async def on_turn_complete(self, state: State) -> None:
        if not state["trajectory"]:
            return
        last_step = state["trajectory"][-1]
        actor_id = last_step.get("extras", {}).get("actor_id")
        if not actor_id:
            return

        completion = last_step.get("completion", [])
        if not completion:
            return
        content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])

        e = state["extras"]
        action = self._parse_action(content, e, actor_id)
        action = self._validate_action(action, e, actor_id)
        e["actions_hand"] += 1
        e["actions_round"][actor_id] += 1

        self._process_action(action, e, actor_id)

        if self._betting_complete(e):
            self._advance_phase(state)

    def _parse_action(self, text, extras, actor_id):
        try:
            m = re.search(r"\{[^}]+\}", text)
            if m:
                action = json.loads(m.group(0))
                if "action" in action:
                    action["action"] = action["action"].lower()
                    if action["action"] == "raise" and "amount" in action:
                        action["amount"] = int(action["amount"])
                    return action
        except (json.JSONDecodeError, ValueError):
            pass
        to_call = extras["current_bet"] - extras["bets_round"][actor_id]
        return {"action": "check"} if to_call == 0 else {"action": "fold"}

    def _validate_action(self, action, extras, actor_id):
        chips = extras["chips"][actor_id]
        to_call = extras["current_bet"] - extras["bets_round"][actor_id]
        act = action["action"]

        if act == "check" and to_call > 0:
            return {"action": "call"} if chips >= to_call else {"action": "allin"}
        if act == "call" and chips <= to_call:
            return {"action": "allin"}
        if act == "raise":
            amt = action.get("amount", 0)
            max_raise = chips + extras["bets_round"][actor_id]
            if amt >= max_raise:
                return {"action": "allin"}
            min_raise = extras["current_bet"] + self.big_blind
            action["amount"] = max(min(amt, max_raise), min_raise)
        return action

    def _process_action(self, action, extras, actor_id):
        act = action["action"]
        if act == "fold":
            extras["folded"].append(actor_id)
        elif act == "call":
            to_call = extras["current_bet"] - extras["bets_round"][actor_id]
            amt = min(to_call, extras["chips"][actor_id])
            extras["chips"][actor_id] -= amt
            extras["bets_round"][actor_id] += amt
            extras["contributions"][actor_id] += amt
        elif act == "raise":
            additional = action["amount"] - extras["bets_round"][actor_id]
            additional = min(additional, extras["chips"][actor_id])
            extras["chips"][actor_id] -= additional
            extras["bets_round"][actor_id] += additional
            extras["current_bet"] = extras["bets_round"][actor_id]
            extras["contributions"][actor_id] += additional
            for p in self.player_ids:
                if p != actor_id and p not in extras["folded"] and extras["chips"][p] > 0:
                    extras["actions_round"][p] = 0
        elif act == "allin":
            amt = extras["chips"][actor_id]
            extras["bets_round"][actor_id] += amt
            extras["chips"][actor_id] = 0
            extras["contributions"][actor_id] += amt
            if extras["bets_round"][actor_id] > extras["current_bet"]:
                extras["current_bet"] = extras["bets_round"][actor_id]
                for p in self.player_ids:
                    if p != actor_id and p not in extras["folded"] and extras["chips"][p] > 0:
                        extras["actions_round"][p] = 0

    def _betting_complete(self, extras) -> bool:
        if extras["phase"] == "complete":
            return False
        active = [p for p in self.player_ids if p not in extras["folded"]]
        if len(active) <= 1:
            return False
        can_act = [p for p in active if extras["chips"][p] > 0]
        if not can_act:
            return True
        for p in can_act:
            if extras["actions_round"][p] == 0:
                return False
            if extras["bets_round"][p] < extras["current_bet"]:
                return False
        return True

    def _advance_phase(self, state: State):
        e = state["extras"]
        e["bets_round"] = {p: 0 for p in self.player_ids}
        e["current_bet"] = 0
        e["actions_round"] = {p: 0 for p in self.player_ids}

        phases = ["preflop", "flop", "turn", "river", "showdown"]
        idx = phases.index(e["phase"])
        e["phase"] = phases[idx + 1]

        if e["phase"] == "flop":
            e["deck"].pop()
            e["community"].extend([e["deck"].pop() for _ in range(3)])
        elif e["phase"] in ("turn", "river"):
            e["deck"].pop()
            e["community"].append(e["deck"].pop())
        elif e["phase"] == "showdown":
            self._resolve_showdown(state)

    def _resolve_showdown(self, state: State):
        e = state["extras"]
        active = [p for p in self.player_ids if p not in e["folded"]]
        community = e["community"]

        # Deal remaining community cards if needed
        while len(community) < 5 and e["deck"]:
            e["deck"].pop()
            if e["deck"]:
                community.append(e["deck"].pop())

        if len(active) == 1:
            winner = active[0]
            e["chips"][winner] += sum(e["contributions"].values())
        else:
            winners, evals = find_winners(active, e["hole_cards"], community)
            pot = sum(e["contributions"].values())
            share = pot // len(winners)
            for w in winners:
                e["chips"][w] += share

        e["hands_played"] += 1
        if e["hands_played"] < self.num_hands:
            has_chips = [p for p in self.player_ids if e["chips"][p] > 0]
            if len(has_chips) >= 2:
                e["dealer_idx"] = (e["dealer_idx"] + 1) % self.num_players
                self._start_hand(state)
                return

        e["phase"] = "complete"

    # ---- Stop Conditions ----

    async def should_stop(self, state: State) -> bool:
        e = state.get("extras", {})
        if e.get("phase") == "complete":
            return True
        active = [p for p in self.player_ids if p not in e.get("folded", [])]
        if len(active) <= 1:
            # Award pot to last player standing
            if active:
                e["chips"][active[0]] += sum(e.get("contributions", {}).values())
            e["hands_played"] = e.get("hands_played", 0) + 1
            e["phase"] = "complete"
            return True
        if e.get("actions_hand", 0) >= self.max_actions:
            self._resolve_showdown(state)
            return True
        return False

    async def on_game_end(self, state: State) -> None:
        e = state["extras"]
        ranking = sorted(self.player_ids, key=lambda p: e["chips"][p], reverse=True)
        e["final_ranking"] = ranking
        e["session_winner"] = ranking[0]
        e["profits"] = {p: e["chips"][p] - self.starting_chips for p in self.player_ids}


# =============================================================================
# Agent Factory
# =============================================================================

def create_agents(num_players: int) -> dict[str, Agent]:
    agents = {}
    for i in range(num_players):
        pid = f"player{i + 1}"
        config = PLAYER_CONFIGS[i] if i < len(PLAYER_CONFIGS) else None

        if config:
            client, model = get_actor_client(config["endpoint"])
            strategy = config.get("strategy", "balanced")
            is_trainable = config.get("is_trainable", False)
        else:
            client, model = None, None
            strategy = "balanced"
            is_trainable = False

        system_prompt = (
            f"You are playing {num_players}-player No-Limit Texas Hold'em Poker.\n\n"
            f"{STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS['balanced'])}\n\n"
            "On your turn, output ONLY a JSON object:\n"
            '- Fold: {"action": "fold"}\n'
            '- Check: {"action": "check"}\n'
            '- Call: {"action": "call"}\n'
            '- Raise: {"action": "raise", "amount": 100}\n'
            '- All-in: {"action": "allin"}\n\n'
            "Output ONLY the JSON, nothing else."
        )

        agents[pid] = Agent(
            id=pid,
            system_prompt=system_prompt,
            max_tokens=50,
            is_trainable=is_trainable,
            model=model,
            client=client,
        )

    return agents


# =============================================================================
# Environment Loader
# =============================================================================

def load_environment(
    num_players: int = 6,
    num_hands: int = 1,
    max_actions: int = 50,
    starting_chips: int = 1000,
    small_blind: int = 5,
    big_blind: int = 10,
    num_examples: int = -1,
):
    """
    Composition:
        Task   = PokerTask (full Texas Hold'em rules + scoring)
        Agents = {player1..N} with strategy system prompts
        Env    = MultiAgentEnv(task, agents)
    """
    task = PokerTask(
        num_players=num_players,
        num_hands=num_hands,
        max_actions=max_actions,
        starting_chips=starting_chips,
        small_blind=small_blind,
        big_blind=big_blind,
        num_examples=num_examples,
    )

    agents = create_agents(num_players)

    return MultiAgentEnv(
        task=task,
        agents=agents,
        max_turns=num_hands * max_actions * num_players + 10,
    )
