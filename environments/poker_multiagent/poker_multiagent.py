from __future__ import annotations

import itertools
import random
from collections import Counter

from datasets import Dataset

import verifiers as vf
from verifiers.types import Messages, State, SystemMessage, UserMessage


class PokerMultiAgentEnv(vf.MultiAgentEnv):
    SUITS = "cdhs"
    RANKS = "23456789TJQKA"
    SYSTEM_PROMPT = (
        "You are a poker player in a single-hand no-limit Texas Hold'em game. "
        "On your turn, call the take_action tool exactly once with one of: fold, check, call, raise. "
        "For raise, provide amount as raise_to total chips committed by you this street."
    )

    def __init__(
        self,
        num_players: int = 4,
        starting_stack: int = 100,
        small_blind: int = 1,
        big_blind: int = 2,
        max_raises_per_street: int = 2,
        seed: int | None = None,
        max_turns: int = 120,
        **kwargs,
    ):
        if num_players < 2 or num_players > 9:
            raise ValueError("num_players must be between 2 and 9")
        if starting_stack <= 0:
            raise ValueError("starting_stack must be positive")
        if small_blind <= 0 or big_blind <= 0:
            raise ValueError("small_blind and big_blind must be positive")
        if big_blind < small_blind:
            raise ValueError("big_blind must be >= small_blind")
        if starting_stack < big_blind:
            raise ValueError("starting_stack must be >= big_blind")
        if max_raises_per_street < 0:
            raise ValueError("max_raises_per_street must be non-negative")

        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_raises_per_street = max_raises_per_street
        self.seed = seed

        super().__init__(tools=[], max_turns=max_turns, **kwargs)
        self.add_tool(self.take_action, args_to_skip=["state"])

    def _append_hand_log(self, state: State, line: str) -> None:
        if "hand_log_lines" not in state:
            state["hand_log_lines"] = []
        state["hand_log_lines"].append(line)

    def _log_snapshot(self, state: State, label: str) -> None:
        board = (
            " ".join(state["community_cards"]) if state["community_cards"] else "(none)"
        )
        self._append_hand_log(
            state,
            f"[{label}] street={state['street']} pot={state['pot']} current_bet={state['current_bet']} pending={state['pending_to_act']}",
        )
        self._append_hand_log(state, f"board={board}")
        for actor_id in state["seats"]:
            player = state["players"][actor_id]
            hole_cards = " ".join(player["hole_cards"])
            self._append_hand_log(
                state,
                f"{actor_id}: stack={player['stack']} folded={player['folded']} street_contrib={state['street_contrib'][actor_id]} hole_cards={hole_cards}",
            )
        self._append_hand_log(state, f"action_log_tail={state['action_log'][-8:]}")

    def _action_result(
        self,
        state: State,
        actor_id: str,
        action: str,
        message: str,
    ) -> str:
        self._append_hand_log(
            state, f"result actor={actor_id} action={action} message={message}"
        )
        self._log_snapshot(state, "post_action")
        return message

    def _coerce_optional_int(self, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                return None
            lowered = text.lower()
            if lowered in {"none", "null"}:
                return None
            try:
                if "." in text or "e" in lowered:
                    parsed = float(text)
                    if parsed.is_integer():
                        return int(parsed)
                    return None
                return int(text)
            except ValueError:
                return None
        return None

    def _actor_id(self, seat: int) -> str:
        return f"player_{seat}"

    def _seat(self, actor_id: str) -> int:
        return int(actor_id.rsplit("_", 1)[1])

    def _seat_cycle(self, start_seat: int) -> list[int]:
        return [
            (start_seat + offset) % self.num_players
            for offset in range(self.num_players)
        ]

    def _active_players(self, state: State) -> list[str]:
        return [
            actor_id
            for actor_id in state["seats"]
            if not state["players"][actor_id]["folded"]
        ]

    def _actionable_players_from(
        self, state: State, start_seat: int, exclude: set[str] | None = None
    ) -> list[str]:
        excluded = exclude or set()
        actor_ids: list[str] = []
        for seat in self._seat_cycle(start_seat):
            actor_id = self._actor_id(seat)
            player = state["players"][actor_id]
            if actor_id in excluded:
                continue
            if player["folded"]:
                continue
            if player["stack"] <= 0:
                continue
            actor_ids.append(actor_id)
        return actor_ids

    def _build_deck(self) -> list[str]:
        return [rank + suit for rank in self.RANKS for suit in self.SUITS]

    def _deal_cards(self, state: State, count: int) -> list[str]:
        deck = state["deck"]
        if len(deck) < count:
            raise ValueError("Deck does not have enough cards")
        return [deck.pop() for _ in range(count)]

    def _post_forced_bet(
        self, state: State, actor_id: str, amount: int, label: str
    ) -> None:
        posted = min(state["players"][actor_id]["stack"], amount)
        state["players"][actor_id]["stack"] -= posted
        state["street_contrib"][actor_id] += posted
        state["pot"] += posted
        state["action_log"].append(f"{actor_id} posts {label}: {posted}")
        self._append_hand_log(
            state,
            f"forced_bet actor={actor_id} label={label} requested={amount} posted={posted}",
        )

    def _small_blind_seat(self) -> int:
        if self.num_players == 2:
            return 0
        return 1

    def _big_blind_seat(self) -> int:
        if self.num_players == 2:
            return 1
        return 2

    def _first_preflop_seat(self) -> int:
        if self.num_players == 2:
            return 0
        return (self._big_blind_seat() + 1) % self.num_players

    def get_all_actors(self, state: State) -> dict[str, str]:
        return {self._actor_id(i): self.SYSTEM_PROMPT for i in range(self.num_players)}

    def get_initial_actor_id(self, actors: dict[str, str], state: State) -> str:
        return self._actor_id(self._first_preflop_seat())

    def get_next_actor_id(self, state: State) -> str:
        pending = state.get("pending_to_act", [])
        if pending:
            return pending[0]
        return state["trajectory_id"]

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)

        if self.seed is not None:
            seed = self.seed
        else:
            seed = int(state.get("input", {}).get("seed", state.get("example_id", 0)))
        rng = random.Random(seed)

        seats = [self._actor_id(i) for i in range(self.num_players)]
        players = {
            actor_id: {
                "seat": self._seat(actor_id),
                "stack": self.starting_stack,
                "folded": False,
                "hole_cards": [],
            }
            for actor_id in seats
        }

        deck = self._build_deck()
        rng.shuffle(deck)

        state["seed"] = seed
        state["seats"] = seats
        state["players"] = players
        state["deck"] = deck
        state["community_cards"] = []
        state["dealer_seat"] = 0
        state["small_blind_seat"] = self._small_blind_seat()
        state["big_blind_seat"] = self._big_blind_seat()
        state["pot"] = 0
        state["street"] = "preflop"
        state["street_contrib"] = {actor_id: 0 for actor_id in seats}
        state["current_bet"] = 0
        state["last_raise_size"] = self.big_blind
        state["street_raise_count"] = 0
        state["pending_to_act"] = []
        state["action_log"] = []
        state["hand_log_lines"] = []
        state["deck_initial_order"] = list(deck)
        state["max_possible_payout"] = self.num_players * self.starting_stack
        state["winner_winnings"] = 0
        state["winner_winnings_by_player"] = {}
        state["player_streets_seen"] = {actor_id: 1 for actor_id in seats}

        for _ in range(2):
            for actor_id in seats:
                players[actor_id]["hole_cards"].append(state["deck"].pop())

        small_blind_actor = self._actor_id(state["small_blind_seat"])
        big_blind_actor = self._actor_id(state["big_blind_seat"])
        self._post_forced_bet(state, small_blind_actor, self.small_blind, "small blind")
        self._post_forced_bet(state, big_blind_actor, self.big_blind, "big blind")
        state["current_bet"] = max(state["street_contrib"].values())

        first_preflop_actor = self._actor_id(self._first_preflop_seat())
        state["pending_to_act"] = self._actionable_players_from(
            state,
            self._first_preflop_seat(),
        )
        state["trajectory_id"] = first_preflop_actor

        self.logger.debug(
            "poker.setup seed=%s players=%s stack=%s sb=%s bb=%s first_actor=%s",
            seed,
            self.num_players,
            self.starting_stack,
            small_blind_actor,
            big_blind_actor,
            first_preflop_actor,
        )
        self._append_hand_log(
            state,
            f"setup seed={seed} num_players={self.num_players} starting_stack={self.starting_stack} small_blind={self.small_blind} big_blind={self.big_blind}",
        )
        self._append_hand_log(
            state, f"deck_initial_order={' '.join(state['deck_initial_order'])}"
        )
        self._log_snapshot(state, "setup_complete")
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict:
        if tool_name == "take_action":
            return {**tool_args, "state": state}
        return tool_args

    def _min_raise_to(self, state: State, actor_id: str) -> int:
        current_bet = state["current_bet"]
        if current_bet == 0:
            return self.big_blind
        return current_bet + state["last_raise_size"]

    def _legal_options(self, actor_id: str, state: State) -> dict:
        player = state["players"][actor_id]
        actor_contrib = state["street_contrib"][actor_id]
        to_call = max(0, state["current_bet"] - actor_contrib)
        stack = player["stack"]

        can_check = to_call == 0
        can_call = to_call > 0 and stack >= to_call

        min_raise_to = self._min_raise_to(state, actor_id)
        max_raise_to = actor_contrib + stack
        can_raise = (
            state["street_raise_count"] < self.max_raises_per_street
            and stack > to_call
            and min_raise_to <= max_raise_to
        )

        legal_actions = ["fold"]
        if can_check:
            legal_actions.append("check")
        if can_call:
            legal_actions.append("call")
        if can_raise:
            legal_actions.append("raise")

        return {
            "to_call": to_call,
            "can_check": can_check,
            "can_call": can_call,
            "can_raise": can_raise,
            "min_raise_to": min_raise_to if can_raise else None,
            "max_raise_to": max_raise_to if can_raise else None,
            "legal_actions": legal_actions,
        }

    def _render_game_state_prompt(self, actor_id: str, state: State) -> str:
        player = state["players"][actor_id]
        options = self._legal_options(actor_id, state)

        player_lines = []
        for pid in state["seats"]:
            p = state["players"][pid]
            status = "folded" if p["folded"] else "active"
            player_lines.append(
                f"- {pid}: stack={p['stack']}, status={status}, street_commit={state['street_contrib'][pid]}"
            )

        board = (
            " ".join(state["community_cards"]) if state["community_cards"] else "(none)"
        )
        hole_cards = " ".join(player["hole_cards"])
        action_log = state["action_log"][-12:]
        log_text = (
            "\n".join(f"- {entry}" for entry in action_log)
            if action_log
            else "- (none)"
        )

        raise_instruction = ""
        if options["can_raise"]:
            raise_instruction = (
                f"raise is legal with amount in [{options['min_raise_to']}, {options['max_raise_to']}], "
                "where amount means raise_to total chips committed by you this street."
            )
        else:
            raise_instruction = "raise is not legal right now."

        return (
            f"Street: {state['street']}\n"
            f"Pot: {state['pot']}\n"
            f"Board: {board}\n"
            f"You are: {actor_id}\n"
            f"Your hole cards: {hole_cards}\n"
            f"Your stack: {player['stack']}\n"
            f"Your street contribution: {state['street_contrib'][actor_id]}\n"
            f"To call: {options['to_call']}\n"
            f"can_check={options['can_check']} can_call={options['can_call']} can_raise={options['can_raise']}\n"
            f"{raise_instruction}\n"
            f"Legal actions: {', '.join(options['legal_actions'])}\n"
            "\nPlayers:\n"
            f"{chr(10).join(player_lines)}\n"
            "\nRecent action log:\n"
            f"{log_text}\n"
        )

    def get_prompt_for_actor(self, messages: Messages, state: State) -> Messages:
        actor_id = state["trajectory_id"]
        system_prompt = state["system_prompts"][actor_id]
        game_state_message = UserMessage(
            content=self._render_game_state_prompt(actor_id, state)
        )
        if len(messages) == 0:
            return [SystemMessage(content=system_prompt), game_state_message]
        return [game_state_message]

    def _remove_from_pending(self, state: State, actor_id: str) -> None:
        state["pending_to_act"] = [
            pid for pid in state["pending_to_act"] if pid != actor_id
        ]

    def _commit_chips(self, state: State, actor_id: str, chips: int) -> None:
        state["players"][actor_id]["stack"] -= chips
        state["street_contrib"][actor_id] += chips
        state["pot"] += chips

    def _fold_player(self, state: State, actor_id: str, reason: str) -> None:
        if not state["players"][actor_id]["folded"]:
            state["players"][actor_id]["folded"] = True
        self._remove_from_pending(state, actor_id)
        state["action_log"].append(f"{actor_id} folds ({reason})")

    def _start_new_street(self, state: State, street: str) -> None:
        state["street"] = street
        state["street_contrib"] = {pid: 0 for pid in state["seats"]}
        state["current_bet"] = 0
        state["last_raise_size"] = self.big_blind
        state["street_raise_count"] = 0
        for actor_id in state["seats"]:
            if not state["players"][actor_id]["folded"]:
                state["player_streets_seen"][actor_id] += 1
        first_to_act = (state["dealer_seat"] + 1) % self.num_players
        state["pending_to_act"] = self._actionable_players_from(state, first_to_act)

    def _advance_street(self, state: State) -> None:
        street = state["street"]
        if street == "preflop":
            new_cards = self._deal_cards(state, 3)
            state["community_cards"].extend(new_cards)
            state["action_log"].append(f"Dealer deals flop: {' '.join(new_cards)}")
            self._append_hand_log(
                state, f"street_transition flop cards={' '.join(new_cards)}"
            )
            self._start_new_street(state, "flop")
            self.logger.debug(
                "poker.street advanced=flop board=%s", state["community_cards"]
            )
            return
        if street == "flop":
            new_card = self._deal_cards(state, 1)
            state["community_cards"].extend(new_card)
            state["action_log"].append(f"Dealer deals turn: {new_card[0]}")
            self._append_hand_log(state, f"street_transition turn card={new_card[0]}")
            self._start_new_street(state, "turn")
            self.logger.debug(
                "poker.street advanced=turn board=%s", state["community_cards"]
            )
            return
        if street == "turn":
            new_card = self._deal_cards(state, 1)
            state["community_cards"].extend(new_card)
            state["action_log"].append(f"Dealer deals river: {new_card[0]}")
            self._append_hand_log(state, f"street_transition river card={new_card[0]}")
            self._start_new_street(state, "river")
            self.logger.debug(
                "poker.street advanced=river board=%s", state["community_cards"]
            )
            return
        if street == "river":
            self._run_showdown(state)
            return

    def _finalize_single_winner(self, state: State, winner: str) -> None:
        winnings = state["pot"]
        state["players"][winner]["stack"] += winnings
        state["winner_winnings"] = winnings
        state["winner_winnings_by_player"] = {winner: winnings}
        state["pot"] = 0
        state["street"] = "finished"
        state["pending_to_act"] = []
        state["final_env_response"] = (
            f"Hand finished. Winner by folds: {winner}. Winnings: {winnings}. "
            f"Final board: {' '.join(state['community_cards']) if state['community_cards'] else '(none)'}"
        )
        state["action_log"].append(f"Hand ends: {winner} wins by folds")
        self._append_hand_log(
            state, f"hand_end reason=folds winner={winner} winnings={winnings}"
        )
        self._log_snapshot(state, "hand_finished_folds")
        self.logger.info("poker.finish reason=folds winner=%s", winner)

    def _hand_rank_name(self, category: int) -> str:
        names = {
            8: "straight flush",
            7: "four of a kind",
            6: "full house",
            5: "flush",
            4: "straight",
            3: "three of a kind",
            2: "two pair",
            1: "one pair",
            0: "high card",
        }
        return names[category]

    def _run_showdown(self, state: State) -> None:
        if len(state["community_cards"]) < 5:
            missing = 5 - len(state["community_cards"])
            new_cards = self._deal_cards(state, missing)
            state["community_cards"].extend(new_cards)
            state["action_log"].append(
                f"Dealer deals remaining board: {' '.join(new_cards)}"
            )
            self._append_hand_log(
                state, f"street_transition showdown_fill cards={' '.join(new_cards)}"
            )

        contenders = self._active_players(state)
        scores = {
            actor_id: self._evaluate_seven(
                state["players"][actor_id]["hole_cards"] + state["community_cards"]
            )
            for actor_id in contenders
        }
        for actor_id in contenders:
            hole_cards = " ".join(state["players"][actor_id]["hole_cards"])
            self._append_hand_log(
                state,
                f"showdown_hand actor={actor_id} hole={hole_cards} score={scores[actor_id]}",
            )
        best_score = max(scores.values())
        winners = [
            actor_id for actor_id, score in scores.items() if score == best_score
        ]

        pot = state["pot"]
        share = pot // len(winners)
        remainder = pot % len(winners)
        payouts = {actor_id: share for actor_id in winners}

        for actor_id in winners:
            state["players"][actor_id]["stack"] += share

        if remainder > 0:
            start_seat = (state["dealer_seat"] + 1) % self.num_players
            for seat in self._seat_cycle(start_seat):
                actor_id = self._actor_id(seat)
                if actor_id in winners:
                    state["players"][actor_id]["stack"] += 1
                    payouts[actor_id] += 1
                    remainder -= 1
                    if remainder == 0:
                        break

        state["winner_winnings"] = max(payouts.values()) if payouts else 0
        state["winner_winnings_by_player"] = payouts
        state["pot"] = 0
        state["street"] = "finished"
        state["pending_to_act"] = []

        category_name = self._hand_rank_name(best_score[0])
        winners_text = ", ".join(winners)
        board_text = " ".join(state["community_cards"])
        state["final_env_response"] = (
            f"Hand finished at showdown. Winners: {winners_text}. "
            f"Best hand category: {category_name}. Final board: {board_text}."
        )
        state["action_log"].append(
            f"Showdown winners: {winners_text} ({category_name})"
        )
        self._append_hand_log(
            state,
            f"hand_end reason=showdown winners={winners_text} category={category_name} best_score={best_score}",
        )
        self._log_snapshot(state, "hand_finished_showdown")
        self.logger.info(
            "poker.showdown winners=%s category=%s", winners_text, category_name
        )

    def _progress_game(self, state: State) -> None:
        while state.get("final_env_response") is None:
            active = self._active_players(state)
            if len(active) == 1:
                self._finalize_single_winner(state, active[0])
                return

            state["pending_to_act"] = [
                actor_id
                for actor_id in state["pending_to_act"]
                if not state["players"][actor_id]["folded"]
                and state["players"][actor_id]["stack"] > 0
            ]

            if state["pending_to_act"]:
                return

            if state["street"] in {"preflop", "flop", "turn", "river"}:
                self._advance_street(state)
                continue
            return

    async def take_action(
        self,
        action: str,
        amount: int | None = None,
        state: State | None = None,
    ) -> str:
        if state is None:
            return "Error: missing state"
        if state.get("final_env_response") is not None:
            return "Hand is already finished."

        actor_id = state["trajectory_id"]
        action = action.strip().lower()
        raw_amount = amount
        amount = self._coerce_optional_int(raw_amount)
        player = state["players"][actor_id]
        actor_contrib = state["street_contrib"][actor_id]
        to_call = max(0, state["current_bet"] - actor_contrib)
        self._append_hand_log(
            state,
            f"action actor={actor_id} action={action} amount_raw={raw_amount!r} amount={amount} to_call={to_call} stack={player['stack']} street={state['street']}",
        )

        self.logger.debug(
            "poker.action actor=%s action=%s amount_raw=%r amount=%s to_call=%s stack=%s street=%s",
            actor_id,
            action,
            raw_amount,
            amount,
            to_call,
            player["stack"],
            state["street"],
        )

        if actor_id not in state["pending_to_act"]:
            self._fold_player(state, actor_id, "acted out of turn")
            self.logger.warning(
                "poker.action illegal_fold actor=%s reason=out_of_turn", actor_id
            )
            self._progress_game(state)
            return self._action_result(
                state, actor_id, action, "Illegal action (out of turn). Player folded."
            )

        if action == "fold":
            self._fold_player(state, actor_id, "chose fold")
            self._progress_game(state)
            return self._action_result(state, actor_id, action, "Player folded.")

        if action == "check":
            if to_call != 0:
                self._fold_player(state, actor_id, "illegal check while facing bet")
                self.logger.warning(
                    "poker.action illegal_fold actor=%s reason=illegal_check_to_call_%s",
                    actor_id,
                    to_call,
                )
                self._progress_game(state)
                return self._action_result(
                    state,
                    actor_id,
                    action,
                    "Illegal action (check while facing a bet). Player folded.",
                )
            self._remove_from_pending(state, actor_id)
            state["action_log"].append(f"{actor_id} checks")
            self._progress_game(state)
            return self._action_result(state, actor_id, action, "Check accepted.")

        if action == "call":
            if to_call <= 0:
                self._fold_player(state, actor_id, "illegal call with nothing to call")
                self.logger.warning(
                    "poker.action illegal_fold actor=%s reason=illegal_call_zero",
                    actor_id,
                )
                self._progress_game(state)
                return self._action_result(
                    state,
                    actor_id,
                    action,
                    "Illegal action (nothing to call). Player folded.",
                )
            if player["stack"] < to_call:
                self._fold_player(state, actor_id, "illegal call without enough chips")
                self.logger.warning(
                    "poker.action illegal_fold actor=%s reason=insufficient_for_call",
                    actor_id,
                )
                self._progress_game(state)
                return self._action_result(
                    state,
                    actor_id,
                    action,
                    "Illegal action (insufficient chips to call). Player folded.",
                )
            self._commit_chips(state, actor_id, to_call)
            self._remove_from_pending(state, actor_id)
            state["action_log"].append(f"{actor_id} calls {to_call}")
            self._progress_game(state)
            return self._action_result(
                state, actor_id, action, f"Call accepted for {to_call}."
            )

        if action == "raise":
            if state["street_raise_count"] >= self.max_raises_per_street:
                self._fold_player(state, actor_id, "raise cap reached")
                self.logger.warning(
                    "poker.action illegal_fold actor=%s reason=raise_cap", actor_id
                )
                self._progress_game(state)
                return self._action_result(
                    state,
                    actor_id,
                    action,
                    "Illegal action (raise cap reached). Player folded.",
                )
            if not isinstance(amount, int):
                self._fold_player(
                    state, actor_id, "missing or non-integer raise amount"
                )
                self.logger.warning(
                    "poker.action illegal_fold actor=%s reason=bad_raise_amount",
                    actor_id,
                )
                self._progress_game(state)
                return self._action_result(
                    state,
                    actor_id,
                    action,
                    "Illegal action (raise requires integer amount). Player folded.",
                )
            if amount <= state["current_bet"]:
                self._fold_player(state, actor_id, "raise_to must exceed current bet")
                self.logger.warning(
                    "poker.action illegal_fold actor=%s reason=raise_not_above_current",
                    actor_id,
                )
                self._progress_game(state)
                return self._action_result(
                    state,
                    actor_id,
                    action,
                    "Illegal action (raise_to must exceed current bet). Player folded.",
                )

            min_raise_to = self._min_raise_to(state, actor_id)
            if amount < min_raise_to:
                self._fold_player(
                    state, actor_id, f"raise_to below minimum {min_raise_to}"
                )
                self.logger.warning(
                    "poker.action illegal_fold actor=%s reason=raise_below_min",
                    actor_id,
                )
                self._progress_game(state)
                return self._action_result(
                    state,
                    actor_id,
                    action,
                    f"Illegal action (raise_to below minimum {min_raise_to}). Player folded.",
                )

            delta = amount - actor_contrib
            if delta <= to_call:
                self._fold_player(state, actor_id, "raise does not exceed call amount")
                self.logger.warning(
                    "poker.action illegal_fold actor=%s reason=raise_not_real_raise",
                    actor_id,
                )
                self._progress_game(state)
                return self._action_result(
                    state,
                    actor_id,
                    action,
                    "Illegal action (raise must exceed call amount). Player folded.",
                )
            if delta > player["stack"]:
                self._fold_player(state, actor_id, "insufficient chips for raise")
                self.logger.warning(
                    "poker.action illegal_fold actor=%s reason=insufficient_for_raise",
                    actor_id,
                )
                self._progress_game(state)
                return self._action_result(
                    state,
                    actor_id,
                    action,
                    "Illegal action (insufficient chips to raise). Player folded.",
                )

            previous_bet = state["current_bet"]
            self._commit_chips(state, actor_id, delta)
            state["current_bet"] = amount
            state["last_raise_size"] = amount - previous_bet
            state["street_raise_count"] += 1
            state["action_log"].append(f"{actor_id} raises to {amount}")

            start_seat = (self._seat(actor_id) + 1) % self.num_players
            state["pending_to_act"] = self._actionable_players_from(
                state,
                start_seat,
                exclude={actor_id},
            )

            self._progress_game(state)
            return self._action_result(
                state, actor_id, action, f"Raise accepted to {amount}."
            )

        self._fold_player(state, actor_id, f"unknown action '{action}'")
        self.logger.warning(
            "poker.action illegal_fold actor=%s reason=unknown_action", actor_id
        )
        self._progress_game(state)
        return self._action_result(
            state, actor_id, action, f"Illegal action ({action}). Player folded."
        )

    def _rank_values(self, cards: list[str]) -> list[int]:
        rank_map = {rank: index + 2 for index, rank in enumerate(self.RANKS)}
        return [rank_map[card[0]] for card in cards]

    def _straight_high(self, ranks: list[int]) -> int | None:
        unique = sorted(set(ranks))
        if len(unique) != 5:
            return None
        if unique[-1] - unique[0] == 4:
            return unique[-1]
        if unique == [2, 3, 4, 5, 14]:
            return 5
        return None

    def _evaluate_five(self, cards: list[str]) -> tuple[int, tuple[int, ...]]:
        ranks = self._rank_values(cards)
        suits = [card[1] for card in cards]
        counts = Counter(ranks)

        flush = len(set(suits)) == 1
        straight_high = self._straight_high(ranks)

        by_count = sorted(
            counts.items(), key=lambda item: (item[1], item[0]), reverse=True
        )
        count_values = sorted(counts.values(), reverse=True)

        if flush and straight_high is not None:
            return 8, (straight_high,)

        if count_values == [4, 1]:
            four_rank = by_count[0][0]
            kicker = by_count[1][0]
            return 7, (four_rank, kicker)

        if count_values == [3, 2]:
            trip_rank = by_count[0][0]
            pair_rank = by_count[1][0]
            return 6, (trip_rank, pair_rank)

        if flush:
            return 5, tuple(sorted(ranks, reverse=True))

        if straight_high is not None:
            return 4, (straight_high,)

        if count_values == [3, 1, 1]:
            trip_rank = by_count[0][0]
            kickers = sorted(
                [rank for rank, count in counts.items() if count == 1], reverse=True
            )
            return 3, (trip_rank, *kickers)

        if count_values == [2, 2, 1]:
            pairs = sorted(
                [rank for rank, count in counts.items() if count == 2], reverse=True
            )
            kicker = [rank for rank, count in counts.items() if count == 1][0]
            return 2, (pairs[0], pairs[1], kicker)

        if count_values == [2, 1, 1, 1]:
            pair_rank = [rank for rank, count in counts.items() if count == 2][0]
            kickers = sorted(
                [rank for rank, count in counts.items() if count == 1], reverse=True
            )
            return 1, (pair_rank, *kickers)

        return 0, tuple(sorted(ranks, reverse=True))

    def _evaluate_seven(self, cards: list[str]) -> tuple[int, tuple[int, ...]]:
        best_score: tuple[int, tuple[int, ...]] | None = None
        for combo in itertools.combinations(cards, 5):
            score = self._evaluate_five(list(combo))
            if best_score is None or score > best_score:
                best_score = score
        assert best_score is not None
        return best_score


def load_environment(
    seed: int | None = None,
    num_seed_rows: int = 500,
) -> vf.Environment:
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "Poker hand setup placeholder.",
                    }
                ],
                "task": "poker_multiagent",
                "seed": row_seed,
            }
            for row_seed in range(num_seed_rows)
        ]
    )

    async def winner_winnings_reward(state: State) -> float:
        if state.get("error") is not None or state.get("final_env_response") is None:
            return -0.5

        max_possible_payout = state.get("max_possible_payout", 0)
        winner_winnings = state.get("winner_winnings", 0)
        if (
            not isinstance(max_possible_payout, (int, float))
            or max_possible_payout <= 0
        ):
            return 0.0
        if not isinstance(winner_winnings, (int, float)):
            return 0.0

        normalized = winner_winnings / max_possible_payout
        return max(0.0, min(1.0, float(normalized)))

    async def streets_seen_reward(state: State) -> float:
        """Reward for total streets seen across all players. Encourages staying in hands."""
        if state.get("error") is not None or state.get("final_env_response") is None:
            return 0.0
        player_streets = state.get("player_streets_seen", {})
        if not player_streets:
            return 0.0
        num_players = len(state["seats"])
        # max 4 streets per player: preflop, flop, turn, river
        total = sum(player_streets.values())
        return total / (num_players * 4)

    async def fewer_losers_reward(state: State) -> float:
        """Reward for fewer players losing money. Encourages selective play."""
        if state.get("error") is not None or state.get("final_env_response") is None:
            return 0.0
        num_players = len(state["seats"])
        starting_stack = state.get("max_possible_payout", 0) / max(num_players, 1)
        num_losers = sum(
            1
            for actor_id in state["seats"]
            if state["players"][actor_id]["stack"] < starting_stack
        )
        return 1.0 - (num_losers / num_players)

    rubric = vf.Rubric(
        funcs=[winner_winnings_reward, streets_seen_reward, fewer_losers_reward],
        weights=[1.0, 1.0, 1.0],
    )
    return PokerMultiAgentEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=None,
        seed=seed,
    )
