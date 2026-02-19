from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
from abc import abstractmethod
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf
from verifiers.types import Messages, State, TrajectoryStep
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.message_utils import concat_messages


class MultiAgentEnv(vf.StatefulToolEnv):
    """
    Multi-agent environment on top of StatefulToolEnv.

    `state["trajectory_id"]` is the active actor id.
    """

    HANDOFF_TAG_PATTERN = re.compile(
        r"<handoff>\s*(\{.*\})\s*</handoff>\s*$",
        flags=re.DOTALL,
    )

    @abstractmethod
    def get_initial_actor_id(self, actors: dict[str, str], state: State) -> str:
        pass

    @abstractmethod
    def get_next_actor_id(self, state: State) -> str:
        pass

    @abstractmethod
    def get_all_actors(self, state: State) -> dict[str, str]:
        pass

    @abstractmethod
    def get_handoff_schema(self, actor_id: str, state: State) -> dict[str, Any]:
        """Return JSON-schema-like object for this actor's end-of-turn handoff."""

    @abstractmethod
    async def apply_handoff(
        self, actor_id: str, handoff: dict[str, Any], state: State
    ) -> Messages | str | None:
        """Apply a parsed handoff to state and return environment response messages."""

    @abstractmethod
    def get_prompt_for_actor(self, messages: Messages, state: State) -> Messages:
        pass

    # Override ToolEnv's stop condition. Missing tool calls are valid if a handoff is emitted.
    async def no_tools_called(self, state: vf.State) -> bool:
        return False

    # Default passthrough so subclasses that don't use tool arg mutation can skip overriding.
    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict:
        return tool_args

    def _normalize_messages(self, messages: Messages | str | None) -> Messages:
        if messages is None:
            return []
        if isinstance(messages, str):
            return [{"role": "assistant", "content": messages}]
        if isinstance(messages, list):
            return messages
        raise ValueError(f"Expected list/str messages, got {type(messages).__name__}")

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        actors = self.get_all_actors(state)
        initial_actor_id = self.get_initial_actor_id(actors, state)
        if initial_actor_id not in actors:
            raise ValueError(
                f"Initial actor ID '{initial_actor_id}' not found in actors"
            )

        handoff_schemas: dict[str, dict[str, Any]] = {}
        system_prompts: dict[str, str] = {}
        for actor_id, base_prompt in actors.items():
            schema = self._normalize_handoff_schema(
                self.get_handoff_schema(actor_id, state), actor_id
            )
            handoff_schemas[actor_id] = schema
            system_prompts[actor_id] = self._compose_system_prompt_with_handoff(
                base_prompt, schema
            )

        state["trajectory_id"] = initial_actor_id
        state["system_prompts"] = system_prompts
        state["handoff_schemas"] = handoff_schemas
        state["handoff_history"] = []
        state["last_handoff_by_actor"] = {}
        self.logger.debug(
            "multiagent.setup actors=%s initial_actor=%s",
            list(actors.keys()),
            initial_actor_id,
        )
        return state

    def _compose_system_prompt_with_handoff(
        self, base_prompt: str, schema: dict[str, Any]
    ) -> str:
        schema_text = json.dumps(schema, indent=2, ensure_ascii=True, sort_keys=True)
        return (
            f"{base_prompt}\n\n"
            "Turn contract:\n"
            "- You may call tools as needed during your turn.\n"
            "- To end your turn, output exactly one handoff block in this format:\n"
            "  <handoff>{...}</handoff>\n"
            "- The JSON object must match the handoff schema exactly.\n\n"
            "Handoff schema:\n"
            f"```json\n{schema_text}\n```"
        )

    def _normalize_handoff_schema(
        self, schema: dict[str, Any], actor_id: str
    ) -> dict[str, Any]:
        if not isinstance(schema, dict):
            raise ValueError(f"Handoff schema for actor '{actor_id}' must be a dict")
        normalized = deepcopy(schema)
        if normalized.get("type") != "object":
            raise ValueError(
                f"Handoff schema for actor '{actor_id}' must have type='object'"
            )
        properties = normalized.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ValueError(
                f"Handoff schema for actor '{actor_id}' must define non-empty properties"
            )
        required = normalized.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(k, str) for k in required
        ):
            raise ValueError(
                f"Handoff schema for actor '{actor_id}' must define required as list[str]"
            )
        for required_key in required:
            if required_key not in properties:
                raise ValueError(
                    f"Handoff schema for actor '{actor_id}' has required key '{required_key}' not present in properties"
                )
        if "additionalProperties" not in normalized:
            normalized["additionalProperties"] = False
        return normalized

    def _schema_for_actor(self, actor_id: str, state: State) -> dict[str, Any]:
        handoff_schemas = state.get("handoff_schemas", {})
        if isinstance(handoff_schemas, dict):
            schema = handoff_schemas.get(actor_id)
            if isinstance(schema, dict):
                return schema
        schema = self._normalize_handoff_schema(
            self.get_handoff_schema(actor_id, state), actor_id
        )
        if not isinstance(handoff_schemas, dict):
            handoff_schemas = {}
            state["handoff_schemas"] = handoff_schemas
        handoff_schemas[actor_id] = schema
        return schema

    def _message_field(self, message: Any, field: str, default: Any = None) -> Any:
        if isinstance(message, dict):
            return message.get(field, default)
        return getattr(message, field, default)

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
                    continue
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    chunks.append(part_text)
            return "\n".join(chunks).strip()
        return ""

    def parse_handoff(
        self, actor_id: str, last_message: Any, state: State
    ) -> dict[str, Any]:
        message_text = self._content_to_text(
            self._message_field(last_message, "content")
        )
        if message_text.strip() == "":
            raise vf.InvalidModelResponseError(
                f"Actor '{actor_id}' produced no handoff block. Expected <handoff>{{...}}</handoff>."
            )

        match = self.HANDOFF_TAG_PATTERN.search(message_text)
        if match is None:
            raise vf.InvalidModelResponseError(
                f"Actor '{actor_id}' must end turn with a <handoff>{{...}}</handoff> block."
            )

        handoff_json = match.group(1)
        try:
            payload = json.loads(handoff_json)
        except json.JSONDecodeError as e:
            raise vf.InvalidModelResponseError(
                f"Actor '{actor_id}' emitted invalid handoff JSON: {e.msg} (line {e.lineno}, column {e.colno})."
            ) from e

        if not isinstance(payload, dict):
            raise vf.InvalidModelResponseError(
                f"Actor '{actor_id}' handoff must be a JSON object."
            )

        schema = self._schema_for_actor(actor_id, state)
        self._validate_handoff_payload(payload, schema, actor_id)
        return payload

    def _validate_handoff_payload(
        self, payload: dict[str, Any], schema: dict[str, Any], actor_id: str
    ) -> None:
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise vf.InvalidModelResponseError(
                f"Handoff schema for actor '{actor_id}' has invalid properties."
            )

        required = schema.get("required", [])
        if not isinstance(required, list):
            raise vf.InvalidModelResponseError(
                f"Handoff schema for actor '{actor_id}' has invalid required list."
            )

        for key in required:
            if key not in payload:
                raise vf.InvalidModelResponseError(
                    f"Actor '{actor_id}' handoff missing required key '{key}'."
                )

        additional_allowed = bool(schema.get("additionalProperties", False))
        if not additional_allowed:
            extra_keys = [key for key in payload if key not in properties]
            if extra_keys:
                raise vf.InvalidModelResponseError(
                    f"Actor '{actor_id}' handoff has unsupported keys: {extra_keys}."
                )

        for key, value in payload.items():
            prop_schema = properties.get(key)
            if not isinstance(prop_schema, dict):
                if additional_allowed:
                    continue
                raise vf.InvalidModelResponseError(
                    f"Actor '{actor_id}' handoff key '{key}' is not defined in schema."
                )
            self._validate_property_value(actor_id, key, value, prop_schema)

    def _validate_property_value(
        self, actor_id: str, key: str, value: Any, schema: dict[str, Any]
    ) -> None:
        if "enum" in schema:
            enum_values = schema["enum"]
            if isinstance(enum_values, list) and value not in enum_values:
                raise vf.InvalidModelResponseError(
                    f"Actor '{actor_id}' handoff key '{key}' must be one of {enum_values}, got {value!r}."
                )

        prop_type = schema.get("type")
        if prop_type is not None:
            if isinstance(prop_type, str):
                allowed_types = [prop_type]
            elif isinstance(prop_type, list) and all(
                isinstance(item, str) for item in prop_type
            ):
                allowed_types = prop_type
            else:
                raise vf.InvalidModelResponseError(
                    f"Handoff schema for actor '{actor_id}' key '{key}' has invalid 'type'."
                )

            if not any(self._matches_json_type(value, t) for t in allowed_types):
                allowed = ", ".join(allowed_types)
                raise vf.InvalidModelResponseError(
                    f"Actor '{actor_id}' handoff key '{key}' must be of type [{allowed}], got {type(value).__name__}."
                )

        if isinstance(value, str):
            min_length = schema.get("minLength")
            if isinstance(min_length, int) and len(value) < min_length:
                raise vf.InvalidModelResponseError(
                    f"Actor '{actor_id}' handoff key '{key}' must have minLength={min_length}."
                )
            max_length = schema.get("maxLength")
            if isinstance(max_length, int) and len(value) > max_length:
                raise vf.InvalidModelResponseError(
                    f"Actor '{actor_id}' handoff key '{key}' must have maxLength={max_length}."
                )

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum = schema.get("minimum")
            if isinstance(minimum, (int, float)) and value < minimum:
                raise vf.InvalidModelResponseError(
                    f"Actor '{actor_id}' handoff key '{key}' must be >= {minimum}."
                )
            maximum = schema.get("maximum")
            if isinstance(maximum, (int, float)) and value > maximum:
                raise vf.InvalidModelResponseError(
                    f"Actor '{actor_id}' handoff key '{key}' must be <= {maximum}."
                )

    def _matches_json_type(self, value: Any, json_type: str) -> bool:
        if json_type == "string":
            return isinstance(value, str)
        if json_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if json_type == "number":
            return (
                isinstance(value, int) and not isinstance(value, bool)
            ) or isinstance(value, float)
        if json_type == "boolean":
            return isinstance(value, bool)
        if json_type == "array":
            return isinstance(value, list)
        if json_type == "object":
            return isinstance(value, dict)
        if json_type == "null":
            return value is None
        return False

    def _terminate_on_invalid_handoff(
        self, actor_id: str, error: vf.InvalidModelResponseError, state: State
    ) -> Messages | str | None:
        reason = str(error)
        state["rollout_completed_cleanly"] = False
        state["malformed_handoff"] = {
            "actor": actor_id,
            "reason": reason,
        }
        state["final_env_response"] = (
            f"Rollout ended due to malformed handoff from {actor_id}: {reason}"
        )
        self.logger.warning(
            "multiagent.handoff_invalid actor=%s reason=%s",
            actor_id,
            reason,
        )
        return state["final_env_response"]

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        if not messages:
            raise vf.InvalidModelResponseError("Expected non-empty message history.")

        last_msg = messages[-1]
        if self._message_field(last_msg, "role") != "assistant":
            raise vf.InvalidModelResponseError(
                "Expected assistant message at end of turn for multi-agent handoff."
            )

        actor_id = state["trajectory_id"]
        tool_calls = self._message_field(last_msg, "tool_calls", None) or []
        if isinstance(tool_calls, list) and tool_calls:
            self.logger.debug(
                "multiagent.turn actor=%s mode=tools tool_calls=%s",
                actor_id,
                len(tool_calls),
            )
            return await super().env_response(messages, state, **kwargs)

        try:
            handoff = self.parse_handoff(actor_id, last_msg, state)
        except vf.InvalidModelResponseError as e:
            env_response = self._terminate_on_invalid_handoff(actor_id, e, state)
            return self._normalize_messages(env_response)
        env_response = await maybe_await(self.apply_handoff, actor_id, handoff, state)

        handoff_history = state.get("handoff_history")
        if not isinstance(handoff_history, list):
            handoff_history = []
            state["handoff_history"] = handoff_history
        handoff_history.append({"actor": actor_id, "handoff": handoff})

        last_handoff_by_actor = state.get("last_handoff_by_actor")
        if not isinstance(last_handoff_by_actor, dict):
            last_handoff_by_actor = {}
            state["last_handoff_by_actor"] = last_handoff_by_actor
        last_handoff_by_actor[actor_id] = handoff

        next_actor_id = self.get_next_actor_id(state)
        system_prompts = state.get("system_prompts", {})
        if not isinstance(system_prompts, dict) or next_actor_id not in system_prompts:
            raise vf.InvalidModelResponseError(
                f"Next actor '{next_actor_id}' is invalid for current actor '{actor_id}'."
            )
        state["trajectory_id"] = next_actor_id
        self.logger.debug(
            "multiagent.turn actor=%s mode=handoff switched_to=%s",
            actor_id,
            next_actor_id,
        )

        return self._normalize_messages(env_response)

    def last_step_for_trajectory_id(
        self, state: State, trajectory_id: str
    ) -> TrajectoryStep | None:
        for step in reversed(state["trajectory"]):
            if step["trajectory_id"] == trajectory_id:
                return step
        return None

    def messages_for_trajectory_id(
        self, state: State, trajectory_id: str
    ) -> Messages | None:
        step = self.last_step_for_trajectory_id(state, trajectory_id)
        if step is None:
            return None
        step_prompt = self._normalize_messages(step["prompt"])
        step_completion = self._normalize_messages(step["completion"])
        return concat_messages([step_prompt, step_completion])

    async def get_prompt_messages(self, state: State) -> Messages:
        if len(state["trajectory"]) == 0:
            self.logger.debug(
                "multiagent.turn initial actor=%s",
                state["trajectory_id"],
            )
            return self._normalize_messages(self.get_prompt_for_actor([], state))

        self.logger.debug(
            "multiagent.turn resume trajectory_len=%s last_actor=%s",
            len(state["trajectory"]),
            state["trajectory"][-1]["trajectory_id"],
        )
        prev_turn_prompt = self._normalize_messages(state["trajectory"][-1]["prompt"])
        prev_turn_completion = self._normalize_messages(
            state["trajectory"][-1]["completion"]
        )
        prev_messages = concat_messages([prev_turn_prompt, prev_turn_completion])
        env_response = self._normalize_messages(
            await self.env_response(prev_messages, state)
        )
        prev_trajectory_id = state["trajectory"][-1]["trajectory_id"]
        state["trajectory"][-1]["extras"]["env_response"] = env_response
        self.logger.debug(
            "multiagent.env_response stored actor=%s message_count=%s",
            prev_trajectory_id,
            len(env_response),
        )
        actor_id = state["trajectory_id"]
        continuing_same_actor = actor_id == prev_trajectory_id
        messages = self.messages_for_trajectory_id(state, actor_id)
        if messages is None:
            self.logger.debug(
                "multiagent.prompt actor=%s history=none",
                actor_id,
            )
            return self._normalize_messages(self.get_prompt_for_actor([], state))
        actor_messages = messages
        prompt_messages = self._normalize_messages(
            self.get_prompt_for_actor(actor_messages, state)
        )
        if continuing_same_actor:
            self.logger.debug(
                "multiagent.prompt actor=%s history=present env_response_count=%s mode=continue_same_actor",
                actor_id,
                len(env_response),
            )
            return concat_messages([actor_messages, env_response, prompt_messages])

        self.logger.debug(
            "multiagent.prompt actor=%s history=present mode=switched_actor",
            actor_id,
        )
        return concat_messages([actor_messages, prompt_messages])

    async def render_completion(self, state: State):
        """Render latest prompt/completion pair per actor from newest to oldest."""
        if len(state["trajectory"]) == 0:
            state["completion"] = []
            return

        unique_steps: list[TrajectoryStep] = []
        seen_trajectory_ids: set[str] = set()
        for step in reversed(state["trajectory"]):
            trajectory_id = step["trajectory_id"]
            if trajectory_id in seen_trajectory_ids:
                continue
            seen_trajectory_ids.add(trajectory_id)
            unique_steps.append(step)

        completion: Messages = []
        for step in unique_steps:
            step_prompt = self._normalize_messages(step["prompt"])
            step_completion = self._normalize_messages(step["completion"])
            completion = concat_messages([completion, step_prompt, step_completion])

        if state.get("final_env_response") is not None:
            final_resp = self._normalize_messages(state["final_env_response"])
            completion = concat_messages([completion, final_resp])

        state["completion"] = completion


class PokerMultiAgentEnv(MultiAgentEnv):
    SUITS = "cdhs"
    RANKS = "23456789TJQKA"
    DEBUG_LOG_DIRNAME = "outputs/debug"
    EQUITY_EXACT_COMBO_LIMIT = 1500
    EQUITY_MONTE_CARLO_SAMPLES = 600
    SYSTEM_PROMPT = (
        "You are a poker player in a no-limit Texas Hold'em match with multiple hands. "
        "You may use tools during your turn if available. "
        "End your turn with a handoff action of fold, check, call, or raise. "
        "For raise, provide amount as raise_to total chips committed by you this street."
    )

    def __init__(
        self,
        num_players: int = 4,
        starting_stack: int = 100,
        small_blind: int = 1,
        big_blind: int = 2,
        max_raises_per_street: int = 2,
        hands_per_rollout: int = 1,
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
        if hands_per_rollout <= 0:
            raise ValueError("hands_per_rollout must be positive")

        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_raises_per_street = max_raises_per_street
        self.hands_per_rollout = hands_per_rollout

        super().__init__(tools=[], max_turns=max_turns, **kwargs)

    def _append_hand_log(self, state: State, line: str) -> None:
        if "hand_log_lines" not in state:
            state["hand_log_lines"] = []
        state["hand_log_lines"].append(line)
        self._write_debug_line(state, f"HAND_LOG {line}")

    def _record_action(self, state: State, line: str) -> None:
        state["action_log"].append(line)
        state["hand_action_log"].append(line)
        self._write_debug_line(state, f"ACTION {line}")

    def _record_match_event(self, state: State, line: str) -> None:
        state["match_event_log"].append(line)
        self._write_debug_line(state, f"MATCH_EVENT {line}")

    def _append_turn_event(self, state: State, line: str) -> None:
        state["turn_events"].append(line)
        self._write_debug_line(state, f"TURN_EVENT {line}")

    def _consume_turn_events(self, state: State) -> list[str]:
        events = state.get("turn_events", [])
        state["turn_events"] = []
        return events if isinstance(events, list) else []

    def _write_debug_line(self, state: State, line: str) -> None:
        debug_log_path = state.get("debug_log_path")
        if not debug_log_path:
            return
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        try:
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(f"{ts} | {line}\n")
        except OSError as exc:
            self.logger.warning(
                "poker.debug_log_write_failed path=%s error=%s",
                debug_log_path,
                exc,
            )

    def _write_debug_block(self, state: State, header: str, body: str) -> None:
        self._write_debug_line(state, header)
        for line in body.splitlines():
            self._write_debug_line(state, f"  {line}")

    def _init_debug_log(self, state: State) -> None:
        debug_dir = Path(__file__).resolve().parent / self.DEBUG_LOG_DIRNAME
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        seed = state.get("seed")
        example_id = state.get("example_id")
        pid = os.getpid()
        filename = f"poker_debug_ex{example_id}_seed{seed}_{ts}_pid{pid}.log"
        debug_path = (debug_dir / filename).resolve()
        debug_path.write_text("", encoding="utf-8")
        state["debug_log_path"] = str(debug_path)
        self._write_debug_line(
            state,
            f"debug_log_initialized path={state['debug_log_path']} seed={seed} example_id={example_id}",
        )

    def _log_snapshot(self, state: State, label: str) -> None:
        board = (
            " ".join(state["community_cards"]) if state["community_cards"] else "(none)"
        )
        self._append_hand_log(
            state,
            f"[{label}] hand={state['current_hand_number']}/{state['num_hands_target']} street={state['street']} pot={state['pot']} current_bet={state['current_bet']} pending={state['pending_to_act']}",
        )
        self._append_hand_log(state, f"board={board}")
        for actor_id in state["seats"]:
            player = state["players"][actor_id]
            hole_cards = " ".join(player["hole_cards"])
            self._append_hand_log(
                state,
                f"{actor_id}: stack={player['stack']} folded={player['folded']} street_contrib={state['street_contrib'][actor_id]} hand_contrib={state['hand_contrib'][actor_id]} hole_cards={hole_cards}",
            )
        self._append_hand_log(state, f"action_log_tail={state['action_log'][-10:]}")

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
        turn_events = self._consume_turn_events(state)
        if not turn_events:
            return message
        return f"{message}\n\n" + "\n".join(turn_events)

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

    def _players_with_chips(self, state: State) -> list[str]:
        return [
            actor_id
            for actor_id in state["seats"]
            if state["players"][actor_id]["stack"] > 0
        ]

    def _active_players(self, state: State) -> list[str]:
        return [
            actor_id
            for actor_id in state["seats"]
            if not state["players"][actor_id]["folded"]
        ]

    def _next_seat_with_chips(self, state: State, seat: int) -> int | None:
        for offset in range(1, self.num_players + 1):
            candidate = (seat + offset) % self.num_players
            actor_id = self._actor_id(candidate)
            if state["players"][actor_id]["stack"] > 0:
                return candidate
        return None

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

    def _deal_hole_cards(self, state: State) -> None:
        start_seat = (state["dealer_seat"] + 1) % self.num_players
        for _ in range(2):
            for seat in self._seat_cycle(start_seat):
                actor_id = self._actor_id(seat)
                if state["players"][actor_id]["stack"] <= 0:
                    continue
                state["players"][actor_id]["hole_cards"].append(state["deck"].pop())

    def _commit_chips(self, state: State, actor_id: str, chips: int) -> None:
        if chips <= 0:
            return
        state["players"][actor_id]["stack"] -= chips
        state["street_contrib"][actor_id] += chips
        state["hand_contrib"][actor_id] += chips
        state["pot"] += chips

    def _post_forced_bet(
        self, state: State, actor_id: str, amount: int, label: str
    ) -> int:
        posted = min(state["players"][actor_id]["stack"], amount)
        self._commit_chips(state, actor_id, posted)
        self._record_action(state, f"{actor_id} posts {label}: {posted}")
        self._append_hand_log(
            state,
            f"forced_bet actor={actor_id} label={label} requested={amount} posted={posted}",
        )
        return posted

    def _first_preflop_seat(self) -> int:
        if self.num_players == 2:
            return 0
        return 3 % self.num_players

    def get_all_actors(self, state: State) -> dict[str, str]:
        return {self._actor_id(i): self.SYSTEM_PROMPT for i in range(self.num_players)}

    def get_initial_actor_id(self, actors: dict[str, str], state: State) -> str:
        return self._actor_id(self._first_preflop_seat())

    def get_next_actor_id(self, state: State) -> str:
        pending = state.get("pending_to_act", [])
        if pending:
            return pending[0]
        return state["trajectory_id"]

    def get_handoff_schema(self, actor_id: str, state: State) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["fold", "check", "call", "raise"],
                },
                "amount": {"type": ["integer", "null"]},
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)

        input_seed = state.get("input", {}).get("seed", state.get("example_id", 0))
        seed = int(input_seed)
        input_hands = state.get("input", {}).get("num_hands", self.hands_per_rollout)
        num_hands_target = int(input_hands)
        if num_hands_target <= 0:
            raise ValueError("num_hands must be positive")

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

        state["seed"] = seed
        state["seats"] = seats
        state["players"] = players
        state["deck"] = []
        state["community_cards"] = []
        state["dealer_seat"] = 0
        state["small_blind_seat"] = 0
        state["big_blind_seat"] = 0
        state["pot"] = 0
        state["street"] = "preflop"
        state["street_contrib"] = {actor_id: 0 for actor_id in seats}
        state["hand_contrib"] = {actor_id: 0 for actor_id in seats}
        state["current_bet"] = 0
        state["last_raise_size"] = self.big_blind
        state["street_raise_count"] = 0
        state["pending_to_act"] = []
        state["action_log"] = []
        state["hand_action_log"] = []
        state["hand_log_lines"] = []
        state["turn_events"] = []
        state["match_event_log"] = []
        state["deck_initial_order"] = []

        state["num_hands_target"] = num_hands_target
        state["current_hand_number"] = 0
        state["hands_completed"] = 0
        state["hand_summaries"] = []
        state["last_hand_summary"] = None
        state["last_hand_result"] = {}

        state["max_possible_payout"] = self.num_players * self.starting_stack
        state["winner_winnings"] = 0
        state["winner_winnings_by_player"] = {}
        state["player_streets_seen"] = {actor_id: 0 for actor_id in seats}
        state["showdown_hands"] = 0
        state["fold_win_hands"] = 0
        state["total_pot_distributed"] = 0
        state["player_action_count"] = 0
        state["illegal_action_count"] = 0
        state["voluntary_fold_count"] = 0
        state["forced_fold_count"] = 0
        state["decision_records"] = []

        state["match_payouts_by_player"] = {actor_id: 0 for actor_id in seats}
        state["rollout_completed_cleanly"] = True
        state["malformed_handoff"] = None
        self._init_debug_log(state)
        started = self._start_next_hand(state, initial=True, emit_turn_event=False)
        if not started:
            self._finalize_match(state, reason="insufficient active players")
            return state

        self._progress_game(state)
        self.logger.debug(
            "poker.setup seed=%s players=%s stack=%s hands=%s first_actor=%s",
            seed,
            self.num_players,
            self.starting_stack,
            num_hands_target,
            state["trajectory_id"],
        )
        self._append_hand_log(
            state,
            f"setup seed={seed} num_players={self.num_players} starting_stack={self.starting_stack} small_blind={self.small_blind} big_blind={self.big_blind} hands={num_hands_target}",
        )
        self._log_snapshot(state, "setup_complete")
        return state

    def _start_next_hand(
        self, state: State, initial: bool, emit_turn_event: bool
    ) -> bool:
        alive = self._players_with_chips(state)
        if len(alive) < 2:
            return False

        if initial:
            current_dealer = state["dealer_seat"]
            current_actor = self._actor_id(current_dealer)
            if state["players"][current_actor]["stack"] <= 0:
                next_dealer = self._next_seat_with_chips(state, current_dealer)
                if next_dealer is None:
                    return False
                state["dealer_seat"] = next_dealer
        else:
            next_dealer = self._next_seat_with_chips(state, state["dealer_seat"])
            if next_dealer is None:
                return False
            state["dealer_seat"] = next_dealer

        hand_number = state["hands_completed"] + 1
        state["current_hand_number"] = hand_number

        for actor_id in state["seats"]:
            player = state["players"][actor_id]
            player["folded"] = player["stack"] <= 0
            player["hole_cards"] = []

        hand_seed = state["seed"] * 1_000_003 + hand_number
        rng = random.Random(hand_seed)
        deck = self._build_deck()
        rng.shuffle(deck)
        state["deck"] = deck
        state["deck_initial_order"] = list(deck)
        state["community_cards"] = []
        state["pot"] = 0
        state["street"] = "preflop"
        state["street_contrib"] = {actor_id: 0 for actor_id in state["seats"]}
        state["hand_contrib"] = {actor_id: 0 for actor_id in state["seats"]}
        state["current_bet"] = 0
        state["last_raise_size"] = self.big_blind
        state["street_raise_count"] = 0
        state["pending_to_act"] = []
        state["hand_action_log"] = []
        state["turn_events"] = []

        self._deal_hole_cards(state)

        if len(alive) == 2:
            small_blind_seat = state["dealer_seat"]
            big_blind_seat = self._next_seat_with_chips(state, small_blind_seat)
            first_preflop_seat = small_blind_seat
        else:
            small_blind_seat = self._next_seat_with_chips(state, state["dealer_seat"])
            if small_blind_seat is None:
                return False
            big_blind_seat = self._next_seat_with_chips(state, small_blind_seat)
            if big_blind_seat is None:
                return False
            first_preflop_seat = self._next_seat_with_chips(state, big_blind_seat)

        if big_blind_seat is None or first_preflop_seat is None:
            return False

        state["small_blind_seat"] = small_blind_seat
        state["big_blind_seat"] = big_blind_seat

        small_blind_actor = self._actor_id(small_blind_seat)
        big_blind_actor = self._actor_id(big_blind_seat)
        sb_posted = self._post_forced_bet(
            state, small_blind_actor, self.small_blind, "small blind"
        )
        bb_posted = self._post_forced_bet(
            state, big_blind_actor, self.big_blind, "big blind"
        )
        state["current_bet"] = max(state["street_contrib"].values())
        state["pending_to_act"] = self._actionable_players_from(
            state, first_preflop_seat
        )
        if state["pending_to_act"]:
            state["trajectory_id"] = state["pending_to_act"][0]

        for actor_id in state["seats"]:
            if not state["players"][actor_id]["folded"]:
                state["player_streets_seen"][actor_id] += 1

        dealer_actor = self._actor_id(state["dealer_seat"])
        start_line = (
            f"Hand {hand_number}/{state['num_hands_target']} starts. "
            f"Dealer: {dealer_actor}. Blinds: {small_blind_actor}({sb_posted})/{big_blind_actor}({bb_posted})."
        )
        self._record_match_event(state, start_line)
        if emit_turn_event:
            self._append_turn_event(state, start_line)

        self.logger.debug(
            "poker.hand_start hand=%s dealer=%s sb=%s bb=%s first_actor=%s",
            hand_number,
            dealer_actor,
            small_blind_actor,
            big_blind_actor,
            state["pending_to_act"][0] if state["pending_to_act"] else None,
        )
        self._append_hand_log(state, f"hand_start seed={hand_seed} {start_line}")
        self._log_snapshot(state, f"hand_{hand_number}_start")
        return True

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
        can_call = to_call > 0 and stack > 0

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
            if p["stack"] <= 0:
                status = "out"
            elif p["folded"]:
                status = "folded"
            else:
                status = "active"
            player_lines.append(
                f"- {pid}: stack={p['stack']}, status={status}, street_commit={state['street_contrib'][pid]}, hand_commit={state['hand_contrib'][pid]}"
            )

        board = (
            " ".join(state["community_cards"]) if state["community_cards"] else "(none)"
        )
        hole_cards = " ".join(player["hole_cards"])
        action_log = state["hand_action_log"][-12:]
        log_text = (
            "\n".join(f"- {entry}" for entry in action_log)
            if action_log
            else "- (none)"
        )
        match_events = state["match_event_log"][-5:]
        match_events_text = (
            "\n".join(f"- {entry}" for entry in match_events)
            if match_events
            else "- (none)"
        )

        if options["can_raise"]:
            raise_instruction = (
                f"raise is legal with amount in [{options['min_raise_to']}, {options['max_raise_to']}], "
                "where amount means raise_to total chips committed by you this street."
            )
        else:
            raise_instruction = "raise is not legal right now."

        if options["can_call"] and player["stack"] < options["to_call"]:
            call_instruction = f"You may call all-in for {player['stack']} (short of full call {options['to_call']})."
        else:
            call_instruction = ""

        last_hand_summary = state.get("last_hand_summary") or "(none yet)"
        return (
            f"Match hand: {state['current_hand_number']} of {state['num_hands_target']}\n"
            f"Hands completed: {state['hands_completed']}\n"
            f"Last hand summary: {last_hand_summary}\n"
            f"Street: {state['street']}\n"
            f"Pot: {state['pot']}\n"
            f"Board: {board}\n"
            f"You are: {actor_id}\n"
            f"Your hole cards: {hole_cards}\n"
            f"Your stack: {player['stack']}\n"
            f"Your street contribution: {state['street_contrib'][actor_id]}\n"
            f"Your hand contribution: {state['hand_contrib'][actor_id]}\n"
            f"To call: {options['to_call']}\n"
            f"can_check={options['can_check']} can_call={options['can_call']} can_raise={options['can_raise']}\n"
            f"{raise_instruction}\n"
            f"{call_instruction}\n"
            f"Legal actions: {', '.join(options['legal_actions'])}\n"
            "\nPlayers:\n"
            f"{chr(10).join(player_lines)}\n"
            "\nRecent action log (this hand):\n"
            f"{log_text}\n"
            "\nRecent match events:\n"
            f"{match_events_text}\n"
        )

    def get_prompt_for_actor(self, messages: Messages, state: State) -> Messages:
        actor_id = state["trajectory_id"]
        system_prompt = state["system_prompts"][actor_id]
        prompt_text = self._render_game_state_prompt(actor_id, state)
        self._write_debug_block(
            state,
            f"PROMPT actor={actor_id} hand={state.get('current_hand_number')} street={state.get('street')} pending={state.get('pending_to_act')}",
            prompt_text,
        )
        game_state_message = {"role": "user", "content": prompt_text}
        if len(messages) == 0:
            return [{"role": "system", "content": system_prompt}, game_state_message]
        return [game_state_message]

    def _remove_from_pending(self, state: State, actor_id: str) -> None:
        state["pending_to_act"] = [
            pid for pid in state["pending_to_act"] if pid != actor_id
        ]

    def _record_player_action_outcome(
        self,
        state: State,
        actor_id: str,
        declared_action: str,
        *,
        illegal: bool,
        forced_fold: bool,
    ) -> None:
        state["player_action_count"] = int(state.get("player_action_count", 0)) + 1
        if illegal:
            state["illegal_action_count"] = (
                int(state.get("illegal_action_count", 0)) + 1
            )
        if forced_fold:
            state["forced_fold_count"] = int(state.get("forced_fold_count", 0)) + 1
        elif declared_action == "fold":
            state["voluntary_fold_count"] = (
                int(state.get("voluntary_fold_count", 0)) + 1
            )
        self._append_hand_log(
            state,
            (
                f"action_outcome actor={actor_id} declared={declared_action} "
                f"illegal={illegal} forced_fold={forced_fold} "
                f"totals(actions={state['player_action_count']}, "
                f"illegal={state['illegal_action_count']}, "
                f"voluntary_folds={state['voluntary_fold_count']}, "
                f"forced_folds={state['forced_fold_count']})"
            ),
        )

    def _winner_share_for_board(
        self, state: State, contenders: list[str], full_board: list[str], actor_id: str
    ) -> float:
        if actor_id not in contenders:
            return 0.0
        if len(contenders) == 1:
            return 1.0
        scores = {
            pid: self._evaluate_seven(state["players"][pid]["hole_cards"] + full_board)
            for pid in contenders
        }
        best_score = max(scores.values())
        winners = [pid for pid, score in scores.items() if score == best_score]
        if actor_id not in winners:
            return 0.0
        return 1.0 / len(winners)

    def _estimate_showdown_equity(self, state: State, actor_id: str) -> float:
        player = state["players"][actor_id]
        if player["folded"] or len(player["hole_cards"]) != 2:
            return 0.0

        contenders = [
            pid
            for pid in state["seats"]
            if (not state["players"][pid]["folded"])
            and len(state["players"][pid]["hole_cards"]) == 2
        ]
        if actor_id not in contenders:
            return 0.0
        if len(contenders) == 1:
            return 1.0

        board = list(state["community_cards"])
        missing_board = max(0, 5 - len(board))
        if missing_board == 0:
            return self._winner_share_for_board(state, contenders, board, actor_id)

        used_cards = set(board)
        for pid in state["seats"]:
            used_cards.update(state["players"][pid]["hole_cards"])
        remaining_cards = [
            card for card in self._build_deck() if card not in used_cards
        ]
        if len(remaining_cards) < missing_board:
            return 0.0

        combo_count = math.comb(len(remaining_cards), missing_board)
        if combo_count <= self.EQUITY_EXACT_COMBO_LIMIT:
            runouts = itertools.combinations(remaining_cards, missing_board)
            total_runouts = combo_count
        else:
            total_runouts = self.EQUITY_MONTE_CARLO_SAMPLES
            seed_text = (
                f"{state.get('seed', 0)}:"
                f"{state.get('current_hand_number', 0)}:"
                f"{state.get('street', 'preflop')}:"
                f"{actor_id}:"
                f"{len(state.get('decision_records', []))}"
            )
            rng = random.Random(seed_text)
            runouts = [
                tuple(rng.sample(remaining_cards, missing_board))
                for _ in range(total_runouts)
            ]

        if total_runouts <= 0:
            return 0.0
        total_share = 0.0
        for runout in runouts:
            full_board = board + list(runout)
            total_share += self._winner_share_for_board(
                state, contenders, full_board, actor_id
            )
        return total_share / total_runouts

    def _record_decision_snapshot(
        self,
        state: State,
        actor_id: str,
        action: str,
        *,
        to_call: int,
        pot_before: int,
        equity: float,
        illegal: bool,
    ) -> None:
        record = {
            "hand_number": int(state.get("current_hand_number", 0)),
            "street": state.get("street", "preflop"),
            "actor_id": actor_id,
            "action": action,
            "to_call": int(to_call),
            "pot_before": int(pot_before),
            "equity": float(max(0.0, min(1.0, equity))),
            "illegal": bool(illegal),
        }
        if "decision_records" not in state:
            state["decision_records"] = []
        state["decision_records"].append(record)
        self._append_hand_log(
            state,
            (
                f"decision_snapshot actor={actor_id} action={action} "
                f"to_call={record['to_call']} pot_before={record['pot_before']} "
                f"equity={record['equity']:.4f} illegal={record['illegal']}"
            ),
        )

    def _fold_player(self, state: State, actor_id: str, reason: str) -> None:
        if not state["players"][actor_id]["folded"]:
            state["players"][actor_id]["folded"] = True
        self._remove_from_pending(state, actor_id)
        self._record_action(state, f"{actor_id} folds ({reason})")

    def _start_new_street(self, state: State, street: str) -> None:
        state["street"] = street
        state["street_contrib"] = {pid: 0 for pid in state["seats"]}
        state["current_bet"] = 0
        state["last_raise_size"] = self.big_blind
        state["street_raise_count"] = 0
        for actor_id in state["seats"]:
            if not state["players"][actor_id]["folded"]:
                state["player_streets_seen"][actor_id] += 1
        first_to_act = self._next_seat_with_chips(state, state["dealer_seat"])
        if first_to_act is None:
            state["pending_to_act"] = []
            return
        state["pending_to_act"] = self._actionable_players_from(state, first_to_act)

    def _advance_street(self, state: State) -> None:
        street = state["street"]
        if street == "preflop":
            new_cards = self._deal_cards(state, 3)
            state["community_cards"].extend(new_cards)
            self._record_action(state, f"Dealer deals flop: {' '.join(new_cards)}")
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
            self._record_action(state, f"Dealer deals turn: {new_card[0]}")
            self._append_hand_log(state, f"street_transition turn card={new_card[0]}")
            self._start_new_street(state, "turn")
            self.logger.debug(
                "poker.street advanced=turn board=%s", state["community_cards"]
            )
            return
        if street == "turn":
            new_card = self._deal_cards(state, 1)
            state["community_cards"].extend(new_card)
            self._record_action(state, f"Dealer deals river: {new_card[0]}")
            self._append_hand_log(state, f"street_transition river card={new_card[0]}")
            self._start_new_street(state, "river")
            self.logger.debug(
                "poker.street advanced=river board=%s", state["community_cards"]
            )
            return
        if street == "river":
            self._run_showdown(state)
            return

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

    def _split_amount_by_seat(
        self, state: State, winners: list[str], amount: int
    ) -> dict[str, int]:
        if not winners or amount <= 0:
            return {}
        share = amount // len(winners)
        remainder = amount % len(winners)
        payouts = {actor_id: share for actor_id in winners}
        if remainder > 0:
            start_seat = (state["dealer_seat"] + 1) % self.num_players
            for seat in self._seat_cycle(start_seat):
                actor_id = self._actor_id(seat)
                if actor_id in payouts:
                    payouts[actor_id] += 1
                    remainder -= 1
                    if remainder == 0:
                        break
        return payouts

    def _resolve_showdown_payouts(
        self, state: State, scores: dict[str, tuple[int, tuple[int, ...]]]
    ) -> dict[str, int]:
        contributions = state["hand_contrib"]
        levels = sorted({chips for chips in contributions.values() if chips > 0})
        payouts = {actor_id: 0 for actor_id in state["seats"]}
        previous_level = 0

        for level in levels:
            layer_participants = [
                actor_id
                for actor_id in state["seats"]
                if contributions[actor_id] >= level
            ]
            layer_amount = (level - previous_level) * len(layer_participants)
            previous_level = level
            if layer_amount <= 0:
                continue

            eligible = [
                actor_id
                for actor_id in layer_participants
                if not state["players"][actor_id]["folded"]
            ]
            if not eligible:
                continue
            best_score = max(scores[actor_id] for actor_id in eligible)
            winners = [
                actor_id for actor_id in eligible if scores[actor_id] == best_score
            ]
            layer_payout = self._split_amount_by_seat(state, winners, layer_amount)
            for actor_id, chips in layer_payout.items():
                payouts[actor_id] += chips
            self._append_hand_log(
                state,
                f"side_pot level={level} amount={layer_amount} eligible={eligible} winners={winners}",
            )

        return {actor_id: chips for actor_id, chips in payouts.items() if chips > 0}

    def _finalize_match(self, state: State, reason: str) -> None:
        state["pending_to_act"] = []
        state["street"] = "finished"
        stacks = {
            actor_id: state["players"][actor_id]["stack"] for actor_id in state["seats"]
        }
        top_stack = max(stacks.values()) if stacks else 0
        leaders = [actor_id for actor_id, stack in stacks.items() if stack == top_stack]
        stacks_text = ", ".join(
            f"{actor_id}={stacks[actor_id]}" for actor_id in state["seats"]
        )
        leaders_text = ", ".join(leaders) if leaders else "(none)"
        base = (
            f"Match finished after {state['hands_completed']} hand(s) ({reason}). "
            f"Winner(s): {leaders_text}. Final stacks: {stacks_text}."
        )
        if state.get("last_hand_summary"):
            base = f"{base} Last hand: {state['last_hand_summary']}"
        if state.get("debug_log_path"):
            base = f"{base} Debug log: {state['debug_log_path']}"
        state["final_env_response"] = base
        self._append_turn_event(state, base)
        self._record_match_event(state, base)
        self.logger.info(
            "poker.match_finished reason=%s hands=%s leaders=%s",
            reason,
            state["hands_completed"],
            leaders_text,
        )

    def _complete_hand(
        self,
        state: State,
        reason: str,
        payouts: dict[str, int],
        winners: list[str],
        category_name: str | None = None,
    ) -> None:
        total_paid = sum(payouts.values())
        if total_paid != state["pot"]:
            raise ValueError(
                f"Internal payout mismatch: pot={state['pot']} paid={total_paid}"
            )

        for actor_id, chips in payouts.items():
            state["players"][actor_id]["stack"] += chips
            state["match_payouts_by_player"][actor_id] += chips

        state["winner_winnings"] = max(payouts.values()) if payouts else 0
        state["winner_winnings_by_player"] = payouts
        state["total_pot_distributed"] += total_paid
        state["pot"] = 0
        state["street"] = "finished"
        state["pending_to_act"] = []
        state["hands_completed"] += 1

        winners_text = ", ".join(winners)
        board_text = (
            " ".join(state["community_cards"]) if state["community_cards"] else "(none)"
        )
        if reason == "folds":
            hand_summary = (
                f"Hand {state['hands_completed']} finished by folds. "
                f"Winner: {winners_text}. Winnings: {state['winner_winnings']}. "
                f"Board: {board_text}."
            )
            state["fold_win_hands"] += 1
            self._record_action(state, f"Hand ends: {winners_text} wins by folds")
        else:
            category = category_name or "unknown"
            hand_summary = (
                f"Hand {state['hands_completed']} finished at showdown. "
                f"Winners: {winners_text}. Best hand category: {category}. "
                f"Payouts: {payouts}. Board: {board_text}."
            )
            state["showdown_hands"] += 1
            self._record_action(state, f"Showdown winners: {winners_text} ({category})")

        state["last_hand_summary"] = hand_summary
        state["last_hand_result"] = {
            "reason": reason,
            "winners": winners,
            "payouts": payouts,
            "category_name": category_name,
            "hand_number": state["hands_completed"],
        }
        state["hand_summaries"].append(hand_summary)
        self._record_match_event(state, hand_summary)
        self._append_turn_event(state, hand_summary)

        self._append_hand_log(state, f"hand_end reason={reason} payouts={payouts}")
        self._log_snapshot(state, "hand_finished")
        self.logger.info(
            "poker.hand_finished hand=%s reason=%s winners=%s payouts=%s",
            state["hands_completed"],
            reason,
            winners_text,
            payouts,
        )

        alive_after_hand = self._players_with_chips(state)
        if len(alive_after_hand) <= 1:
            self._finalize_match(state, reason="single player remaining")
            return
        if state["hands_completed"] >= state["num_hands_target"]:
            self._finalize_match(state, reason="hand limit reached")
            return

        started = self._start_next_hand(state, initial=False, emit_turn_event=True)
        if not started:
            self._finalize_match(state, reason="insufficient active players")

    def _finalize_single_winner(self, state: State, winner: str) -> None:
        payouts = {winner: state["pot"]}
        self._complete_hand(state, reason="folds", payouts=payouts, winners=[winner])

    def _run_showdown(self, state: State) -> None:
        if len(state["community_cards"]) < 5:
            missing = 5 - len(state["community_cards"])
            new_cards = self._deal_cards(state, missing)
            state["community_cards"].extend(new_cards)
            self._record_action(
                state, f"Dealer deals remaining board: {' '.join(new_cards)}"
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

        payouts = self._resolve_showdown_payouts(state, scores)
        if not payouts:
            raise ValueError("Showdown produced no payouts")
        winners = sorted(
            payouts,
            key=lambda actor_id: (-payouts[actor_id], self._seat(actor_id)),
        )
        best_score = max(scores.values())
        category_name = self._hand_rank_name(best_score[0])
        self._complete_hand(
            state,
            reason="showdown",
            payouts=payouts,
            winners=winners,
            category_name=category_name,
        )

    def _progress_game(self, state: State) -> None:
        while state.get("final_env_response") is None:
            active = self._active_players(state)
            if len(active) == 1 and state["street"] in {
                "preflop",
                "flop",
                "turn",
                "river",
            }:
                self._finalize_single_winner(state, active[0])
                continue

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

    async def apply_handoff(self, actor_id: str, handoff: dict, state: State):
        self._write_debug_line(state, f"HANDOFF actor={actor_id} payload={handoff}")
        action = str(handoff["action"]).strip().lower()
        amount = self._coerce_optional_int(handoff.get("amount"))
        return await self.take_action(action=action, amount=amount, state=state)

    async def take_action(
        self,
        action: str,
        amount: int | None = None,
        state: State | None = None,
    ) -> str:
        if state is None:
            return "Error: missing state"
        if state.get("final_env_response") is not None:
            return "Match is already finished."

        actor_id = state["trajectory_id"]
        action = action.strip().lower()
        raw_amount = amount
        amount = self._coerce_optional_int(raw_amount)
        player = state["players"][actor_id]
        actor_contrib = state["street_contrib"][actor_id]
        to_call = max(0, state["current_bet"] - actor_contrib)
        pot_before = state["pot"]
        equity_cache: float | None = None

        def equity_snapshot() -> float:
            nonlocal equity_cache
            if equity_cache is None:
                equity_cache = self._estimate_showdown_equity(state, actor_id)
            return equity_cache

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
            self._record_player_action_outcome(
                state,
                actor_id,
                action,
                illegal=True,
                forced_fold=True,
            )
            self._record_decision_snapshot(
                state,
                actor_id,
                action,
                to_call=to_call,
                pot_before=pot_before,
                equity=0.0,
                illegal=True,
            )
            self._fold_player(state, actor_id, "acted out of turn")
            self.logger.warning(
                "poker.action illegal_fold actor=%s reason=out_of_turn", actor_id
            )
            self._progress_game(state)
            return self._action_result(
                state, actor_id, action, "Illegal action (out of turn). Player folded."
            )

        if action == "fold":
            self._record_player_action_outcome(
                state,
                actor_id,
                action,
                illegal=False,
                forced_fold=False,
            )
            self._record_decision_snapshot(
                state,
                actor_id,
                action,
                to_call=to_call,
                pot_before=pot_before,
                equity=equity_snapshot(),
                illegal=False,
            )
            self._fold_player(state, actor_id, "chose fold")
            self._progress_game(state)
            return self._action_result(state, actor_id, action, "Player folded.")

        if action == "check":
            if to_call != 0:
                self._record_player_action_outcome(
                    state,
                    actor_id,
                    action,
                    illegal=True,
                    forced_fold=True,
                )
                self._record_decision_snapshot(
                    state,
                    actor_id,
                    action,
                    to_call=to_call,
                    pot_before=pot_before,
                    equity=equity_snapshot(),
                    illegal=True,
                )
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
            self._record_player_action_outcome(
                state,
                actor_id,
                action,
                illegal=False,
                forced_fold=False,
            )
            self._record_decision_snapshot(
                state,
                actor_id,
                action,
                to_call=to_call,
                pot_before=pot_before,
                equity=equity_snapshot(),
                illegal=False,
            )
            self._remove_from_pending(state, actor_id)
            self._record_action(state, f"{actor_id} checks")
            self._progress_game(state)
            return self._action_result(state, actor_id, action, "Check accepted.")

        if action == "call":
            if to_call <= 0:
                self._record_player_action_outcome(
                    state,
                    actor_id,
                    action,
                    illegal=True,
                    forced_fold=True,
                )
                self._record_decision_snapshot(
                    state,
                    actor_id,
                    action,
                    to_call=to_call,
                    pot_before=pot_before,
                    equity=equity_snapshot(),
                    illegal=True,
                )
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
            if player["stack"] <= 0:
                self._record_player_action_outcome(
                    state,
                    actor_id,
                    action,
                    illegal=True,
                    forced_fold=True,
                )
                self._record_decision_snapshot(
                    state,
                    actor_id,
                    action,
                    to_call=to_call,
                    pot_before=pot_before,
                    equity=equity_snapshot(),
                    illegal=True,
                )
                self._fold_player(state, actor_id, "illegal call with zero stack")
                self.logger.warning(
                    "poker.action illegal_fold actor=%s reason=call_with_zero_stack",
                    actor_id,
                )
                self._progress_game(state)
                return self._action_result(
                    state,
                    actor_id,
                    action,
                    "Illegal action (no chips to call). Player folded.",
                )

            self._record_player_action_outcome(
                state,
                actor_id,
                action,
                illegal=False,
                forced_fold=False,
            )
            self._record_decision_snapshot(
                state,
                actor_id,
                action,
                to_call=to_call,
                pot_before=pot_before,
                equity=equity_snapshot(),
                illegal=False,
            )
            call_amount = min(player["stack"], to_call)
            self._commit_chips(state, actor_id, call_amount)
            self._remove_from_pending(state, actor_id)
            if call_amount < to_call:
                self._record_action(
                    state,
                    f"{actor_id} calls all-in {call_amount} (short of {to_call})",
                )
                result = f"Call accepted as all-in for {call_amount} (to_call was {to_call})."
            else:
                self._record_action(state, f"{actor_id} calls {to_call}")
                result = f"Call accepted for {to_call}."
            self._progress_game(state)
            return self._action_result(state, actor_id, action, result)

        if action == "raise":
            if state["street_raise_count"] >= self.max_raises_per_street:
                self._record_player_action_outcome(
                    state,
                    actor_id,
                    action,
                    illegal=True,
                    forced_fold=True,
                )
                self._record_decision_snapshot(
                    state,
                    actor_id,
                    action,
                    to_call=to_call,
                    pot_before=pot_before,
                    equity=equity_snapshot(),
                    illegal=True,
                )
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
                self._record_player_action_outcome(
                    state,
                    actor_id,
                    action,
                    illegal=True,
                    forced_fold=True,
                )
                self._record_decision_snapshot(
                    state,
                    actor_id,
                    action,
                    to_call=to_call,
                    pot_before=pot_before,
                    equity=equity_snapshot(),
                    illegal=True,
                )
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
                self._record_player_action_outcome(
                    state,
                    actor_id,
                    action,
                    illegal=True,
                    forced_fold=True,
                )
                self._record_decision_snapshot(
                    state,
                    actor_id,
                    action,
                    to_call=to_call,
                    pot_before=pot_before,
                    equity=equity_snapshot(),
                    illegal=True,
                )
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
                self._record_player_action_outcome(
                    state,
                    actor_id,
                    action,
                    illegal=True,
                    forced_fold=True,
                )
                self._record_decision_snapshot(
                    state,
                    actor_id,
                    action,
                    to_call=to_call,
                    pot_before=pot_before,
                    equity=equity_snapshot(),
                    illegal=True,
                )
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
                self._record_player_action_outcome(
                    state,
                    actor_id,
                    action,
                    illegal=True,
                    forced_fold=True,
                )
                self._record_decision_snapshot(
                    state,
                    actor_id,
                    action,
                    to_call=to_call,
                    pot_before=pot_before,
                    equity=equity_snapshot(),
                    illegal=True,
                )
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
                self._record_player_action_outcome(
                    state,
                    actor_id,
                    action,
                    illegal=True,
                    forced_fold=True,
                )
                self._record_decision_snapshot(
                    state,
                    actor_id,
                    action,
                    to_call=to_call,
                    pot_before=pot_before,
                    equity=equity_snapshot(),
                    illegal=True,
                )
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

            self._record_player_action_outcome(
                state,
                actor_id,
                action,
                illegal=False,
                forced_fold=False,
            )
            self._record_decision_snapshot(
                state,
                actor_id,
                action,
                to_call=to_call,
                pot_before=pot_before,
                equity=equity_snapshot(),
                illegal=False,
            )
            previous_bet = state["current_bet"]
            self._commit_chips(state, actor_id, delta)
            state["current_bet"] = amount
            state["last_raise_size"] = amount - previous_bet
            state["street_raise_count"] += 1
            self._record_action(state, f"{actor_id} raises to {amount}")

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

        self._record_player_action_outcome(
            state,
            actor_id,
            action,
            illegal=True,
            forced_fold=True,
        )
        self._record_decision_snapshot(
            state,
            actor_id,
            action,
            to_call=to_call,
            pot_before=pot_before,
            equity=equity_snapshot(),
            illegal=True,
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
    num_seed_rows: int = 5000,
    hands_per_rollout: int = 1,
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

    def has_no_malformed_handoff(state: State) -> bool:
        if state.get("error") is not None:
            return False
        if state.get("final_env_response") is None:
            return False
        if state.get("malformed_handoff"):
            return False
        return bool(state.get("rollout_completed_cleanly", True))

    def has_no_illegal_actions(state: State) -> bool:
        if not has_no_malformed_handoff(state):
            return False
        return int(state.get("illegal_action_count", 0)) == 0

    def task_rewards_enabled(state: State) -> bool:
        return has_no_malformed_handoff(state) and has_no_illegal_actions(state)

    async def malformed_handoff_count(state: State) -> float:
        return 1.0 if state.get("malformed_handoff") else 0.0

    async def illegal_action_count_metric(state: State) -> float:
        return float(max(int(state.get("illegal_action_count", 0)), 0))

    async def streets_seen_reward(state: State) -> float:
        """Reward for total streets seen across all players. Encourages staying in hands."""
        if not task_rewards_enabled(state):
            return 0.0
        player_streets = state.get("player_streets_seen", {})
        if not player_streets:
            return 0.0
        num_players = len(state["seats"])
        hands_target = max(int(state.get("num_hands_target", 1)), 1)
        # max 4 streets per player per hand: preflop, flop, turn, river
        total = sum(player_streets.values())
        return total / (num_players * 4 * hands_target)

    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    async def pot_odds_decision_quality_reward(state: State) -> float:
        """
        Reward action quality using equity vs pot-odds at decision time.
        Uses recorded per-action equity snapshots from game state.
        """
        if not task_rewards_enabled(state):
            return 0.0
        records = state.get("decision_records", [])
        if not isinstance(records, list) or not records:
            return 0.0

        max_pot = max(float(state.get("max_possible_payout", 1)), 1.0)
        weighted_score = 0.0
        total_weight = 0.0

        for record in records:
            if not isinstance(record, dict):
                continue
            if bool(record.get("illegal", False)):
                continue

            action = str(record.get("action", "")).strip().lower()
            equity = _clamp01(float(record.get("equity", 0.0)))
            to_call = max(float(record.get("to_call", 0.0)), 0.0)
            pot_before = max(float(record.get("pot_before", 0.0)), 0.0)

            if to_call > 0.0:
                break_even = to_call / max(pot_before + to_call, 1e-9)
                edge = equity - break_even
                if action == "fold":
                    score = _clamp01(0.5 - (2.5 * edge))
                elif action == "call":
                    score = _clamp01(0.5 + (2.5 * edge))
                elif action == "raise":
                    score = _clamp01(0.45 + (3.0 * edge))
                else:
                    continue
            else:
                if action == "check":
                    score = _clamp01(0.85 - max(0.0, equity - 0.70) * 1.5)
                elif action == "raise":
                    score = _clamp01(0.35 + (1.2 * equity))
                elif action == "fold":
                    score = 0.0
                else:
                    continue

            weight = 1.0 + min(pot_before / max_pot, 0.5)
            weighted_score += score * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return weighted_score / total_weight

    async def avg_abs_net_winnings_metric(state: State) -> float:
        """Average absolute net winnings per player, normalized by starting stack."""
        if not task_rewards_enabled(state):
            return 0.0
        seats = state.get("seats", [])
        if not seats:
            return 0.0
        num_players = len(seats)
        starting_stack = state.get("max_possible_payout", 0) / max(num_players, 1)
        if starting_stack <= 0:
            return 0.0
        deltas = [
            float(state["players"][actor_id]["stack"] - starting_stack)
            for actor_id in seats
        ]
        return sum(abs(delta) for delta in deltas) / (num_players * starting_stack)

    async def bluff_success_reward(state: State) -> float:
        """Reward for winning by folds with a weak hand. Encourages bluffing."""
        if not task_rewards_enabled(state):
            return 0.0
        last_hand_result = state.get("last_hand_result", {})
        if last_hand_result.get("reason") != "folds":
            return 0.0
        winners = last_hand_result.get("winners", [])
        if not winners:
            return 0.0
        winner = winners[0]
        hole_cards = state["players"][winner]["hole_cards"]
        if len(hole_cards) != 2:
            return 0.0
        # simple preflop hand strength: rank sum normalized, bonus for pairs/suited
        rank_order = "23456789TJQKA"
        r0 = rank_order.index(hole_cards[0][0]) + 2  # 2-14
        r1 = rank_order.index(hole_cards[1][0]) + 2
        strength = (r0 + r1) / 28.0  # AA=1.0, 2-3=0.18
        if hole_cards[0][1] == hole_cards[1][1]:  # suited
            strength += 0.05
        if r0 == r1:  # paired
            strength += 0.10
        strength = min(strength, 1.0)
        # reward is inverse: weaker hand = better bluff
        return 1.0 - strength

    async def showdown_rate(state: State) -> float:
        """Fraction of completed hands that reached showdown."""
        if not task_rewards_enabled(state):
            return 0.0
        hands_completed = max(int(state.get("hands_completed", 0)), 0)
        if hands_completed == 0:
            return 0.0
        return float(state.get("showdown_hands", 0)) / hands_completed

    async def aggression_ratio(state: State) -> float:
        """Raises / (raises + calls + checks) across all players."""
        if not task_rewards_enabled(state):
            return 0.0
        action_log = state.get("action_log", [])
        raises = 0
        calls = 0
        checks = 0
        for entry in action_log:
            if not isinstance(entry, str):
                continue
            if " raises to " in entry:
                raises += 1
            elif " calls " in entry:
                calls += 1
            elif " checks" in entry:
                checks += 1
        total = raises + calls + checks
        if total == 0:
            return 0.0
        return raises / total

    async def pot_size_metric(state: State) -> float:
        """Average pot size relative to total chips in play."""
        if not task_rewards_enabled(state):
            return 0.0
        max_possible = state.get("max_possible_payout", 0)
        hands_completed = max(int(state.get("hands_completed", 0)), 0)
        if not max_possible or hands_completed == 0:
            return 0.0
        total_paid = float(state.get("total_pot_distributed", 0))
        return total_paid / (max_possible * hands_completed)

    rubric = vf.Rubric(
        funcs=[
            streets_seen_reward,
            pot_odds_decision_quality_reward,
            malformed_handoff_count,
            illegal_action_count_metric,
            avg_abs_net_winnings_metric,
            bluff_success_reward,
            showdown_rate,
            aggression_ratio,
            pot_size_metric,
        ],
        weights=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return PokerMultiAgentEnv(
        dataset=dataset,
        rubric=rubric,
        hands_per_rollout=hands_per_rollout,
        system_prompt=None,
    )
