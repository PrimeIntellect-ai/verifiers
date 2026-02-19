from __future__ import annotations

import json
import re
from abc import abstractmethod
from copy import deepcopy
from typing import Any

import verifiers as vf
from verifiers.utils.async_utils import maybe_await
from verifiers.types import AssistantMessage, Messages, State, TrajectoryStep
from verifiers.utils.message_utils import concat_messages, normalize_messages


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
        self, actor_id: str, last_message: AssistantMessage, state: State
    ) -> dict[str, Any]:
        message_text = self._content_to_text(last_message.content)
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
        if not isinstance(messages[-1], AssistantMessage):
            raise vf.InvalidModelResponseError(
                "Expected assistant message at end of turn for multi-agent handoff."
            )

        actor_id = state["trajectory_id"]
        last_msg = messages[-1]
        tool_calls = last_msg.tool_calls or []
        if tool_calls:
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
            return normalize_messages(
                env_response, field_name="invalid_handoff_termination"
            )
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

        if env_response is None:
            return []
        return normalize_messages(env_response, field_name="apply_handoff")

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
        step_prompt = normalize_messages(step["prompt"], field_name="trajectory.prompt")
        step_completion = normalize_messages(
            step["completion"], field_name="trajectory.completion"
        )
        return concat_messages([step_prompt, step_completion])

    async def get_prompt_messages(self, state: State) -> Messages:
        if len(state["trajectory"]) == 0:
            self.logger.debug(
                "multiagent.turn initial actor=%s",
                state["trajectory_id"],
            )
            return normalize_messages(
                self.get_prompt_for_actor([], state),
                field_name="get_prompt_for_actor",
            )

        self.logger.debug(
            "multiagent.turn resume trajectory_len=%s last_actor=%s",
            len(state["trajectory"]),
            state["trajectory"][-1]["trajectory_id"],
        )
        prev_turn_prompt = normalize_messages(
            state["trajectory"][-1]["prompt"], field_name="trajectory.prompt"
        )
        prev_turn_completion = normalize_messages(
            state["trajectory"][-1]["completion"], field_name="trajectory.completion"
        )
        prev_messages = concat_messages([prev_turn_prompt, prev_turn_completion])
        env_response = await self.env_response(prev_messages, state)
        env_response_messages = normalize_messages(
            env_response, field_name="env_response"
        )
        prev_trajectory_id = state["trajectory"][-1]["trajectory_id"]
        state["trajectory"][-1]["extras"]["env_response"] = env_response_messages
        self.logger.debug(
            "multiagent.env_response stored actor=%s message_count=%s",
            prev_trajectory_id,
            len(env_response_messages),
        )

        actor_id = state["trajectory_id"]
        continuing_same_actor = actor_id == prev_trajectory_id
        messages = self.messages_for_trajectory_id(state, actor_id)
        if messages is None:
            self.logger.debug(
                "multiagent.prompt actor=%s history=none",
                actor_id,
            )
            return normalize_messages(
                self.get_prompt_for_actor([], state),
                field_name="get_prompt_for_actor",
            )

        actor_messages = messages
        prompt_messages = normalize_messages(
            self.get_prompt_for_actor(actor_messages, state),
            field_name="get_prompt_for_actor",
        )

        # Tool-use turn: stay on the same actor and include tool messages to continue.
        if continuing_same_actor:
            self.logger.debug(
                "multiagent.prompt actor=%s history=present env_response_count=%s mode=continue_same_actor",
                actor_id,
                len(env_response_messages),
            )
            return concat_messages(
                [actor_messages, env_response_messages, prompt_messages]
            )

        # Handoff turn: actor switched; next actor does not need previous actor's env response.
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
        for i, step in enumerate(unique_steps):
            step_prompt = normalize_messages(
                step["prompt"], field_name=f"trajectory[{i}].prompt"
            )
            step_completion = normalize_messages(
                step["completion"], field_name=f"trajectory[{i}].completion"
            )
            completion = concat_messages([completion, step_prompt, step_completion])

        if state.get("final_env_response") is not None:
            completion = concat_messages(
                [
                    completion,
                    normalize_messages(
                        state["final_env_response"], field_name="final_env_response"
                    ),
                ]
            )

        state["completion"] = completion
