from __future__ import annotations

import os
import re
from abc import abstractmethod
from typing import Any

from datasets import load_dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.utils.async_utils import maybe_await
from verifiers.types import (
    AssistantMessage,
    Messages,
    State,
    SystemMessage,
    TrajectoryStep,
    UserMessage,
)
from verifiers.utils.message_utils import (
    concat_messages,
    normalize_messages,
)


def content_to_text(content: Any, *, separator: str = "\n") -> str:
    """Extract plain text from message content (string, list of parts, or other)."""
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
        return separator.join(chunks).strip()
    return ""


# ---------------------------------------------------------------------------
# MultiAgentEnv – copied from verifiers/envs/multiagent_env.py
# (not yet published in the verifiers package)
# Modified to use pure XML handoffs instead of JSON-in-XML.
# ---------------------------------------------------------------------------


class MultiAgentEnv(vf.StatefulToolEnv):
    """
    Multi-agent environment on top of StatefulToolEnv.

    Each actor ends their turn with an XML tag: <tag_name>content</tag_name>.
    `state["trajectory_id"]` is the active actor id.
    """

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
    def get_handoff_tag(self, actor_id: str, state: State) -> str:
        """Return the XML tag name for this actor's end-of-turn handoff."""

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

        handoff_tags: dict[str, str] = {}
        system_prompts: dict[str, str] = {}
        for actor_id, base_prompt in actors.items():
            tag = self.get_handoff_tag(actor_id, state)
            if not tag or not isinstance(tag, str):
                raise ValueError(
                    f"Handoff tag for actor '{actor_id}' must be a non-empty string"
                )
            handoff_tags[actor_id] = tag
            system_prompts[actor_id] = self._compose_system_prompt(base_prompt, tag)

        state["trajectory_id"] = initial_actor_id
        state["system_prompts"] = system_prompts
        state["handoff_tags"] = handoff_tags
        state["handoff_history"] = []
        state["last_handoff_by_actor"] = {}

        self.logger.debug(
            "multiagent.setup actors=%s initial_actor=%s",
            list(actors.keys()),
            initial_actor_id,
        )
        return state

    def get_turn_contract_text(self, tag: str) -> str:
        """Return the instructional text appended to every actor's system prompt.

        Override this to customize the turn-contract wording.
        """
        return (
            "To end your turn, wrap your final response in XML tags:\n"
            f"  <{tag}>your response</{tag}>"
        )

    def _compose_system_prompt(self, base_prompt: str, tag: str) -> str:
        return f"{base_prompt}\n\n{self.get_turn_contract_text(tag)}"

    def _tag_for_actor(self, actor_id: str, state: State) -> str:
        return state["handoff_tags"][actor_id]

    def parse_handoff(
        self, actor_id: str, last_message: AssistantMessage, state: State
    ) -> dict[str, Any]:
        message_text = content_to_text(last_message.content)
        if message_text.strip() == "":
            tag = self._tag_for_actor(actor_id, state)
            raise vf.InvalidModelResponseError(
                f"Actor '{actor_id}' produced empty message. Expected <{tag}>...</{tag}>."
            )

        tag = self._tag_for_actor(actor_id, state)
        pattern = re.compile(
            rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", flags=re.DOTALL
        )
        match = pattern.search(message_text)
        if match is None:
            raise vf.InvalidModelResponseError(
                f"Actor '{actor_id}' must end turn with <{tag}>...</{tag}> block."
            )

        content = match.group(1).strip()
        if not content:
            raise vf.InvalidModelResponseError(
                f"Actor '{actor_id}' handoff <{tag}> is empty."
            )

        return {tag: content}

    def on_invalid_handoff(
        self, actor_id: str, error: vf.InvalidModelResponseError, state: State
    ) -> Messages | str | None:
        """Handle an invalid handoff from *actor_id*.

        The default implementation terminates the rollout and records the
        failure in ``state``.  Subclasses can override this to retry, force a
        default action, or apply custom recovery logic.
        """
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
            env_response = self.on_invalid_handoff(actor_id, e, state)
            if env_response is None:
                return []
            return normalize_messages(
                env_response, field_name="invalid_handoff_termination"
            )
        env_response = await maybe_await(self.apply_handoff, actor_id, handoff, state)

        state["handoff_history"].append({"actor": actor_id, "handoff": handoff})
        state["last_handoff_by_actor"][actor_id] = handoff

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


# ---------------------------------------------------------------------------
# Ethics Debate Environment
# ---------------------------------------------------------------------------

ARGUER = "arguer"
CRITIC = "critic"


class EthicsDebateEnv(MultiAgentEnv):
    def __init__(self, num_rounds: int = 2, **kwargs):
        self.num_rounds = num_rounds
        # 2*num_rounds + 1 debate turns, +1 extra so the loop processes the final handoff
        super().__init__(tools=[], max_turns=2 * num_rounds + 2, **kwargs)

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state["prompt"] = normalize_messages(state["prompt"], field_name="prompt")
        state["current_argument"] = None
        state["current_critique"] = None
        state["final_argument"] = None
        return state

    def get_all_actors(self, state: State) -> dict[str, str]:
        return {
            ARGUER: (
                "You are arguing an ethics question. "
                "Present a clear position with reasoning. "
                "When critiqued, address the weaknesses and strengthen your argument."
            ),
            CRITIC: (
                "You are critiquing an ethical argument. "
                "Identify gaps, logical fallacies, and missing perspectives. "
                "Be specific and constructive."
            ),
        }

    def get_initial_actor_id(self, actors: dict[str, str], state: State) -> str:
        return ARGUER

    def get_next_actor_id(self, state: State) -> str:
        return CRITIC if state["trajectory_id"] == ARGUER else ARGUER

    def get_handoff_tag(self, actor_id: str, state: State) -> str:
        return "argument" if actor_id == ARGUER else "critique"

    def get_turn_contract_text(self, tag: str) -> str:
        return (
            f"IMPORTANT: When you are done, wrap your final response in <{tag}>...</{tag}> tags. "
            f"Do NOT write anything after the closing </{tag}> tag."
        )

    async def apply_handoff(
        self, actor_id: str, handoff: dict[str, Any], state: State
    ) -> str | None:
        if actor_id == ARGUER:
            state["current_argument"] = handoff["argument"]
            # Final arguer turn: all prior rounds complete
            if len(state["handoff_history"]) == 2 * self.num_rounds:
                state["final_argument"] = handoff["argument"]
                state["final_env_response"] = "Debate complete."
                return state["final_env_response"]
        else:
            state["current_critique"] = handoff["critique"]
        return None

    def get_prompt_for_actor(self, messages: Messages, state: State) -> Messages:
        actor_id = state["trajectory_id"]
        system = SystemMessage(content=state["system_prompts"][actor_id])
        question = content_to_text(state["prompt"][0].content)

        if not messages:
            if actor_id == ARGUER:
                return [system, UserMessage(content=question)]
            return [
                system,
                UserMessage(
                    content=f"Question: {question}\n\nArgument to critique:\n{state['current_argument']}"
                ),
            ]

        if actor_id == ARGUER:
            return [
                UserMessage(
                    content=f"Critique received:\n{state['current_critique']}\n\nRefine your argument."
                )
            ]
        return [
            UserMessage(
                content=f"Revised argument:\n{state['current_argument']}\n\nIdentify remaining weaknesses."
            )
        ]


# ---------------------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------------------


def load_environment(
    judge_model: str = "google/gemini-3-flash-preview",
    judge_base_url: str = "https://api.pinference.ai/api/v1",
    judge_api_key_var: str = "PRIME_API_KEY",
    num_rounds: int = 2,
    **kwargs,
) -> vf.Environment:
    vf.ensure_keys([judge_api_key_var])

    dataset = load_dataset("ergotts/ethics_questions", split="train")
    dataset = dataset.map(
        lambda x: {"prompt": [{"role": "user", "content": x["question"]}]},
        remove_columns=dataset.column_names,
    )

    judge_client = AsyncOpenAI(
        api_key=os.environ[judge_api_key_var],
        base_url=judge_base_url,
    )

    judge_system = (
        "You are a philosophy professor grading ethics arguments. "
        "Be rigorous and critical. A generic, well-intentioned argument scores 3-5. "
        "Only arguments showing genuine philosophical rigor score 7+."
    )

    judge_user_template = (
        "Question: {question}\n\n"
        "Argument:\n{argument}\n\n"
        "Score each dimension 0-2 (0=absent, 1=superficial, 2=thorough):\n\n"
        "1. THESIS CLARITY: Does it state a specific, defensible position?\n"
        "   A vague 'it depends' or generic both-sidesism = 0. Clear stance with defined scope = 2.\n\n"
        "2. LOGICAL STRUCTURE: Are claims warranted by reasoning, not just asserted?\n"
        "   Deduct for logical fallacies (slippery slope, false dichotomy, appeal to authority).\n\n"
        "3. COUNTERARGUMENT HANDLING: Does it engage the STRONGEST version of opposing views\n"
        "   (steel-man), or only weak caricatures (straw-man)? Explains WHY they fail, not just asserts it?\n\n"
        "4. NUANCE: Acknowledges edge cases, limitations, or conditions where the position might not hold?\n"
        "   Distinguishes between related but different concepts?\n\n"
        "5. DEPTH: Goes beyond common platitudes? Engages specific ethical frameworks\n"
        "   (utilitarianism, deontology, virtue ethics), historical examples, or thought experiments?\n\n"
        "Sum the five scores (0-10). Respond with ONLY the total number."
    )

    async def argument_quality(state: State) -> float:
        final_arg = state.get("final_argument")
        if not final_arg:
            return 0.0
        question = content_to_text(state["prompt"][0].content)
        resp = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": judge_system},
                {
                    "role": "user",
                    "content": judge_user_template.format(
                        question=question, argument=final_arg
                    ),
                },
            ],
        )
        text = resp.choices[0].message.content or ""
        numbers = re.findall(r"\b(10|[0-9])\b", text)
        return float(numbers[0]) / 10.0 if numbers else 0.0

    rubric = vf.Rubric(funcs=[argument_quality])

    return EthicsDebateEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=None,
        num_rounds=num_rounds,
        **kwargs,
    )
