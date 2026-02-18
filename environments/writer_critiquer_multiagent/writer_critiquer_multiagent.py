from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence

from datasets import Dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import (
    AssistantMessage,
    Messages,
    State,
    SystemMessage,
    Tool,
    UserMessage,
)

logger = logging.getLogger(__name__)


WRITING_JUDGE_PROMPT = """You are grading the final draft from a writer/critic collaboration.

Topic:
{answer}

Final draft:
{response}

Score the final draft from 0 to 10 using this rubric:
1) Topic fit and relevance (0-2): Does the draft directly and fully address the topic?
2) Clarity and coherence (0-2): Is the writing easy to follow with clear logic and transitions?
3) Organization and structure (0-2): Is there a clear beginning, middle, and end with good flow?
4) Style and engagement (0-2): Is the writing compelling, vivid, and appropriately toned?
5) Mechanics and polish (0-2): Grammar, diction, and sentence quality.

Output format (required):
SCORE: <number from 0 to 10>
RATIONALE: <1-3 sentences with key strengths and weaknesses>
"""


DEFAULT_TOPICS = [
    "A city redesign that prioritizes pedestrians over cars.",
    "How a community can rebuild trust after a major local controversy.",
    "The ethics of using AI assistants in classrooms.",
    "A personal story about learning patience through failure.",
    "Why public libraries still matter in a digital-first world.",
    "A policy memo arguing for or against a four-day work week.",
    "How climate adaptation changes daily life in coastal towns.",
    "The role of mentorship in early-career growth.",
]


class WriterCritiquerMultiAgentEnv(vf.MultiAgentEnv):
    WRITER_ID = "writer"
    CRITIQUER_ID = "critiquer"

    WRITER_TOOL_NAME = "submit_draft"
    CRITIQUER_TOOL_NAME = "submit_feedback"

    def __init__(self, max_turns: int = 6, **kwargs):
        super().__init__(tools=[], max_turns=max_turns, **kwargs)
        self.add_tool(self.submit_draft, args_to_skip=["state"])
        self.add_tool(self.submit_feedback, args_to_skip=["state"])

        self.actor_tool_names = {
            self.WRITER_ID: self.WRITER_TOOL_NAME,
            self.CRITIQUER_ID: self.CRITIQUER_TOOL_NAME,
        }
        self.tool_defs_by_name = {
            tool_def.name: tool_def for tool_def in (self.tool_defs or [])
        }

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)

        input_obj = state.get("input", {})
        topic = input_obj.get("topic", "") if isinstance(input_obj, dict) else ""
        if not isinstance(topic, str) or topic.strip() == "":
            prompt_messages = state["prompt"]
            if not prompt_messages:
                raise ValueError("Expected non-empty state['prompt']")
            topic = str(prompt_messages[0].content)

        state["topic"] = topic.strip()
        state["draft_history"] = []
        state["feedback_history"] = []
        state["collaboration_log"] = []

        self._set_visible_tool_for_actor(state["trajectory_id"], state)
        self.logger.debug(
            "writer_critiquer.setup topic=%r initial_actor=%s max_turns=%s",
            state["topic"][:80],
            state["trajectory_id"],
            self.max_turns,
        )
        return state

    def get_all_actors(self, state: State) -> dict[str, str]:
        return {
            self.WRITER_ID: (
                "You are the writer in a two-agent collaboration. "
                "Write strong, clear, specific prose that addresses the topic and latest critique. "
                f"On every turn call `{self.WRITER_TOOL_NAME}` exactly once with your full draft text."
            ),
            self.CRITIQUER_ID: (
                "You are the critiquer in a two-agent collaboration. "
                "Give specific, actionable, high-signal feedback that improves structure, clarity, style, and correctness. "
                f"On every turn call `{self.CRITIQUER_TOOL_NAME}` exactly once with your feedback text."
            ),
        }

    def get_initial_actor_id(self, actors: dict[str, str], state: State) -> str:
        return self.WRITER_ID

    def get_next_actor_id(self, state: State) -> str:
        if state["trajectory_id"] == self.WRITER_ID:
            return self.CRITIQUER_ID
        return self.WRITER_ID

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict:
        return {**tool_args, "state": state}

    def _set_visible_tool_for_actor(self, actor_id: str, state: State) -> None:
        tool_name = self.actor_tool_names[actor_id]
        tool_def: Tool | None = self.tool_defs_by_name.get(tool_name)
        if tool_def is None:
            raise ValueError(f"Missing tool definition for '{tool_name}'")
        state["tool_defs"] = [tool_def]

    def _latest_draft(self, state: State) -> str | None:
        draft_history = state.get("draft_history", [])
        if isinstance(draft_history, list) and draft_history:
            last = draft_history[-1]
            return str(last)
        return None

    def _latest_feedback(self, state: State) -> str | None:
        feedback_history = state.get("feedback_history", [])
        if isinstance(feedback_history, list) and feedback_history:
            last = feedback_history[-1]
            return str(last)
        return None

    async def submit_draft(self, writing: str, state: State) -> str:
        """
        Store a writer draft version.

        Args:
            writing: Full draft text from the writer.
            state: Rollout state (injected by environment).

        Returns:
            Confirmation string.
        """
        text = writing.strip()
        if text == "":
            return "Error: draft cannot be empty."

        state["draft_history"].append(text)
        draft_num = len(state["draft_history"])
        state["collaboration_log"].append(
            {
                "actor": self.WRITER_ID,
                "event": "draft",
                "index": draft_num,
                "content": text,
            }
        )
        self.logger.debug(
            "writer_critiquer.draft num=%d len=%d preview=%r",
            draft_num,
            len(text),
            text[:120],
        )
        return f"Stored draft #{draft_num}."

    async def submit_feedback(self, feedback: str, state: State) -> str:
        """
        Store a critique feedback message.

        Args:
            feedback: Feedback text from the critiquer.
            state: Rollout state (injected by environment).

        Returns:
            Confirmation string.
        """
        text = feedback.strip()
        if text == "":
            return "Error: feedback cannot be empty."

        state["feedback_history"].append(text)
        feedback_num = len(state["feedback_history"])
        state["collaboration_log"].append(
            {
                "actor": self.CRITIQUER_ID,
                "event": "feedback",
                "index": feedback_num,
                "content": text,
            }
        )
        self.logger.debug(
            "writer_critiquer.feedback num=%d len=%d preview=%r",
            feedback_num,
            len(text),
            text[:120],
        )
        return f"Stored feedback #{feedback_num}."

    def _writer_user_prompt(self, state: State) -> str:
        topic = state["topic"]
        latest_feedback = self._latest_feedback(state)
        feedback_text = (
            latest_feedback
            if latest_feedback is not None
            else "No prior feedback yet. Write the first complete draft."
        )

        return (
            f"Writing topic:\n{topic}\n\n"
            f"Most recent feedback:\n{feedback_text}\n\n"
            "Write a complete draft that directly addresses the topic and incorporates the feedback where useful. "
            "Call submit_draft once with the full draft in the `writing` field."
        )

    def _critiquer_user_prompt(self, state: State) -> str:
        topic = state["topic"]
        latest_draft = self._latest_draft(state)
        draft_text = (
            latest_draft if latest_draft is not None else "No draft is available yet."
        )

        return (
            f"Writing topic:\n{topic}\n\n"
            f"Most recent draft:\n{draft_text}\n\n"
            "Give specific, prioritized, and actionable critique to improve the next draft. "
            "Cover structure, argument quality, clarity, style, and mechanics. "
            "Call submit_feedback once with your feedback in the `feedback` field."
        )

    def get_prompt_for_actor(self, messages: Messages, state: State) -> Messages:
        actor_id = state["trajectory_id"]
        self._set_visible_tool_for_actor(actor_id, state)

        if actor_id == self.WRITER_ID:
            user_content = self._writer_user_prompt(state)
        else:
            user_content = self._critiquer_user_prompt(state)

        self.logger.debug(
            "writer_critiquer.prompt actor=%s drafts=%d feedbacks=%d history_len=%d",
            actor_id,
            len(state.get("draft_history", [])),
            len(state.get("feedback_history", [])),
            len(messages),
        )

        user_message = UserMessage(content=user_content)
        if len(messages) == 0:
            system_prompt = state["system_prompts"][actor_id]
            return [SystemMessage(content=system_prompt), user_message]
        return [user_message]


def _build_dataset(topics: Sequence[str]) -> Dataset:
    rows = []
    for topic in topics:
        topic_text = str(topic).strip()
        if topic_text == "":
            continue
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"Topic: {topic_text}"}],
                "task": "writer_critiquer_multiagent",
                "answer": topic_text,
                "topic": topic_text,
            }
        )

    if not rows:
        raise ValueError("At least one non-empty topic is required")
    return Dataset.from_list(rows)


def _parse_judge_score(judge_response: str) -> float:
    labeled = re.search(
        r"(?i)score\s*[:=]\s*(10(?:\.0+)?|[0-9](?:\.\d+)?)", judge_response
    )
    if labeled:
        value = float(labeled.group(1))
    else:
        fallback = re.search(r"\b(10(?:\.0+)?|[0-9](?:\.\d+)?)\b", judge_response)
        if fallback is None:
            return 0.0
        value = float(fallback.group(1))

    value = max(0.0, min(10.0, value))
    return value / 10.0


def load_environment(
    topics: Sequence[str] | None = None,
    max_turns: int = 5,
    judge_model: str = "openai/gpt-5.2",
    judge_base_url: str = "https://api.pinference.ai/api/v1",
    judge_api_key_var: str = "PRIME_API_KEY",
) -> vf.Environment:
    vf.ensure_keys([judge_api_key_var])

    selected_topics: list[str]
    if topics is None:
        selected_topics = list(DEFAULT_TOPICS)
    elif isinstance(topics, str):
        selected_topics = [topics]
    else:
        selected_topics = [str(topic) for topic in topics]

    dataset = _build_dataset(selected_topics)

    judge_client = AsyncOpenAI(
        api_key=os.environ[judge_api_key_var],
        base_url=judge_base_url,
    )

    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=WRITING_JUDGE_PROMPT,
    )

    async def writing_quality_reward(judge, state: State, **kwargs) -> float:
        draft_history = state.get("draft_history", [])
        if not isinstance(draft_history, list) or not draft_history:
            logger.debug("writer_critiquer.judge skipped, no drafts")
            return 0.0

        final_draft = str(draft_history[-1])
        topic = str(state.get("topic", state.get("answer", "")))
        topic_message = UserMessage(content=f"Topic: {topic}")
        completion_message = AssistantMessage(content=final_draft)

        formatted_judge_prompt = WRITING_JUDGE_PROMPT.format(
            answer=topic, response=final_draft
        )
        logger.info(
            "writer_critiquer.judge_input topic=%r draft_len=%d draft_num=%d\n--- JUDGE PROMPT ---\n%s\n--- END JUDGE PROMPT ---",
            topic[:80],
            len(final_draft),
            len(draft_history),
            formatted_judge_prompt,
        )

        judge_response = await judge(
            [topic_message],
            [completion_message],
            topic,
            state,
        )
        score = _parse_judge_score(judge_response)
        logger.info(
            "writer_critiquer.judge_output score=%.2f\n--- JUDGE RESPONSE ---\n%s\n--- END JUDGE RESPONSE ---",
            score,
            judge_response,
        )
        return score

    rubric.add_reward_func(writing_quality_reward, weight=1.0)

    return WriterCritiquerMultiAgentEnv(
        dataset=dataset,
        rubric=rubric,
        system_prompt=None,
        max_turns=max_turns,
    )
