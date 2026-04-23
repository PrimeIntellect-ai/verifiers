"""OpenAI-compatible text model for mini-SWE-agent.

Installed into the sandbox by the harness and loaded through mini's
``--model-class`` import path.
"""

from __future__ import annotations

import os
import time
from typing import Any, cast

from minisweagent.models.utils.actions_text import (  # ty: ignore[unresolved-import]
    format_observation_messages,
    parse_regex_actions,
)
from minisweagent.models.utils.openai_multimodal import (  # ty: ignore[unresolved-import]
    expand_multimodal_content,
)
from openai import OpenAI
from pydantic import BaseModel


class OpenAITextModelConfig(BaseModel):
    """mini-SWE-agent model settings backed by an OpenAI-compatible endpoint."""

    model_name: str
    model_kwargs: dict[str, Any] = {}
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, "
        "found {{actions|length}} actions."
    )
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


class OpenAITextModel:
    """Small mini-SWE-agent model adapter that calls the OpenAI SDK directly."""

    def __init__(self, *, config_class: type = OpenAITextModelConfig, **kwargs):
        """Create an OpenAI client using the intercepted sandbox endpoint."""
        self.config = config_class(**kwargs)
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "intercepted"),
            base_url=os.environ["OPENAI_BASE_URL"],
        )

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Run one chat completion and return mini-SWE-agent's message shape."""
        create_completion = cast(Any, self.client.chat.completions.create)
        response = create_completion(
            model=self.config.model_name,
            messages=self._prepare_messages(messages),
            **(self._model_kwargs() | kwargs),
        )
        message = response.choices[0].message.model_dump()
        message["content"] = message.get("content") or ""
        message["extra"] = {
            "actions": self._parse_actions(message["content"]),
            "response": response.model_dump(),
            "cost": 0.0,
            "timestamp": time.time(),
        }
        return message

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Remove mini-SWE-agent bookkeeping before sending messages upstream."""
        return [
            {k: v for k, v in message.items() if k != "extra"} for message in messages
        ]

    def _model_kwargs(self) -> dict[str, Any]:
        """Drop LiteLLM-only options before calling the OpenAI SDK."""
        ignored = {"api_base", "custom_llm_provider", "drop_params"}
        return {k: v for k, v in self.config.model_kwargs.items() if k not in ignored}

    def _parse_actions(self, content: str) -> list[dict]:
        """Extract mini-SWE-agent shell actions from the model response."""
        return parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
        )

    def format_message(self, **kwargs) -> dict:
        """Format a user/assistant message with optional multimodal expansion."""
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format command observations for the next mini-SWE-agent turn.

        The unused message parameter is part of mini-SWE-agent's model interface.
        """
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        """Expose config values to mini-SWE-agent Jinja templates."""
        return self.config.model_dump() | kwargs

    def serialize(self) -> dict:
        """Return mini-SWE-agent metadata for trajectory serialization."""
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
