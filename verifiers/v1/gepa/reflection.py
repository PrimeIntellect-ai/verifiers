"""The reflection (teacher) LM: a plain synchronous string-in/string-out call, decoupled from
rollout execution entirely. Built straight from the v1 client config — the sync sibling of
`verifiers.v1.clients.config.build_async_openai` — so the resolved API key and extra headers
(e.g. Prime team billing) apply to reflection calls exactly as they do to rollouts.
"""

from typing import Callable

from openai import OpenAI

from verifiers.v1.clients.config import resolve_api_key
from verifiers.v1.gepa.config import GEPAConfig


def build_reflection_lm(config: GEPAConfig) -> Callable[[str], str]:
    client_config = config.reflection_client or config.client
    client = OpenAI(
        base_url=client_config.base_url,
        api_key=resolve_api_key(client_config),
        default_headers=client_config.headers or None,
    )
    model = config.reflection_model or config.model

    def reflection_lm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content or ""

    return reflection_lm
