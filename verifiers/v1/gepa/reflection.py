"""The reflection (teacher) LM: GEPA's proposer calls this synchronously to turn a prompt into
text. Backed by a vf `Judge` — `Judge.complete` owns the model client (same key / header /
Prime-billing resolution rollouts use), so we never build one here. GEPA calls it from the
worker thread running `optimize()`, so each call drives the async `complete` on a private loop.
"""

import asyncio
from typing import Callable

from verifiers.v1.gepa.config import GEPAConfig
from verifiers.v1.judge import Judge, JudgeConfig


def build_reflection_lm(config: GEPAConfig) -> Callable[[str], str]:
    client = config.reflection_client or config.client
    judge = Judge(
        JudgeConfig(
            model=config.reflection_model or config.model,
            base_url=client.base_url,
            api_key_var=client.api_key_var,
            headers=client.headers,
        )
    )

    def reflection_lm(prompt: str) -> str:
        return asyncio.run(judge.complete(prompt)).text

    return reflection_lm
