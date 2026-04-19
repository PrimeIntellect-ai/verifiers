"""GPQA under a two-debater + judge protocol.

A ``vf-eval``-discoverable environment. Wraps ``DebateEnv`` with:
  * GPQA dataset loading (shuffled A/B/C/D)
  * A default 5-slot propose → critique → final schedule matching
    the ``selfplay`` pack's phase names
  * Per-agent OpenRouter client overrides driven by ``--env-args`` JSON

Launch
------
    export HF_TOKEN=...                  # GPQA is gated
    export OPENROUTER_API_KEY=...

    vf-install gpqa_debate               # once
    vf-eval gpqa_debate                  # minimal run
        --provider openrouter
        --model openai/gpt-4.1-mini
        -n 50 -r 4

    # with per-agent overrides
    vf-eval gpqa_debate
        --provider openrouter
        --model openai/gpt-4.1-mini
        -n 50 -r 4
        --env-args '{
            "prompts_ref": "selfplay",
            "judge_model": "anthropic/claude-3.5-sonnet",
            "debater_providers": ["anthropic", "deepinfra"],
            "judge_providers":   ["anthropic"],
            "allow_fallbacks": false,
            "num_eval_examples": 50,
            "subset": "gpqa_diamond"
        }'

env_args schema
---------------
All optional. Any omitted keys inherit the vf-eval top-level model/provider.

    prompts_ref        str      built-in name or /path/to/pack.yaml   "selfplay"
    subset             str      gpqa_diamond | gpqa_main              "gpqa_diamond"
    num_eval_examples  int      dataset slice (-1 = all)              -1
    seed               int      choice-shuffling seed                 0
    schedule           list     override SCHEDULE; list of dicts      DEFAULT_SCHEDULE
    truth_member       str      member credited as winning seat        "debater_a"

    # Per-agent OpenRouter overrides (each optional)
    debater_model      str      slug for both debaters                 None (use default)
    judge_model        str      slug for the judge                     None (use default)
    debater_providers  list[str]  strict provider order for debaters   None (OR chooses)
    judge_providers    list[str]  strict provider order for the judge  None
    allow_fallbacks    bool     if true, OR may fall back to non-listed  False
"""

from __future__ import annotations

import os
import random
from typing import Any

from datasets import Dataset, load_dataset
from openai import AsyncOpenAI

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from verifiers.envs.debate_env import DebateEnv, load_environment as _debate_load_env


LETTERS = ("A", "B", "C", "D")

DEFAULT_SCHEDULE = [
    {"slot_id": 0, "agents": ["debater_a"], "phase": "propose"},
    {"slot_id": 1, "agents": ["debater_b"], "phase": "propose"},
    {"slot_id": 2, "agents": ["debater_a"], "phase": "critique"},
    {"slot_id": 3, "agents": ["debater_b"], "phase": "critique"},
    {"slot_id": 4, "agents": ["judge"], "phase": "final"},
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _format_row(row: dict, rng: random.Random) -> dict:
    correct = row["Correct Answer"].strip()
    choices = [
        correct,
        row["Incorrect Answer 1"].strip(),
        row["Incorrect Answer 2"].strip(),
        row["Incorrect Answer 3"].strip(),
    ]
    rng.shuffle(choices)
    truth_letter = LETTERS[choices.index(correct)]
    body = "\n".join(f"{L}) {c}" for L, c in zip(LETTERS, choices))
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"{row['Question'].strip()}\n\n{body}\n\n"
                    "Reply with exactly one of A, B, C, or D as your <answer>."
                ),
            }
        ],
        "answer": truth_letter,
        "example_id": str(row.get("Record ID") or f"gpqa_{rng.random():.10f}"),
        "task": "gpqa_debate",
    }


def _build_dataset(subset: str, n: int, seed: int) -> Dataset:
    raw = list(load_dataset("Idavidrein/gpqa", subset, split="train"))
    rng = random.Random(seed)
    rows = raw if n == -1 else raw[:n]
    return Dataset.from_list([_format_row(r, rng) for r in rows])


# ---------------------------------------------------------------------------
# OpenRouter client with per-agent provider pinning
# ---------------------------------------------------------------------------


class _OpenRouterProviderClient(OpenAIChatCompletionsClient):
    """Injects OpenRouter's ``extra_body.provider`` preference on every
    ``get_response`` call. Provider prefs are per-request, so per-agent
    pinning requires one instance per distinct provider config — wired
    through ``agent_overrides={...}``.
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        providers: list[str] | None,
        allow_fallbacks: bool,
    ) -> None:
        super().__init__(openai_client)
        self._providers = providers
        self._allow_fallbacks = allow_fallbacks

    async def get_response(
        self, prompt, model, sampling_args=None, tools=None, **kwargs
    ):
        args = dict(sampling_args or {})
        if self._providers:
            extra_body = dict(args.get("extra_body") or {})
            extra_body["provider"] = {
                "order": list(self._providers),
                "allow_fallbacks": self._allow_fallbacks,
            }
            args["extra_body"] = extra_body
        return await super().get_response(
            prompt=prompt, model=model, sampling_args=args, tools=tools, **kwargs
        )


def _make_openrouter_client(
    providers: list[str] | None, allow_fallbacks: bool
) -> _OpenRouterProviderClient:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set — required for per-agent "
            "provider pinning in gpqa_debate"
        )
    return _OpenRouterProviderClient(
        AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1"),
        providers=providers,
        allow_fallbacks=allow_fallbacks,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_environment(
    prompts_ref: str = "selfplay",
    subset: str = "gpqa_diamond",
    num_eval_examples: int = -1,
    seed: int = 0,
    schedule: list[dict] | None = None,
    truth_member: str = "debater_a",
    debater_model: str | None = None,
    judge_model: str | None = None,
    debater_providers: list[str] | None = None,
    judge_providers: list[str] | None = None,
    allow_fallbacks: bool = False,
    **extra: Any,
) -> DebateEnv:
    """vf-eval entry point. See module docstring for env_args schema."""
    # Build agent_overrides only for the dimensions the caller specified.
    # If no per-agent overrides are given, vf-eval's top-level (model,
    # client) flows through unchanged.
    agent_overrides: dict[str, tuple] = {}

    if debater_model is not None or debater_providers is not None:
        debater_client = (
            _make_openrouter_client(debater_providers, allow_fallbacks)
            if debater_providers is not None
            else None
        )
        for mid in ("debater_a", "debater_b"):
            agent_overrides[mid] = (debater_client, debater_model)

    if judge_model is not None or judge_providers is not None:
        judge_client = (
            _make_openrouter_client(judge_providers, allow_fallbacks)
            if judge_providers is not None
            else None
        )
        agent_overrides["judge"] = (judge_client, judge_model)

    return _debate_load_env(
        schedule_slots=schedule or DEFAULT_SCHEDULE,
        members=["debater_a", "debater_b", "judge"],
        truth_member=truth_member,
        prompts_ref=prompts_ref,
        agent_overrides=agent_overrides or None,
        eval_dataset=_build_dataset(subset, num_eval_examples, seed),
        **extra,
    )
