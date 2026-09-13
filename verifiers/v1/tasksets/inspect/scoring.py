"""Inspect scorer metadata and scoring against a Verifiers completion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from verifiers.v1.tasksets.inspect.compat import registry_spec, require_inspect

SUPPORTED_SCORERS = {
    "inspect_ai/answer",
    "inspect_ai/exact",
    "inspect_ai/f1",
    "inspect_ai/includes",
    "inspect_ai/match",
    "inspect_ai/pattern",
}


class InspectScorerSpec(BaseModel):
    """A built-in, output-only Inspect scorer that can cross the v1 wire."""

    model_config = ConfigDict(frozen=True)

    name: str
    args: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def short_name(self) -> str:
        return self.name.rsplit("/", 1)[-1]


def compile_scorer(scorer: Any) -> InspectScorerSpec:
    """Compile an Inspect scorer to the deliberately small portable subset."""
    name, args = registry_spec(scorer)
    if name not in SUPPORTED_SCORERS:
        raise ValueError(
            f"Inspect scorer {name!r} needs a benchmark-specific v1 port; "
            f"supported output-only scorers are {sorted(SUPPORTED_SCORERS)}"
        )
    if name == "inspect_ai/f1" and args.get("answer_fn") is not None:
        raise ValueError("Inspect f1(answer_fn=...) contains executable scorer code")
    try:
        typed_args = TypeAdapter(dict[str, JsonValue]).validate_python(args)
    except ValueError as e:
        raise ValueError(
            f"Inspect scorer {name!r} has non-serializable arguments and needs a "
            "benchmark-specific v1 port"
        ) from e
    return InspectScorerSpec(name=name, args=typed_args)


async def score_completion(
    spec: InspectScorerSpec,
    completion: str,
    *,
    sample_id: str | int,
    prompt: str,
    targets: list[str],
    choices: list[str],
    metadata: dict[str, JsonValue],
) -> tuple[float, dict[str, JsonValue]]:
    """Run a supported Inspect scorer with a minimal scoring-only TaskState."""
    require_inspect()
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
    from inspect_ai.scorer import (
        Target,
        answer,
        exact,
        f1,
        includes,
        match,
        pattern,
        value_to_float,
    )
    from inspect_ai.solver import TaskState

    factories = {
        "answer": answer,
        "exact": exact,
        "f1": f1,
        "includes": includes,
        "match": match,
        "pattern": pattern,
    }
    factory = factories[spec.short_name]
    scorer = factory(**spec.args)
    output = ModelOutput.from_content("verifiers", completion)
    state = TaskState(
        model="verifiers",
        sample_id=sample_id,
        epoch=1,
        input=prompt,
        messages=[
            ChatMessageUser(content=prompt, source="input"),
            ChatMessageAssistant(
                content=completion,
                model="verifiers",
                source="generate",
            ),
        ],
        target=Target(targets),
        choices=choices,
        output=output,
        completed=True,
        metadata=dict(metadata),
    )
    score = await scorer(state, Target(targets))
    if score is None:
        raise ValueError(f"Inspect scorer {spec.name!r} returned no score")
    if isinstance(score.value, Mapping) or (
        isinstance(score.value, Sequence) and not isinstance(score.value, str)
    ):
        raise TypeError(
            f"Inspect scorer {spec.name!r} returned a structured score; add a "
            "benchmark-specific reward reducer"
        )
    value = value_to_float()(score.value)
    if not math.isfinite(value):
        raise ValueError(
            f"Inspect scorer {spec.name!r} returned an unscored/non-finite value"
        )
    details = TypeAdapter(dict[str, JsonValue]).validate_python(
        score.model_dump(mode="json")
    )
    return value, details
