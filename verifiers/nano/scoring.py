"""Scoring: run a taskset's decorated rewards/metrics over a finished transcript.

Rewards and metrics are `async` decorated methods on the Taskset with the fixed
signature `async (self, task, transcript) -> float`. This replaces v1's 440-line
signal machinery (string-named config rewards + named-arg injection):
- each `@metric` value is recorded in `transcript.metrics[name]`,
- each `@reward` value, weighted, is recorded in `transcript.rewards[name]`;
  `transcript.reward` is the computed sum of those contributions.
"""

from verifiers.nano.decorators import discover_decorated
from verifiers.nano.transcript import Transcript


async def score(taskset: object, transcript: Transcript) -> None:
    """Run all `@metric`/`@reward` methods on `taskset`, mutating `transcript`."""
    task = transcript.task

    async def value(fn) -> float:
        return float(await fn(task, transcript))

    for fn in discover_decorated(taskset, "metric"):
        transcript.metrics[fn.__name__] = await value(fn)
    for fn in discover_decorated(taskset, "reward"):
        weight = float(getattr(fn, "_vf_weight", 1.0))
        transcript.rewards[fn.__name__] = await value(fn) * weight
