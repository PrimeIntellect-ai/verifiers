"""Built-in judges, resolved by id (a `taskset.judges` entry's `id`) as
`verifiers.v1.judges.<id>`.

A judge plugin is a module exporting its `Judge` subclass via `__all__` (exactly like a taskset
or harness plugin — see `verifiers.v1.loaders`): the built-ins ship here, any other id names a
locally importable package.
Re-exports each judge's class + config off the package."""

from verifiers.v1.judges.reference import ReferenceJudge, ReferenceJudgeConfig
from verifiers.v1.judges.rubric import Criterion, RubricJudge, RubricJudgeConfig

__all__ = [
    "ReferenceJudge",
    "ReferenceJudgeConfig",
    "Criterion",
    "RubricJudge",
    "RubricJudgeConfig",
]
