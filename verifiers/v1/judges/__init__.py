"""Built-in judges, resolved by id (a `taskset.judges` entry's `id`) as
`verifiers.v1.judges.<id>`.

A judge plugin is a module exporting its `Judge` subclass via `__all__` (exactly like a taskset
or harness plugin — see `verifiers.v1.loaders`): the built-ins ship here, any other id names a
local package or an `org/name[@version]` package installed on demand from the Environments Hub.
Re-exports each judge's class + config off the package."""

from verifiers.v1.judges.binary import BinaryJudge, BinaryJudgeConfig
from verifiers.v1.judges.choice import ChoiceJudge, ChoiceJudgeConfig
from verifiers.v1.judges.rubric import Criterion, RubricJudge, RubricJudgeConfig

__all__ = [
    "BinaryJudge",
    "BinaryJudgeConfig",
    "ChoiceJudge",
    "ChoiceJudgeConfig",
    "Criterion",
    "RubricJudge",
    "RubricJudgeConfig",
]
