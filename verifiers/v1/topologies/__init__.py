"""Built-in topologies, each a plugin module resolved by id (see `verifiers.v1.loaders`):

- `llm-judge` — any taskset, LLM-judged: a solver agent runs the task, a second
  (non-trainable) judge agent — fixed to the in-process `direct` chat loop, one API call —
  grades the solver's final answer against the task and its ground truth, and the verdict
  lands on the solver's trace as a deferred reward.
- `agentic-judge` — any taskset, judged by a real agent: the solver's entire serialized
  trace is uploaded into the judge's own runtime, and the judge (bash+edit `default`
  harness by default, configurable) investigates it with tools before committing to a
  score. Same verdict contract as `llm-judge`.

A topology module exports its `Topology` subclass via `__all__`; custom topologies live
under `environments/` or on the Environments Hub, exactly like tasksets and harnesses.
"""

from verifiers.v1.topologies.agentic_judge import (
    AgenticJudgeConfig,
    AgenticJudgeTopology,
)
from verifiers.v1.topologies.llm_judge import LLMJudgeConfig, LLMJudgeTopology

__all__ = [
    "AgenticJudgeConfig",
    "AgenticJudgeTopology",
    "LLMJudgeConfig",
    "LLMJudgeTopology",
]
