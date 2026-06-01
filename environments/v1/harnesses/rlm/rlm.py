"""``rlm`` harness — installable v1 harness module.

Exports ``load_harness(config: RLMConfig) -> RLM`` so the harness can be
referenced by name from ``vf-eval-v1 <taskset> rlm`` (or directly via
``vf.load_environment(<taskset>, __vf_v1_harness__={"name": "rlm", ...})``).

The implementation comes from ``verifiers.v1.packages.harnesses`` and is
re-exported here for convenience.
"""

from verifiers.v1.packages.harnesses import RLM, RLMConfig

__all__ = ["RLM", "RLMConfig", "load_harness"]


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)
