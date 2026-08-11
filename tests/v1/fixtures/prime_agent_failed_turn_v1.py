"""A Prime Agent ACP turn that fails at the provider boundary."""

import verifiers.v1 as vf


def has_raised_provider_failure(trace: vf.Trace) -> bool:
    """The failed ACP request surfaced as a rollout error, not a clean stop."""
    # A provider failure ends the rollout as an error, so `stop_condition` is
    # "error" rather than None, and the request may fail before any ModelCall is
    # recorded -- `trace.calls` can be empty. What must hold is that the failure
    # is visible as a ProviderError and the rollout did NOT report success.
    if trace.ok or not trace.errors:
        return False
    if trace.stop_condition not in (None, "error"):
        return False
    recorded = [*trace.errors, *(c.error for c in trace.calls if c.error is not None)]
    return any(getattr(error, "type", None) == "ProviderError" for error in recorded)


class PrimeAgentFailedTurnTask(vf.Task):
    pass


class PrimeAgentFailedTurnTaskset(
    vf.Taskset[PrimeAgentFailedTurnTask, vf.TasksetConfig]
):
    def load(self) -> list[PrimeAgentFailedTurnTask]:
        return [
            PrimeAgentFailedTurnTask(
                vf.TaskData(
                    idx=0,
                    prompt="Reply with exactly READY.",
                    system_prompt="Follow the instruction exactly.",
                )
            )
        ]


__all__ = ["PrimeAgentFailedTurnTaskset", "has_raised_provider_failure"]
