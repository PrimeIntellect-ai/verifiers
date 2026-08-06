"""A Prime Agent ACP turn that fails at the provider boundary."""

import verifiers.v1 as vf


def has_raised_provider_failure(trace: vf.Trace) -> bool:
    """The failed ACP request surfaced as a rollout error, not a clean stop."""
    return bool(
        not trace.ok
        and trace.last_error is not None
        and trace.stop_condition is None
        and bool(trace.calls)
        and trace.calls[-1].error is not None
        and trace.calls[-1].error.type == "ProviderError"
    )


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
