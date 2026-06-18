"""Fixture taskset requiring a browser-control harness capability."""

import verifiers.v1 as vf


class CapabilityRequiredTaskset(vf.Taskset[vf.Task, vf.TasksetConfig]):
    REQUIRED_HARNESS_CAPABILITIES = frozenset({vf.HarnessCapability.BROWSER_CONTROL})

    def load_tasks(self) -> list[vf.Task]:
        return [vf.Task(idx=0, prompt="noop")]


__all__ = ["CapabilityRequiredTaskset"]
