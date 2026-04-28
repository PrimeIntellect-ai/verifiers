"""SolveHarness — golden-patch validation through ComposableEnv.

This is the harness used to run a ``SandboxTaskSet``'s gold-patch
validation (``taskset.validate_instance(state)``) through the standard
rollout pipeline — no LLM inference, no agent install — so callers that
already speak ``ComposableEnv`` (``vf-eval``, training-eval configs,
result-jsonl tooling) can score the upper bound on a taskset without
any model.

Used to:

- validate task integrity end-to-end before launching a real agent run
- establish per-instance reward upper bounds
- cheaply iterate on taskset / sandbox plumbing

Example
-------

::

    from verifiers.envs.experimental.composable import (
        ComposableEnv,
        solve_harness,
    )
    from verifiers.envs.experimental.composable.tasksets.swe import (
        make_swe_taskset,
    )

    taskset = make_swe_taskset(backend="r2e")
    env = ComposableEnv(
        taskset=taskset,
        harness=solve_harness(),
        keep_sandbox_for_scoring=True,
    )

The harness sets ``Harness.solve_only=True``; ``ComposableEnv`` reads
that flag and (a) skips agent install / run, and (b) invokes
``taskset.validate_instance(state)`` directly during sandbox setup so
the existing rubric (which reads ``state["test_output"]``) scores the
gold patch as if an agent had produced it.
"""

from __future__ import annotations

from verifiers.envs.experimental.composable.harness import Harness


def solve_harness() -> Harness:
    """Return a no-op harness that triggers gold-patch validation.

    The returned harness has ``solve_only=True`` and a no-op
    ``run_command`` / ``install_script`` — ``ComposableEnv`` short-
    circuits the agent round-trip when it sees this flag and runs
    ``taskset.validate_instance(state)`` inside the sandbox instead.
    """
    return Harness(
        run_command="true",
        install_script=None,
        solve_only=True,
    )
