"""counter (v1, shared state): a `@vf.tool` accumulates into `trace.state`, scored from it.

The v1 state fixture for the e2e matrix. The taskset declares a typed `CounterState` and a
`CounterToolset` whose `bump` tool increments `self.state.count` on each call — the shared
per-rollout state, synced to the host's `trace.state` over the interception server. The `@reward`
reads `trace.state.count` back, so a non-zero reward proves the whole round-trip: tool write ->
host `trace.state` -> scoring. Placement is CLI-tunable like any toolset (colocated / own runtime);
`shared` is excluded by the test (state is per-rollout, not wired to an eval-level server).
"""

import verifiers.v1 as vf

TARGET = (
    3  # calls the prompt asks for; the reward accepts >= 2 (margin for an under-call)
)


class CounterState(vf.State):
    count: int = 0


class CounterToolset(vf.Toolset[vf.ToolsetConfig, CounterState]):
    TOOL_PREFIX = "counter"  # the model sees `counter_bump`

    @vf.tool
    def bump(self) -> str:
        """Increment the shared counter and return its new value."""
        self.state.count += 1
        return f"count={self.state.count}"


class CounterTask(vf.Task[CounterState]):
    tools = (CounterToolset,)
    # Built with the taskset config's `tools` field (placement stays CLI-tunable),
    # resolved by `Task.server_config`.

    @vf.reward(weight=1.0)
    async def counted(self, trace: vf.Trace) -> float:
        # The tool incremented `self.state.count` per call and pushed it to `trace.state` over the
        # interception channel; reading it back non-zero here proves the round-trip (>= 2 accumulated).
        return float(trace.state.count >= 2)


class CounterConfig(vf.TasksetConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class CounterTaskset(vf.Taskset[CounterTask, CounterConfig]):
    def load(self) -> list[CounterTask]:
        return [
            CounterTask(
                idx=0,
                prompt=(
                    f"Call the `counter_bump` tool {TARGET} times, one call per turn — wait for "
                    "each result before the next. After the last result, reply with "
                    "<answer>done</answer>."
                ),
            )
        ]


__all__ = ["CounterTaskset"]


if __name__ == "__main__":
    CounterToolset.run()
