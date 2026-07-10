"""Task-scoped counter tool with typed rollout state."""

import verifiers.v1 as vf

TARGET = 3


class CounterState(vf.State):
    count: int = 0


class CounterToolset(vf.Toolset[vf.ToolsetConfig, CounterState]):
    TOOL_PREFIX = "counter"

    @vf.tool
    def bump(self) -> str:
        """Increment the shared counter and return its new value."""
        self.state.count += 1
        return f"count={self.state.count}"


class CounterTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class CounterTask(vf.Task[vf.TaskData, CounterState, CounterTaskConfig]):
    tools = (CounterToolset,)

    @vf.reward(weight=1.0)
    async def counted(self, trace: vf.Trace) -> float:
        return float(trace.state.count >= 2)


class CounterConfig(vf.TasksetConfig):
    task: CounterTaskConfig = CounterTaskConfig()


class CounterTaskset(vf.Taskset[CounterTask, CounterConfig]):
    def load(self) -> list[CounterTask]:
        return [
            CounterTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        f"Call the `counter_bump` tool {TARGET} times, one call per turn — wait "
                        "for each result before the next. After the last result, reply with "
                        "<answer>done</answer>."
                    ),
                ),
                self.config.task,
            )
        ]


__all__ = ["CounterTaskset"]


if __name__ == "__main__":
    CounterToolset.run()
