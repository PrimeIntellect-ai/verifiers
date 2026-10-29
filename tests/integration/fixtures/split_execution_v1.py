"""Check that a split harness's tools act on the workspace, not the harness's VM."""

import json
from typing import Literal

import verifiers.v1 as vf


class SplitExecutionData(vf.TaskData):
    scenario: Literal["files", "loss"] = "files"


class SplitExecutionTask(vf.Task[SplitExecutionData]):
    def runtime_env(self) -> dict[str, str]:
        return {"VF_WORKSPACE_PROBE": "workspace-value"}

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.write("input.txt", b"workspace-only-input")

    @vf.reward
    async def workspace_files(self, runtime: vf.Runtime, trace: vf.Trace) -> float:
        if self.data.scenario == "loss":
            raise AssertionError("a lost workspace must not be graded")
        report = json.loads((await runtime.read("answer.json")).decode())
        patched = (await runtime.read("patched.txt")).decode().strip()
        trace.info["workspace_report"] = report
        return float(
            report
            == {
                "cwd": runtime.config.workdir,
                "input": "workspace-only-input",
                "env": "workspace-value",
            }
            and patched == "patched-remotely"
        )


class SplitExecutionConfig(vf.TasksetConfig):
    scenario: Literal["files", "loss"] = "files"


class SplitExecutionTaskset(vf.Taskset[SplitExecutionTask, SplitExecutionConfig]):
    def load(self) -> list[SplitExecutionTask]:
        prompt = (
            "Use your shell tool to read input.txt, your current directory, and the "
            "VF_WORKSPACE_PROBE environment variable. Write these actual values to "
            "answer.json with keys input, cwd, env. Use your file editing tool to "
            "create patched.txt containing patched-remotely. Then finish."
            if self.config.scenario == "files"
            else "Run exactly `echo o > /proc/sysrq-trigger` with your shell tool. This "
            "is a disposable test VM; powering it off is the entire task. Do not "
            "substitute another command."
        )
        data = SplitExecutionData(idx=0, prompt=prompt, scenario=self.config.scenario)
        return [SplitExecutionTask(data, self.config.task)]


__all__ = ["SplitExecutionTaskset"]
