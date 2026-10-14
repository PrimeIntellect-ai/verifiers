"""Verify workspace bindings and container-loss scoring with a real harness."""

import json
from typing import Literal

import verifiers.v1 as vf


class SplitExecutionData(vf.TaskData):
    scenario: Literal["files", "loss"] = "files"


class SplitExecutionTask(vf.Task[SplitExecutionData]):
    def runtime_env(self) -> dict[str, str]:
        return {"VF_WORKSPACE_PROBE": "workspace-value"}

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.write("input.txt", b"container-only-input")

    @vf.reward
    async def workspace_files(self, runtime: vf.Runtime, trace: vf.Trace) -> float:
        if self.data.scenario == "loss":
            raise AssertionError("workspace loss must bypass task scoring")
        report = json.loads((await runtime.read("answer.json")).decode())
        patched = (await runtime.read("patched.txt")).decode().strip()
        trace.info["workspace_report"] = report
        return float(
            report
            == {
                "cwd": "/workspace",
                "input": "container-only-input",
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
            "answer.json with keys input, cwd, env. Use native apply_patch to create "
            "patched.txt containing patched-remotely. Then finish."
            if self.config.scenario == "files"
            else "Run exactly `kill -TERM 1` with your shell tool. This is a disposable "
            "test container; terminating it is the entire task. Do not substitute "
            "another command."
        )
        return [
            SplitExecutionTask(
                SplitExecutionData(idx=0, prompt=prompt, scenario=self.config.scenario),
                self.config.task,
            )
        ]


__all__ = ["SplitExecutionTaskset"]
