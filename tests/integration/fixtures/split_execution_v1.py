"""Check that a split harness's tools act on the workspace, not the harness's VM."""

import json

import verifiers.v1 as vf


class SplitExecutionTask(vf.Task):
    def runtime_env(self) -> dict[str, str]:
        return {"VF_WORKSPACE_PROBE": "workspace-value"}

    async def setup(self, runtime: vf.Runtime) -> None:
        await runtime.write("input.txt", b"workspace-only-input")

    @vf.reward
    async def workspace_files(self, runtime: vf.Runtime, trace: vf.Trace) -> float:
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


class SplitExecutionTaskset(vf.Taskset[SplitExecutionTask, vf.TasksetConfig]):
    def load(self) -> list[SplitExecutionTask]:
        prompt = (
            "Use your shell tool to read input.txt, your current directory, and the "
            "VF_WORKSPACE_PROBE environment variable. Write these actual values to "
            "answer.json with keys input, cwd, env. Use your file editing tool to "
            "create patched.txt containing patched-remotely. Then finish."
        )
        return [SplitExecutionTask(vf.TaskData(idx=0, prompt=prompt), self.config.task)]


__all__ = ["SplitExecutionTaskset"]
