"""Installed external taskset and harness fixture for loader regression tests."""

import verifiers.v1 as vf


class ExternalTasksetConfig(vf.TasksetConfig):
    custom_taskset_flag: bool = False


class ExternalData(vf.TaskData):
    pass


class ExternalTask(vf.Task[ExternalData]):
    pass


class ExternalTaskset(vf.Taskset[ExternalTask, ExternalTasksetConfig]):
    def load(self) -> list[ExternalTask]:
        data = ExternalData(idx=0, prompt="external plugin fixture")
        return [ExternalTask(data, self.config.task)]


class ExternalHarnessConfig(vf.HarnessConfig):
    custom_harness_flag: bool = False


class ExternalHarness(vf.Harness[ExternalHarnessConfig]):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls):
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")


__all__ = ["ExternalHarness", "ExternalTaskset"]
