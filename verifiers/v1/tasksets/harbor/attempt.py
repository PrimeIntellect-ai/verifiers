"""Harbor service topology and verification, independent of agent strategy."""

from pathlib import Path

from verifiers.v1.configs.verifier import VerifierConfig
from verifiers.v1.task_attempt import TaskAttempt
from verifiers.v1.tasksets.harbor.compose import compose_services
from verifiers.v1.tasksets.harbor.taskset import HarborTask, verifier_box_data
from verifiers.v1.trace import Trace
from verifiers.v1.utils.compile import resolve_runtime_config


class HarborAttempt(TaskAttempt):
    task: HarborTask

    def __init__(self, task: HarborTask, **kwargs) -> None:
        super().__init__(task, **kwargs)
        if task.data.verifier is not None and self.verifier is None:
            self.verifier = VerifierConfig()
        self._stop_main = None

    async def provision(self) -> None:
        if (
            self._borrowed is not None
            or not (
                Path(self.task.data.task_dir) / "environment/docker-compose.yaml"
            ).is_file()
        ):
            await super().provision()
            return
        self.services, self._stop_main = await self._resources.enter_async_context(
            compose_services(
                resolve_runtime_config(self.placement, self.task),
                self.task,
                trust_compose=self.task.config.trust_compose,
                setup_timeout=self.timeouts.setup,
            )
        )
        declared = {
            entry.service
            for entry in (*self.task.data.collect, *self.task.data.artifacts)
        }
        if missing := declared - self.services.keys():
            raise ValueError(f"Unknown Compose services: {sorted(missing)}")

    async def finalize(self, trace: Trace) -> None:
        await self.task.finalize(trace, self.runtime)
        if self.verifier is not None and self._stop_main is not None:
            await self._stop_main()
            declared = dict.fromkeys(
                entry.service
                for entry in (*self.task.data.collect, *self.task.data.artifacts)
            )
            sidecars = {
                name: self.services[name] for name in declared if name != "main"
            }
            if sidecars:
                await self.task.finalize(trace, self.runtime, sidecars)

    @property
    def verifier_attempt_timeout(self) -> float | None:
        return self.timeouts.scoring

    def verifier_task(self) -> HarborTask:
        if self.task.data.verifier is None:
            return super().verifier_task()
        data = self.task.data
        if data.verifier_image is None and "main" in self.services:
            data = data.model_copy(
                update=self.services["main"].info.model_dump(
                    include={"image", "workdir"}
                )
            )
        return type(self.task)(verifier_box_data(data), self.task.config)
