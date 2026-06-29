from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.nemo_gym import (
    NeMoGymConfig,
    NeMoGymTask,
    NeMoGymTaskset,
)


class NemoGymWorkplaceConfig(NeMoGymConfig):
    nemo_env: Literal["workplace_assistant"] = "workplace_assistant"


class NemoGymWorkplaceTaskset(vf.Taskset[NeMoGymTask, NemoGymWorkplaceConfig]):
    def load_tasks(self) -> list[NeMoGymTask]:
        return NeMoGymTaskset(self.config).load_tasks()

    def tools(self, task: NeMoGymTask) -> list[vf.Toolset]:
        return NeMoGymTaskset(self.config).tools(task)
