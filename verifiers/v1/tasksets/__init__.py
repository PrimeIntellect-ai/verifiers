"""Built-in tasksets, resolved by id (`--taskset.id <id>`) as `verifiers.v1.tasksets.<id>`.

Re-exports each taskset's class + config off the package. `textarena` is not re-exported
here — it imports the optional `textarena` dependency at module load — but still resolves as
`verifiers.v1.tasksets.textarena`."""

from verifiers.v1.tasksets.harbor import HarborConfig, HarborTaskset
from verifiers.v1.tasksets.nemo_gym import NeMoGymConfig, NeMoGymTaskset

__all__ = [
    "HarborConfig",
    "HarborTaskset",
    "NeMoGymConfig",
    "NeMoGymTaskset",
]
