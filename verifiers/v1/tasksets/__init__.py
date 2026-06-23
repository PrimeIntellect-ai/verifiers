"""Built-in tasksets, resolved by id (`--taskset.id <id>`) as `verifiers.v1.tasksets.<id>`.

Re-exports each taskset's class + config off the package. `textarena_v1` is not re-exported
here — it imports the optional `textarena` dependency at module load — but still resolves as
`verifiers.v1.tasksets.textarena_v1`."""

from verifiers.v1.tasksets.harbor_v1 import HarborConfig, HarborTaskset
from verifiers.v1.tasksets.tmax_v1 import TMaxConfig, TMaxTaskset

__all__ = ["HarborConfig", "HarborTaskset", "TMaxConfig", "TMaxTaskset"]
