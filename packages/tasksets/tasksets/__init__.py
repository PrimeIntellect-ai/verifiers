"""Built-in tasksets, resolved by id (`--taskset.id <id>`) as `tasksets.<id>`.

Re-exports each taskset's class + config off the package. `textarena_v1` is not re-exported
here — it imports the optional `textarena` dependency at module load — but still resolves as
`tasksets.textarena_v1`."""

from tasksets.harbor_v1 import HarborConfig, HarborTaskset
from tasksets.multiswe_v1 import MultiSWEConfig, MultiSWETaskset
from tasksets.openswe_v1 import OpenSWEConfig, OpenSWETaskset
from tasksets.swebench_v1 import SWEBenchConfig, SWEBenchTaskset
from tasksets.swerebench_v2_v1 import SWERebenchV2Config, SWERebenchV2Taskset
from tasksets.swesmith_v1 import SWESmithConfig, SWESmithTaskset

__all__ = [
    "HarborConfig",
    "HarborTaskset",
    "MultiSWEConfig",
    "MultiSWETaskset",
    "OpenSWEConfig",
    "OpenSWETaskset",
    "SWEBenchConfig",
    "SWEBenchTaskset",
    "SWERebenchV2Config",
    "SWERebenchV2Taskset",
    "SWESmithConfig",
    "SWESmithTaskset",
]
