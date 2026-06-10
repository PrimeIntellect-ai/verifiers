"""Built-in tasksets, resolved by id (`--taskset.id <id>`) as `tasksets.<id>`.

Re-exports each taskset's class + config off the package. `textarena_v1` is not re-exported
here — it imports the optional `textarena` dependency at module load — but still resolves as
`tasksets.textarena_v1`."""

from tasksets.harbor_v1 import HarborConfig, HarborTaskset

__all__ = ["HarborConfig", "HarborTaskset"]
