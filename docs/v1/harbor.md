# Harbor

verifiers offers built-in support for Harbor via the `HarborTaskset` class. Creating a harbor-based environment is straightforward in most cases:


```python
import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborData, HarborTask, HarborTaskset

# Set the dataset to the same name as registered in the Harbor registry
class TerminalBench2Config(HarborConfig):
    dataset: str = "terminal-bench/terminal-bench-2"


# The data will get loaded automatically
class TerminalBench2Taskset(HarborTaskset, vf.Taskset[HarborTask, TerminalBench2Config]):
    pass
    # No need for a reward function, as it will get inherited from the Harbor dataset
```

You can also write custom code for your environments. A common functionality is to set custom images for tasks that don’t come with an image in their `task.toml`:

```python
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborData, HarborTask, HarborTaskset

class OpenThoughtsTBLiteConfig(HarborConfig):
    dataset: Literal["openthoughts/openthoughts-tblite"] = "openthoughts/openthoughts-tblite"
    # Tell verifiers to ignore the Dockerfile in the task
    ignore_dockerfile: bool = True


class OpenThoughtsTBLiteTaskset(HarborTaskset, vf.Taskset[HarborTask, OpenThoughtsTBLiteConfig]):
    def load(self) -> list[HarborData]:
        # Use the public image instead to avoid building the image at runtime
        return [
            row.model_copy(update={"image": IMAGE_TEMPLATE.format(task=Path(row.task_dir).name)})
            for row in super().load()
        ]
```

To create & re-use images for your environments, build the Dockerfile with Docker and push it to a registry, then set the resulting image reference as the task's `image` field.

## Additional features

Every Harbor taskset can also be modified with a `timeout_multiplier` and a `resource_multiplier`:

```toml
[taskset]
id = "MY_TASKSET"
timeout_multiplier = 2.0
resource_multiplier = 2.0
```

The `timeout_multiplier` multiplies both the reagent and the verifier timeout, while the `resource_multiplier` multiplies the task's CPU, memory and disk space. You might want to use these multipliers when the tasks set too tight limits and/or the agent is slow.

## Shortcomings

verifiers does not have parity with Harbor yet, so some features are missing and currently being worked on. The most notable missing features right now are: 
- `no-network` support for sandbox runtimes ([Harbor Docs](https://www.harborframework.com/docs/tasks/network-policy))
- Shared & separate verifiers ([Harbor Docs](https://www.harborframework.com/docs/tasks#verifier-environment-shared-vs-separate))
- Multi-step tasks ([Harbor Docs](https://www.harborframework.com/docs/tasks/multi-step))
