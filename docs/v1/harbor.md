# Harbor

verifiers offers built-in support for Harbor via the `HarborTaskset` class. Creating a Harbor-based taskset is straightforward in most cases:


```python
import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset

# Set the dataset to the same name as registered in the Harbor registry
class TerminalBench2Config(HarborConfig):
    dataset: str = "terminal-bench/terminal-bench-2"


# The data will get loaded automatically
class TerminalBench2Taskset(HarborTaskset, vf.Taskset[HarborTask, TerminalBench2Config]):
    pass
```

You can also write custom code for your tasksets. A common customization is to set images for tasks that don’t come with one in their `task.toml`:

```python
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset

IMAGE_TEMPLATE = "registry.example.com/openthoughts/{task}:latest"


class OpenThoughtsTBLiteConfig(HarborConfig):
    dataset: Literal["openthoughts/openthoughts-tblite"] = "openthoughts/openthoughts-tblite"
    # Tell verifiers to use the pre-built image
    ignore_dockerfile: bool = True


class OpenThoughtsTBLiteTaskset(HarborTaskset, vf.Taskset[HarborTask, OpenThoughtsTBLiteConfig]):
    def load(self) -> list[HarborTask]:
        # Use the public image instead to avoid building the image at runtime; the row
        # data is frozen, so rebuild each task around an updated copy.
        return [
            HarborTask(
                task.data.model_copy(
                    update={"image": IMAGE_TEMPLATE.format(task=Path(task.data.task_dir).name)}
                ),
                task.config,
            )
            for task in super().load()
        ]
```

To create and reuse images for your tasksets, build the Dockerfile with Docker and push it to a registry, then set the resulting image reference as the task's `image` field.

## Additional features

Every Harbor taskset can also be modified with a `timeout_multiplier` and a `resource_multiplier`:

```toml
[taskset]
id = "MY_TASKSET"
timeout_multiplier = 2.0
resource_multiplier = 2.0
```

The `timeout_multiplier` multiplies both the agent and verifier timeout, while the `resource_multiplier` multiplies the task's CPU, memory and disk space. You might want to use these multipliers when the tasks set too tight limits and/or the agent is slow.

## Separate verifier runtimes

Harbor's separate verifier mode is selected directly from `task.toml`. After the agent finishes, Verifiers collects `/logs/artifacts` plus the task's configured `artifacts`, stops the agent runtime, starts a clean verifier runtime, restores those artifacts at their original absolute paths, and runs the verifier there.

```toml
artifacts = ["/workspace/final-output"]

[verifier]
environment_mode = "separate"
```

Without `[verifier.environment]`, the clean runtime reuses the resolved agent image and Verifiers uploads the trusted `tests/` directory. Tasks with `tests/Dockerfile` must instead publish a verifier image and set `[verifier.environment].docker_image`; Verifiers fails closed because it does not build Harbor Dockerfiles. The prebuilt image must contain `/tests/test.sh`. The verifier environment can independently declare its image, workdir, resources, environment variables, and `public` or `no-network` network access.

Only filesystem artifacts can cross this boundary. Artifact sources must be absolute paths, and their combined transfer is limited to 1 GiB uncompressed, 256 MiB compressed, and 100,000 entries. Runtime images must use a default user that can create the verifier-owned `/logs` and `/tests` paths and restore every declared artifact root. Compose sidecar artifacts, artifact destinations or exclude patterns, verifier healthchecks or MCP servers, accelerator constraints, allowlist networking, Windows verifier images, and multi-step tasks are not supported. A task whose verifier needs a live service, installed system state, or another undeclared agent-runtime side effect must remain shared or first export that state as an artifact.

## Shortcomings

verifiers does not have parity with Harbor yet, so some features are missing and currently being worked on. The most notable missing features right now are: 
- Complete network-policy support, including allowlists and shared-runtime phase switching ([Harbor Docs](https://www.harborframework.com/docs/tasks/network-policy))
- Multi-step tasks ([Harbor Docs](https://www.harborframework.com/docs/tasks/multi-step))
