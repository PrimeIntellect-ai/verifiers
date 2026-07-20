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

On the `prime` runtime any pullable image reference just works: the first sandbox to use an image makes the platform build and cache what it needs from it (for VM sandboxes this build can take ~10 minutes — the eval dashboard marks affected rollouts as `build` and a warning is logged); every later sandbox on the same reference starts in seconds.

## Additional features

By default, each task's declared agent and verifier timeouts are ignored (`ignore_timeouts = true`): Harbor task timeouts are authored against Harbor's own runtime, so enforcing them confounds model capability with the speed of your inference stack. Set `ignore_timeouts = false` (or pass `--no-taskset.ignore-timeouts`) to apply them, e.g. for a faithful comparison against the Harbor implementation.

With `ignore_timeouts = false`, every Harbor taskset can also be modified with a `timeout_multiplier`, and any Harbor taskset with a `resource_multiplier`:

```toml
[taskset]
id = "MY_TASKSET"
ignore_timeouts = false
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

[verifier.environment]
docker_image = "registry.example.com/my-benchmark-verifier:latest"
network_mode = "no-network"
```

Separate verification requires a prebuilt `[verifier.environment].docker_image` containing `/tests/test.sh`; Verifiers does not build `tests/Dockerfile`. The verifier environment can independently declare its workdir, resources, environment variables, and `public` or `no-network` network access. It is supported by the Docker and Prime runtimes.

Only filesystem artifacts can cross this boundary. Artifact sources must be absolute, non-root paths and either regular files or directories containing only regular files and directories. Symlinks and special files are rejected; nested empty directories and file mode bits are not retained. Transfer is limited to 256 MiB across 10,000 files and a 4 MiB path manifest. Runtime images must use a default user that can create `/logs` and restore every declared artifact root. Compose sidecar artifacts, artifact destinations or exclude patterns, verifier healthchecks or MCP servers, accelerator constraints, allowlist networking, Windows verifier images, and multi-step tasks are not supported. A task whose verifier needs a live service, installed system state, or another undeclared agent-runtime side effect must remain shared or first export that state as an artifact.

## Shortcomings

verifiers does not have parity with Harbor yet, so some features are missing and currently being worked on. The most notable missing features right now are:
- Complete network-policy support, including allowlists and shared-runtime phase switching ([Harbor Docs](https://www.harborframework.com/docs/tasks/network-policy))
- Multi-step tasks ([Harbor Docs](https://www.harborframework.com/docs/tasks/multi-step))
