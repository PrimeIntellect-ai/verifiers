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

By default, each task's declared agent and verifier timeouts are ignored (`ignore_timeouts = true`): Harbor task timeouts are authored against Harbor's own runtime, so enforcing them confounds model capability with the speed of your inference stack. Set `ignore_timeouts = false` (or pass `--no-env.taskset.ignore-timeouts`) to apply them, e.g. for a faithful comparison against the Harbor implementation.

With `ignore_timeouts = false`, every Harbor taskset can also be modified with a `timeout_multiplier`, and any Harbor taskset with a `resource_multiplier`:

```toml
[env.taskset]
id = "MY_TASKSET"
ignore_timeouts = false
timeout_multiplier = 2.0
resource_multiplier = 2.0
```

The `timeout_multiplier` multiplies both the agent and verifier timeout, while the `resource_multiplier` multiplies the task's CPU, memory and disk space. You might want to use these multipliers when the tasks set too tight limits and/or the agent is slow.

## Separate verifier runtimes

Harbor's separate verifier mode is selected from `task.toml`. Docker and Prime VM runtimes collect declared artifacts, confirm that the agent runtime is gone, then grade in a fresh runtime on the same provider. Explicit verifier environments need a pullable image unless `ignore_dockerfile = true`.

## Network policies

Harbor's effective agent network policy is applied to Docker or Prime VM harness
runtimes. An `[agent].network_mode` override takes precedence over the `[environment]`
baseline; legacy `[environment].allow_internet` is normalized by Harbor's schema.

| Harbor mode | Task network policy |
| --- | --- |
| `public` | Sets the task allowlist to `["*"]`, leaving the evaluator policy intact. |
| `no-network` | Sets the task allowlist to `[]` (framework routes only). |
| `allowlist` | Sets the task allowlist to `allowed_hosts`. |

Trusted task and harness setup remains online. The policy starts immediately before the
agent and stays active through finalization and scoring. Interception and MCP URLs are
added automatically; evaluator-provided `allow` entries add exceptions and `block`
entries can narrow them. Restricted Harbor tasks require Docker or a Prime VM; Prime
accepts host-level entries and rejects combinations that need both policy modes.

## Shortcomings

verifiers does not have complete Harbor parity yet. The main missing features are verifier network allowlists, shared-runtime phase switching, compose sidecars, and multi-step tasks.
