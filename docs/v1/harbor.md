# Harbor

Use `HarborTaskset` to load tasks from Harbor. Install the `harbor` extra with
Python 3.12 or later:

```bash
uv sync --extra harbor
```

A wrapper only needs to choose the dataset:

```python
import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset


# Set the dataset to the same name as registered in the Harbor registry
class TerminalBench2Config(HarborConfig):
    dataset: str = "terminal-bench/terminal-bench-2"


# The data will get loaded automatically
class TerminalBench2Taskset(
    HarborTaskset, vf.Taskset[HarborTask, TerminalBench2Config]
):
    pass
```

`dataset` accepts a Harbor Hub ID such as `org/name@ref`. Pin a tag, revision, or
digest to keep the dataset fixed. Use `tasks` to select task names.

Export both the taskset and `HarborEnv` in the package's `__init__.py`:

```python
from verifiers.v1.tasksets.harbor import HarborEnv

from terminal_bench_2.taskset import TerminalBench2Taskset

__all__ = ["TerminalBench2Taskset", "HarborEnv"]
```

Install the package before using its taskset ID. Exporting `HarborEnv` is required
for Compose and separate verification; inheriting `HarborTaskset` does not select
that environment by itself.

## Task images

For ordinary container tasks, verifiers needs an image it can pull. Build the
task's Dockerfile, push the image to a registry, and set its reference in the task
data. Compose tasks can also build images; see [Docker Compose](#docker-compose).

To replace images while loading tasks, use `task.with_data(...)`. It copies the
task with the requested fields changed:

```python
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset

IMAGE_TEMPLATE = "registry.example.com/openthoughts/{task}:latest"
VERIFIER_IMAGE_TEMPLATE = "registry.example.com/openthoughts/{task}-verifier:latest"


class OpenThoughtsTBLiteConfig(HarborConfig):
    dataset: Literal["openthoughts/openthoughts-tblite"] = (
        "openthoughts/openthoughts-tblite"
    )
    # Load Dockerfile-only tasks before replacing their images below.
    ignore_dockerfile: bool = True


class OpenThoughtsTBLiteTaskset(
    HarborTaskset, vf.Taskset[HarborTask, OpenThoughtsTBLiteConfig]
):
    def load(self) -> list[HarborTask]:
        return [
            task.with_data(
                image=IMAGE_TEMPLATE.format(task=Path(task.data.task_dir).name),
                verifier_image=VERIFIER_IMAGE_TEMPLATE.format(
                    task=Path(task.data.task_dir).name
                ),
            )
            for task in super().load()
        ]
```

Only replace the fields you need. Set `ignore_dockerfile = true` when you will
supply an image for tasks that otherwise have only a Dockerfile.

`verifier_image` is used only for a separate verifier. It must include the full
`/tests` suite and `/tests/test.sh`. If it is `None`, the verifier uses the task's
`image` and receives the tests from the task package. These overrides do not
change `task.toml`.

Prime caches pullable images on first use. Preparing a new VM image can take
about ten minutes; the dashboard shows `build` and logs a warning. Later
sandboxes using the same image start much faster.

## Timeouts and resources

Harbor's task timeouts are ignored by default (`ignore_timeouts = true`). Set
`ignore_timeouts = false`, or pass `--no-env.taskset.ignore-timeouts`, to use the
task's agent and verifier time limits.

```toml
[env.taskset]
id = "MY_TASKSET"
ignore_timeouts = false
timeout_multiplier = 2.0
resource_multiplier = 2.0
```

`timeout_multiplier` scales both time limits when `ignore_timeouts = false`.
`resource_multiplier` scales CPU, memory, and disk requests regardless of the
timeout setting.

## Docker Compose

With `HarborEnv`, tasks containing `environment/docker-compose.yaml` can run on
local Docker, Prime VMs, or Modal VMs. Choose the runtime with
`--env.agent.runtime.type`.

| Runtime | Requirements |
| --- | --- |
| `docker` | Local Docker, unrestricted networking, and `--env.trust-compose` |
| `prime` | A Prime VM; its network policy covers every service after setup |
| `modal` | Access to Modal's experimental VM backend and `network_access = true` |

Only trust local Compose packages you have reviewed: their definitions can mount
host files and request Docker privileges. Local Compose receives Docker
connection settings, infrastructure variables, task variables, and task-local
`.env` files, rather than the evaluator's full environment.

Compose keeps the task's service commands, entrypoints, dependencies, health
checks, networks, and volumes. The agent runs in `main`. Host networking and GPU
Compose tasks are unsupported. Default runtime settings keep the authored image
and workdir; task settings and non-default runtime overrides take precedence.

For remote Compose, CPU and memory settings cover the whole VM, including
sidecars. Prime also applies the disk request; Modal has no disk-size setting.
Service images must be pullable by Docker inside the VM. A Prime VM image
reference cannot be used as a service image. Services with `build` are built
inside the VM.

Prime VM ports cannot be published externally. Modal exposes main's runtime
service port through an encrypted tunnel, even when main shares another service's
network namespace.

Set a task's `compose_host_image` to choose the VM image that runs Docker.
Docker is installed if missing. Archives made with `docker save` and placed in
`/opt/verifiers/compose-images/` are loaded before services start, so their image
tags can be used without a registry pull.

`HarborEnv` creates and removes the Compose project. An agent uses the existing
main container. Failed attempts retry with a new project through `--env.retries`.

Before separate verification, the entire project or remote VM is removed. The
verifier then runs in a fresh container. Without its own image, it uses main's
resolved image and workdir. If that image was built only inside the remote VM,
publish it separately and declare it as the verifier image.

## Network policies

Harbor's `[agent].network_mode` overrides the policy in `[environment]`.
Harbor also reads `[environment].allow_internet` through its schema.

| Harbor mode | Task network setting |
| --- | --- |
| `public` | `network_allow = ["*"]`; runtime restrictions still apply |
| `no-network` | `network_allow = []`; only framework connections are allowed |
| `allowlist` | `network_allow = allowed_hosts` |

Task and harness setup runs before restrictions start. Restrictions then remain
active through finalization and scoring. Model and MCP connections are added
when using an allowlist or framework-only access.

When both the task and runtime set allowlists, only destinations allowed by both
remain. Blocklists combine. An empty allowlist takes precedence. A nonempty
allowlist cannot be combined with a blocklist. See [runtimes](runtimes.md#network-policies)
for the runtimes that support each policy.

Docker keeps framework connections open even if a block rule matches them.
Prime applies ordinary block rules unchanged, so they may also block framework
connections. Prime accepts host-level entries. Provider-resolved URLs are kept
when their initial destination matches the policy.

Provider web tools run outside the sandbox. Verifiers translates supported host
allowlists for OpenAI Responses web search and Anthropic web search/fetch. It
disables tools when translation would allow broader access. Other hosted tools
and provider-held resources stay disabled under these restrictions.

## Task-declared MCP servers

Harbor's `mcp_servers` entries are exposed through a tool server inside the task
runtime. The harness receives an HTTP MCP endpoint and must support MCP.

The wrapper supports `stdio`, `sse`, and `streamable-http`. For `stdio`, install
the command and its dependencies in the image or during setup. URL entries
connect to an existing service; they do not launch it. The wrapper forwards the
service's tools and schemas without requiring task-specific tool code.

## Artifacts and collect hooks

Declare outputs in `artifacts = [...]` and `[[verifier.collect]]` in `task.toml`.
See the [Harbor artifact docs](https://www.harborframework.com/docs/run-jobs/results-and-artifacts).
Each entry's `service` chooses where to collect it; the default is `main`.

Main's hooks and artifacts are collected during `finalize`. For separate
verification, `HarborEnv` then stops main and collects sidecar outputs. A shared
verifier grades in main and skips sidecar entries. Files are restored at their
original paths, including main's `/logs/artifacts/` directory. Paths from
different services must not overlap in the verifier.

`--env.taskset.artifact-max-bytes` sets the total archive limit per solver across
all services. The default is 32 MiB, including `/logs/artifacts/`. Increase it
for large outputs such as checkpoints. Archives stay in evaluator memory for
grading and are not saved in traces. Prime VM reads stream binary data within
the configured limit.

Two differences from `harbor run` matter:

- A failed collect hook fails the rollout because the missing file may affect grading.
- `destination` is ignored. It controls Harbor's host output folder, not where files are restored for grading.

## Separate verifier environments

Set `[verifier].environment_mode = "separate"` to grade in a fresh container.
`HarborEnv` handles the following steps:

1. Run the solver and collect its declared artifacts and `/logs/artifacts/`.
2. Remove the solver's runtime.
3. Start a fresh verifier runtime, restore the artifacts, and prepare the tests.
4. Run the tests and add their rewards and metrics to the solver's trace.

The verifier uses the solver's resolved runtime settings unless overridden with
`--env.verifier.runtime.*`. Failures during setup, file restoration, test
preparation, or scoring retry according to `--env.verifier.retries`.

The score is read from `/logs/verifier/reward.json`:

- A finite number becomes the reward.
- An object with a `reward` key uses that value as the reward and the other values as metrics.
- An object without `reward` records each value as a separate reward.

All values must be finite numbers. If `reward.json` is missing or invalid,
verifiers tries `reward.txt`.

A declared `[verifier.environment].docker_image` must be pullable and contain the
complete `/tests` suite, including `/tests/test.sh` and its dependencies. Tests
from the task package are not uploaded to that image. Without a separate image,
a fresh copy of the solver image receives the task package's tests. Solver
artifacts under `/tests` are rejected, and old reward files are cleared.

verifiers does not build `tests/Dockerfile`. Build and publish that image, then
set `docker_image`. `ignore_dockerfile` instead uses the solver's image and logs a
warning because this changes the task's declared verifier environment.

A task that requires separate verification fails under an environment that does
not support it. Setting `ignore_separate_verifier = true` forces grading into the
solver's container and removes that isolation.

## Limitations

- Outside Compose, image entrypoints are replaced with a process that keeps the container alive. Start required services during setup; health checks cannot rely on the original entrypoint.
- A shared verifier cannot switch to a different [network policy](https://www.harborframework.com/docs/tasks/network-policy). A separate verifier can use its own policy.
- Verifier images must be built and published in advance.
- [Multi-step tasks](https://www.harborframework.com/docs/tasks/multi-step) are unsupported.
