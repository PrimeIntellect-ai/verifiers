# Harbor

verifiers offers built-in support for Harbor via the `HarborTaskset` class. Creating a Harbor-based taskset is straightforward in most cases:

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

You can also write custom code for your tasksets. Override each task's `image` and `verifier_image` by rebuilding it around an updated copy of its data:

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
        # Task data is frozen, so rebuild each task around an updated copy.
        return [
            HarborTask(
                task.data.model_copy(
                    update={
                        "image": IMAGE_TEMPLATE.format(
                            task=Path(task.data.task_dir).name
                        ),
                        "verifier_image": VERIFIER_IMAGE_TEMPLATE.format(
                            task=Path(task.data.task_dir).name
                        ),
                    }
                ),
                task.config,
            )
            for task in super().load()
        ]
```

Only include the fields you want to replace. `verifier_image` applies to tasks that declare a separate verifier and must contain the complete `/tests` suite, including `/tests/test.sh`. When `verifier_image` is `None`, a separate verifier inherits the task's current `image` and stages the task package's tests. These changes leave `task.toml` unchanged.

To create and reuse images for your tasks, build the Dockerfile with Docker, push it to a registry, and set the resulting image reference in the task data.

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

## Docker Compose

Select `runtime.type = "docker"` to run Compose tasks locally.

With the default Harbor env, tasks containing `environment/docker-compose.yaml`
run their topology through Harbor on local Docker, Prime VMs, or Modal's VM runtime.
Local Docker requires `--env.trust-compose`: task definitions can mount host files
and request Docker privileges, so only enable it for trusted packages. Local Compose
receives Docker connection settings and infrastructure variables rather than the
evaluator's full environment. Task-local `.env` files and declared task env remain available.

Compose preserves service entrypoints, commands, dependencies, health checks,
networking, and volumes; the agent executes in a single `main` container.
Host networking is unsupported. Runtime defaults preserve the authored image and
working directory; task settings and non-default runtime overrides take precedence.

For Prime, set `runtime.type = "prime"`. One VM hosts Docker and all
services, and its network policy applies to every service after trusted setup.
For Modal, set `runtime.type = "modal"`; Compose uses the SDK's experimental VM
backend with Docker support and requires `network_access = true` and access to that
backend. Local Docker Compose also requires unrestricted networking. GPU Compose
tasks are unsupported.

The runtime's CPU and memory settings size the entire remote sandbox, so allow room
for sidecars. Prime also applies the disk request; Modal has no disk-size setting.
Prebuilt service images must be Docker-pullable inside the sandbox; services with a
`build` stanza are built there. Prime-only VM image references cannot serve as inner
container images. Prime VM ports cannot be published externally. Modal publishes
main's runtime service port through its encrypted tunnel, including when main shares
another service's network namespace.

A taskset can set a task's `compose_host_image` to a VM image that hosts the Docker
daemon instead of the stock one. Docker is installed only when the image lacks it, and
`docker save` archives shipped in `/opt/verifiers/compose-images/` load before the
services start, so services referencing their tags pull nothing from a registry.

The Harbor environment removes the entire project or remote sandbox before
separate grading, which retains the ordinary fresh verifier runtime. A verifier
without its own image inherits the resolved main image; a fresh copy also inherits
main's working directory. Images built only
inside a cloud host must be published separately and declared in the verifier environment.

Compose projects are owned by the Harbor environment; agents borrow the existing
Docker main container. Failures retry with a fresh project through
`--env.retries`, rather than retrying an agent inside the same project.

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
added automatically in allowlist and framework-only modes. Concrete task/runtime
allowlists retain their shared entries, while blocklists combine; framework-only access on
either side takes precedence, and concrete allowlists cannot be combined with blocklists.
Docker framework routes take precedence over deny rules, while ordinary Prime deny rules
are applied unchanged and may block a matching route. Restricted Harbor tasks require
Docker or a Prime VM; Prime accepts host-level entries. Provider-resolved URLs are retained
when their initial destination matches the effective policy. OpenAI Responses web search and
Anthropic web search/fetch receive wildcard host allowlists translated to provider domains;
policies that cannot be translated without widening still disable them. Every other hosted
tool and provider-held resource remains disabled.

## Artifacts and collect hooks

`--env.taskset.artifact-max-bytes` sets the total artifact archive budget per solver across all services (default: 32 MiB), including the `/logs/artifacts/` convention directory. Increase it for tasks that transfer trained checkpoints or VM disk files. The budget also applies when scoring is deferred to a separate verifier.

Prime VM bounded reads stream binary data. Collected archives remain in host memory for grading and are excluded from persisted traces.

`artifacts = [...]` and `[[verifier.collect]]` are read from `task.toml` ([Harbor Docs](https://www.harborframework.com/docs/run-jobs/results-and-artifacts)). Each entry's `service` selects the source runtime, defaulting to `main`; additional services come from the Harbor-owned Compose project. Main's hooks and artifacts are collected during `finalize`. For separate grading, the Harbor environment then stops main and collects sidecar evidence; a shared verifier grades in main and skips sidecar entries. Declared paths and main's `/logs/artifacts/` convention directory are restored at their original paths in the grader.

Artifact roots from different services must not overlap, since they share the grader's filesystem.

Two deliberate differences from `harbor run`:

- **A failing collect hook fails the rollout.** Harbor logs it and carries on, because there the output is observability; here it is a grading input, and a silently absent file makes the verifier score a stale state.
- **`destination` has no effect.** It positions a file in Harbor's host trial directory; verifiers has no trial directory (the trace is the record), and Harbor never lets `destination` affect verifier-side placement.

## Separate verifier environments

`[verifier].environment_mode = "separate"` grades in a second box the agent never touched, instead of the one it worked in ([Harbor Docs](https://www.harborframework.com/docs/tasks/verifier)). The harbor env — this taskset's default — grades such tasks in `finalize`: the solver plays the task as usual, its declared artifacts and the `/logs/artifacts/` convention directory are collected while its box is alive, the box is torn down, and the env then provisions a fresh box, restores those artifacts, prepares the verifier tests, and grades there, recording the verifier's rewards and metrics onto the solver's trace. The grading box derives from the solver's runtime policy unless `--env.verifier.runtime.*` names its own; setup, restoration, staging, and scoring failures retry per `--env.verifier.retries` before the episode fails. The score is read from `/logs/verifier/reward.json` — a finite number, or an object of finite numbers: with a `reward` key that key is the score and the rest are recorded as metrics; without one every key is recorded as a separate reward. Missing or invalid, it falls back to `reward.txt`.

Which image the verifier boots from follows Harbor: a declared `[verifier.environment]` if there is one, otherwise a fresh copy of `[environment]`, which is the task's own image. A dedicated `[verifier.environment].docker_image` must contain the complete `/tests` suite, including `/tests/test.sh` and its dependencies; that suite runs without uploading packaged tests, matching Harbor. When grading in a fresh copy of the solver image (including the `ignore_dockerfile` fallback), `/tests` is replaced with the task package's tests so stale image files cannot affect grading. Solver artifacts rooted under `/tests` are rejected before the tests run, and old reward files are cleared in either case.

A declared `[verifier.environment]` needs a pullable `docker_image`. Without one Harbor would build the verifier image from `tests/Dockerfile`, and verifiers never builds images — so build and push it yourself and name the resulting reference, exactly as for `[environment]`. `ignore_dockerfile` grades in the agent's image instead, which means the verifier runs somewhere the task never declared; it warns when it does.

Under any other env, a separate-verifier task refuses to grade in the agent's box rather than silently losing its isolation. `ignore_separate_verifier = true` forces every task back into shared grading, trading the isolation for one sandbox per task.

## Shortcomings

verifiers does not have parity with Harbor yet, so some features are missing and currently being worked on. The most notable missing features right now are:

- Outside Compose, image `ENTRYPOINT`s are replaced with a keepalive, so `[environment.healthcheck]` cannot depend on entrypoint-based setup or services
- Switching to a different verifier-phase network policy for a *shared* verifier ([Harbor Docs](https://www.harborframework.com/docs/tasks/network-policy)); a separate verifier's own policy is applied
- Building a verifier image from `tests/Dockerfile`, which Harbor does when a declared `[verifier.environment]` names no `docker_image`. A separate verifier image itself is supported — it just has to be pre-built and pullable (see above), because verifiers never builds images
- Multi-step tasks ([Harbor Docs](https://www.harborframework.com/docs/tasks/multi-step))
