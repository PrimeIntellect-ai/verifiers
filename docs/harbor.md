# Harbor

verifiers supports single-step Harbor 0.24.0 tasks through the `HarborTaskset` class (tasks declaring multi-step `steps` are rejected; use `tasks` to select single-step tasks from mixed datasets). Install the `harbor` extra on Python 3.12 or later to get started:

```bash
uv sync --extra harbor
```

Creating a Harbor-based taskset only requires pointing `HarborConfig` at a dataset:

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

`dataset` accepts a Harbor Hub ID such as `org/name@ref` (you can pin a tag, revision, or digest to keep the dataset fixed), or a bare `name@version` paired with `repo`, `registry_path`, or `registry_url` when loading from a custom registry. All of these settings live under `env.taskset`, including `tasks` for filtering specific task names.

Export the taskset in your package's `__init__.py` and install the package before running it by taskset ID:

```python
from terminal_bench_2.taskset import TerminalBench2Taskset

__all__ = ["TerminalBench2Taskset"]
```

Harbor is a task-format adapter: its tasks work directly under `SingleAgentEnv`, `BestOfNEnv`, `AgenticJudgeEnv`, and other standard environments.

## Task images

For standard single-container tasks, verifiers requires a pullable container image rather than building Dockerfiles on the fly (Compose tasks can build service images inside the sandbox; see [Docker Compose](#docker-compose)). To prepare images for your tasks, build each Dockerfile with Docker, push it to a registry, and set the resulting image reference on the task data.

You can override each task's `image` and `verifier_image` in `load()` using `task.with_data(...)`, which returns a copy of the task with only the specified fields updated:

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

Set `ignore_dockerfile = true` when loading tasks that only ship a Dockerfile so `super().load()` accepts them before you swap in your pre-built image references. These overrides only update the in-memory `TaskData` and leave `task.toml` untouched.

`verifier_image` applies to tasks that declare a separate verifier and must contain the complete `/tests` suite, including `/tests/test.sh`. When resolving a separate verifier's environment, Harbor prefers `[verifier.environment].docker_image`, followed by a dedicated `tests/Dockerfile` or `tests/docker-compose.yaml`, and finally falls back to the agent environment. A verifier section that only declares resource limits inherits the agent image and stages the task package's tests, and setting `verifier_image=None` likewise uses the task's `image` with the packaged tests staged in. Dedicated verifier builds without a pre-built image are rejected unless `ignore_dockerfile` explicitly enables fallback to the agent image.

On the `prime` runtime, any pullable image reference works out of the box: the first sandbox to use an image triggers a platform build and cache (for VM sandboxes this initial build can take around 10 minutes, during which the eval dashboard marks affected rollouts as `build` and logs a warning), and every subsequent sandbox using the same reference starts in seconds.

## Timeouts and resources

By default, each task's declared agent and verifier timeouts are ignored (`ignore_timeouts = true`): Harbor task timeouts are authored against Harbor's own runtime, so enforcing them directly can confound model capability with the speed of your inference stack. Set `ignore_timeouts = false` (or pass `--no-env.taskset.ignore-timeouts`) to enforce the task's agent and verifier time limits, for example when running a direct comparison against Harbor.

You can also scale task timeouts and resource requests across a Harbor taskset with `timeout_multiplier` and `resource_multiplier`:

```toml
[env.taskset]
id = "MY_TASKSET"
ignore_timeouts = false
timeout_multiplier = 2.0
resource_multiplier = 2.0
```

`timeout_multiplier` scales both the agent and verifier timeouts whenever `ignore_timeouts = false`, while `resource_multiplier` scales the task's CPU, memory, and disk requests regardless of the timeout setting. Both are useful when tasks specify tight resource bounds or when you are evaluating a slower model or harness.

If a Harbor task sets `[agent].user`, verifiers maps it to `TaskData.agent_user`; that account must already exist in the image with a writable home directory (see [execution users](runtimes.md#execution-users)). `[verifier].user` and collect-hook `user` settings are not supported.

## Rewards

When reading verifier output, `reward.json` takes precedence over `reward.txt`. If `reward.json` exists, it must contain a JSON object of finite numeric values; invalid, unreadable, or oversized JSON fails grading rather than silently falling back to `reward.txt`. When `reward.json` is absent, `reward.txt` supplies the scalar `reward` value. In either case, a nonzero exit code from the verifier script does not override a valid reward file.

By default, the `reward` key in `reward.json` supplies the primary scalar reward and any other named entries are recorded as trace metrics. You can select a different key as the primary reward in your task config:

```toml
[env.taskset.task]
reward_key = "accuracy"
```

If the selected key is missing from `reward.json`, grading fails (named values are never implicitly summed). The same reward resolution applies to both shared and separate verifiers.

## Docker Compose

Tasks that include `environment/docker-compose.yaml` declare a multi-container stack that the rollout's deployment provisions on local Docker (`--env.agent.runtime.type docker`), Prime VMs (`prime`), or Modal's experimental VM backend (`modal`). Docker Compose v2 (`docker compose`) is required on local Docker and provider VM hosts; standalone `docker-compose` v1 is not supported.

- **`docker`**: runs against local Docker and requires unrestricted networking along with `--env.agent.deployment.trust-compose`. Because Compose files can mount host files and request Docker privileges, only enable `--env.agent.deployment.trust-compose` for task packages you have reviewed. Local Compose receives Docker connection settings, infrastructure variables, declared task variables, and task-local `.env` files rather than the evaluator's full host environment.
- **`prime`**: runs a single Prime VM that hosts the Docker daemon and all Compose services, applying the task's network policy across every service once trusted setup finishes.
- **`modal`**: runs inside Modal's experimental VM backend with Docker support, which requires access to that backend and `network_access = true`.

Compose preserves service entrypoints, commands, dependencies, health checks, networks, and volumes, and runs the agent inside the `main` container (host networking and GPU Compose tasks are unsupported). Default runtime settings preserve the authored image and working directory, while explicit task settings and non-default runtime overrides take precedence.

On remote runtimes (`prime` and `modal`), the configured CPU and memory size the entire VM, so leave room for sidecar containers; Prime also applies the disk request, whereas Modal does not have a disk-size setting. Prebuilt service images must be Docker-pullable from inside the VM (a Prime VM image reference cannot be used as an inner service image), and services with a `build` stanza are built inside the VM. Prime VM ports cannot be published externally, while Modal exposes `main`'s runtime service port through an encrypted tunnel even when `main` shares another service's network namespace.

A taskset can also set `compose_host_image` on a task to choose a custom VM image for hosting the Docker daemon instead of the stock image. Docker is installed automatically when the image lacks it, and any `docker save` archives placed in `/opt/verifiers/compose-images/` are loaded before services start so services referencing those tags do not need to pull from a registry.

The rollout's deployment owns the Compose project and exposes `main` as its workspace handle (see [deployment and services](runtimes.md#deployment-and-services)). Agent rollout retries create a fresh deployment and Compose project, while isolated scoring retries reuse the collected submission in fresh verifier environments. For separate grading, the rollout stops `main` before collecting sidecar evidence, and the deployment tears down the Compose project and any host VM before provisioning the fresh verifier runtime. A verifier without its own dedicated image inherits `main`'s resolved image and working directory; if `main`'s image was built only inside a remote VM, publish it separately and declare it in the verifier environment so the fresh verifier runtime can pull it.

## Network policies

Harbor's effective agent network policy is applied on Docker and Prime VM runtimes. An `[agent].network_mode` setting overrides the baseline policy in `[environment]`, and legacy `[environment].allow_internet` flags are normalized through Harbor's schema:

| Harbor mode | Task network setting |
| --- | --- |
| `public` | `network_allow = ["*"]`, leaving the evaluator runtime policy intact |
| `no-network` | `network_allow = []`, allowing only framework connections |
| `allowlist` | `network_allow = allowed_hosts` |

Task and harness setup runs online before network restrictions take effect. The policy is applied immediately before the agent launches and stays active through finalization and artifact collection, with model interception and MCP endpoints added automatically in allowlist and `no-network` modes.

For shared grading, verifiers reopens network access before staging and running the tests (even when the agent runtime used an allowlist or blocklist) and does not switch to a separate `[verifier].network_mode`. For separate grading, verifiers provisions a fresh runtime and applies the verifier's network policy combined with the verifier runtime's policy (or the resolved verifier environment's baseline policy if no verifier-specific mode is set).

When both the task and the runtime specify allowlists, only destinations permitted by both remain; blocklists combine across both, an empty allowlist (`no-network`) takes precedence, and nonempty allowlists cannot be combined with blocklists (see [runtimes](runtimes.md#network-policies) for runtime support). Note that Docker keeps framework routes open even if a block rule matches them, whereas Prime applies block rules unchanged and accepts host-level entries. Provider-resolved URLs are kept when their initial destination matches the active policy. Because provider-hosted web tools execute outside the sandbox, verifiers translates supported host allowlists into provider domain rules for OpenAI Responses web search and Anthropic web search/fetch, disabling those tools whenever a policy cannot be translated without widening access; all other hosted tools and provider-held resources remain disabled under network restrictions.

## Task-declared MCP servers

When a Harbor task declares `mcp_servers` in `task.toml`, verifiers exposes them through a tool server inside the task runtime and passes an HTTP MCP endpoint to the harness (which requires a harness that supports MCP).

The bridge supports `stdio`, `sse`, and `streamable-http` servers, forwarding their tools and schemas automatically without any task-specific wrapper code. For `stdio` servers, make sure the server command and its dependencies are installed in the image or during task `setup`; URL-based entries connect to an existing service rather than launching one.

## Artifacts and collect hooks

`artifacts = [...]` and `[[verifier.collect]]` are read from `task.toml` ([Harbor artifact docs](https://www.harborframework.com/docs/run-jobs/results-and-artifacts)). Each entry's `service` selects the source runtime, defaulting to `main`, with additional services resolved from the deployment's Compose project. Missing Harbor artifacts are treated as optional during collection, so your test suite should fail if a required output file is absent (by contrast, native `vf.Artifact` declarations default to required).

`main`'s collect hooks and artifacts are gathered during task `finalize`, while the agent's harness session is still open, so they can observe any background service the agent left running (see [Finishing a rollout](architecture.md#finishing-a-rollout)). When running separate verification, the rollout then stops `main` through the deployment and collects any sidecar evidence, whereas a shared verifier grades inside `main` and skips sidecar entries. Declared paths and `main`'s `/logs/artifacts/` convention directory are restored at their original paths in the grading container, so artifact roots from different services must not overlap.

`--env.taskset.artifact-max-bytes` caps the total artifact archive size per solver across all services (default: 32 MiB, including `/logs/artifacts/`). Increase this limit for tasks that transfer large outputs such as model checkpoints or disk images. Prime VM reads stream binary data within the configured cap, and collected archives stay in evaluator memory for grading without being persisted in traces.

Two behavioral differences from `harbor run` are worth noting:

- **A failing collect hook fails the rollout.** Harbor logs a collection failure and continues because it treats collected files as observability artifacts; in verifiers, collected files are inputs to the grader, and ignoring a failed hook would let the verifier score stale or incomplete state.
- **`destination` has no effect.** In Harbor, `destination` controls where a file lands in the host trial directory; verifiers records everything in the trace rather than a host trial directory, and always restores files at their original source paths for grading.

## Separate verifier environments

Setting `[verifier].environment_mode = "separate"` grades the task in a fresh container that the agent never touched ([Harbor verifier docs](https://www.harborframework.com/docs/tasks/verifier)). The rollout lifecycle collects the solver's declared artifacts and `/logs/artifacts/` while its deployment is alive, stops the workspace before collecting sidecar evidence, tears down the solver deployment, and provisions a fresh grading target. It then restores the artifacts, stages the verifier tests, and records the resulting [rewards and metrics](#rewards) onto the solver's trace. Setup, artifact restoration, and scoring retry in fresh targets according to `agent.scoring.retries` (defaulting to two retries) before failing the attempt.

Image resolution follows Harbor: a dedicated `[verifier.environment].docker_image` must be pullable and already contain the complete `/tests` suite (including `/tests/test.sh` and its dependencies), which runs as-is without uploading packaged tests. When the verifier instead boots a fresh copy of the solver image (including under the `ignore_dockerfile` fallback), verifiers replaces `/tests` with the tests from the task package so stale files in the image cannot affect grading. In both cases, solver artifacts rooted under `/tests` are rejected and any pre-existing reward files are cleared before the tests run.

Because verifiers never builds container images on the fly, a `[verifier.environment]` that relies on `tests/Dockerfile` without a pre-built `docker_image` must be built, pushed to a registry, and referenced via `docker_image` (or `verifier_image`). Setting `ignore_dockerfile = true` falls back to running the verifier inside the solver's image and logs a warning since that diverges from the task's declared verifier environment. Setting `ignore_separate_verifier = true` forces every task back into shared grading inside the solver's container, trading grading isolation for a single sandbox per task.

## Verifier placement

Harbor's verifier mode becomes the task's default scoring placement (`TaskData.scoring.mode`). Leaving `env.agent.scoring.mode` unset honors the Harbor task's declaration, while setting `"shared"` or `"isolated"` explicitly overrides it (see [verifier placement](env.md#verifier-placement)). Fresh graders use the same image resolution and reward parser as shared graders and receive only declared artifacts, including collected sidecar evidence.

Harbor `[verifier.env]` supplies the task's default scoring environment variables. `agent.scoring.env` merges over those defaults and applies in both shared and isolated scoring (including agentic judging), with `${VAR}` templates resolved when scoring begins; entries set on `TaskData.scoring.env` take precedence over Harbor defaults.

## Limitations

verifiers does not yet support every Harbor feature. The main current limitations are:

- Outside Compose, container `ENTRYPOINT`s are replaced with a keepalive process, so services required by the task or by `[environment.healthcheck]` must be started during setup rather than relying on the image entrypoint.
- Shared grading reopens network access instead of switching to a separate verifier-phase network policy; use separate verification (`environment_mode = "separate"`) when the grader requires its own restricted network policy.
- Verifier images defined via `tests/Dockerfile` must be built and published in advance.
- [Multi-step tasks](https://www.harborframework.com/docs/tasks/multi-step) are not supported.
