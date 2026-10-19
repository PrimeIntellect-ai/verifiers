# Harbor

verifiers supports single-step Harbor 0.24.0 tasks through the `HarborTaskset` class. Install the `harbor` extra on Python 3.12 or later to get started:

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

For standard single-container tasks, verifiers requires a pullable container image rather than building Dockerfiles on the fly (Compose tasks can build service images inside the sandbox; see [Deployment and services](runtimes.md#deployment-and-services)). To prepare images for your tasks, build each Dockerfile with Docker, push it to a registry, and set the resulting image reference on the task data.

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

`verifier_image` applies to tasks that declare a separate verifier and must contain the complete `/tests` suite, including `/tests/test.sh`.

## Timeouts and resources

**By default, each task's declared agent and verifier timeouts are ignored** (`ignore_timeouts = true`): Harbor task timeouts are authored against Harbor's own runtime, so enforcing them directly can confound model capability with the speed of your inference stack. Set `ignore_timeouts = false` (or pass `--no-env.taskset.ignore-timeouts`) to enforce the task's agent and verifier time limits, for example when running a direct comparison against Harbor.

You can also scale task timeouts and resource requests across a Harbor taskset with `timeout_multiplier` and `resource_multiplier`:

```toml
[env.taskset]
id = "MY_TASKSET"
ignore_timeouts = false
timeout_multiplier = 2.0
resource_multiplier = 2.0
```

`timeout_multiplier` scales both the agent and verifier timeouts whenever `ignore_timeouts = false`, while `resource_multiplier` scales the task's CPU, memory, and disk requests regardless of the timeout setting. Both are useful when tasks specify tight resource bounds or when you are evaluating a slower model or harness.

## Network policies

Harbor's effective agent network policy is applied on Docker and Prime VM runtimes. An `[agent].network_mode` setting overrides the baseline policy in `[environment]`, and legacy `[environment].allow_internet` flags are normalized through Harbor's schema:

| Harbor mode | Task network setting |
| --- | --- |
| `public` | `network.allow = ["*"]`, leaving the evaluator runtime policy intact |
| `no-network` | `network.allow = []`, allowing only framework connections |
| `allowlist` | `network.allow = allowed_hosts` |

Task and harness setup runs online before network restrictions take effect. The policy is applied immediately before the agent launches and stays active through finalization and artifact collection, with model interception and MCP endpoints added automatically in allowlist and `no-network` modes.

Grading follows Harbor's verifier phase policy: `[verifier].network_mode`, or the `[environment]` baseline when the verifier declares none. A separate verifier runs under it. A shared-box verifier grades under it when it restricts the network and the agent phase is restricted too, since the box must start with a policy to take one; otherwise shared grading opens the network. A task's `scoring.network` replaces it.

When both the task and the runtime specify allowlists, only destinations permitted by both remain; blocklists combine across both, an empty allowlist (`no-network`) takes precedence, and nonempty allowlists cannot be combined with blocklists (see [runtimes](runtimes.md#network-policies) for runtime support).

Because provider-hosted web tools execute outside the sandbox, verifiers translates supported host allowlists into provider domain rules for OpenAI Responses web search and Anthropic web search/fetch, disabling those tools whenever a policy cannot be translated without widening access; all other hosted tools and provider-held resources remain disabled under network restrictions. This prevents [certain reward hacks](https://www.primeintellect.ai/blog/universal-offline-sandbox-escape).

## Task-declared MCP servers

When a Harbor task declares `mcp_servers` in `task.toml`, verifiers exposes them through a tool server inside the task runtime and passes an HTTP MCP endpoint to the harness (which requires a harness that supports MCP).

The bridge supports `stdio`, `sse`, and `streamable-http` servers, forwarding their tools and schemas automatically without any task-specific wrapper code. For `stdio` servers, make sure the server command and its dependencies are installed in the image or during task `setup`; URL-based entries connect to an existing service rather than launching one.

## Artifacts and collect hooks

`artifacts = [...]` and `[[verifier.collect]]` are read from `task.toml` ([Harbor artifact docs](https://www.harborframework.com/docs/run-jobs/results-and-artifacts)). Each entry's `service` selects the source runtime, defaulting to `main`, with additional services resolved from the deployment's Compose project. Missing Harbor artifacts are treated as optional during collection, so your test suite should fail if a required output file is absent (by contrast, native `vf.Artifact` declarations default to required).

Saving a trace writes its collected files alongside `traces.jsonl`, under `artifacts/<trace-id>/`; artifact bytes are excluded from the JSON. `destination` sets the saved path relative to that directory. Without it, files mirror their source paths with the leading `/` removed. For example, `{ source = "/app/report.json", destination = "report.json" }` saves `artifacts/<trace-id>/report.json` and restores `/app/report.json` for grading.

Each trace's artifact directory contains a `manifest.json` recording the source, service, destination, and status (`ok`, `empty`, `skipped`, or `failed`). Conflicting destinations keep the first saved artifact and skip the later one; source-mirrored directories may merge without overwriting files. Artifact roots from different services must not overlap, since they share the grader's filesystem.

One behavioral difference from `harbor run` is worth noting: **A failing collect hook fails the rollout.** Harbor logs a collection failure and continues because it treats collected files as observability artifacts; in verifiers, collected files are inputs to the grader, and ignoring a failed hook would let the verifier score stale or incomplete state.

## Separate verifier environments

Setting `[verifier].environment_mode = "separate"` grades the task in a fresh container that the agent never touched ([Harbor verifier docs](https://www.harborframework.com/docs/tasks/verifier)). The rollout lifecycle collects the solver's declared artifacts and `/logs/artifacts/` while its deployment is alive, stops the workspace before collecting sidecar evidence, tears down the solver deployment, and provisions a fresh grading target. It then restores the artifacts, stages the verifier tests, and records the resulting rewards and metrics onto the solver's trace. Setup, artifact restoration, and scoring retry in fresh targets according to `agent.scoring.retries` (defaulting to two retries) before failing the attempt.

Because verifiers never builds container images on the fly, a `[verifier.environment]` that relies on `tests/Dockerfile` without a pre-built `docker_image` must be built, pushed to a registry, and referenced via `docker_image` (or `verifier_image`). Setting `ignore_dockerfile = true` falls back to running the verifier inside the solver's image and logs a warning since that diverges from the task's declared verifier environment. Setting `ignore_separate_verifier = true` forces every task back into shared grading inside the solver's container, trading grading isolation for a single sandbox per task.

## Limitations

verifiers does not yet support every Harbor feature. The main current limitations are:

- Outside Compose, container `ENTRYPOINT`s are replaced with a keepalive process, so services required by the task or by `[environment.healthcheck]` must be started during setup rather than relying on the image entrypoint.
- A restricted shared-box verifier beside a public agent phase grades with open network: the box started without a policy. Use separate verification (`environment_mode = "separate"`) when such a grader needs its own policy or runtime.
- Verifier images defined via `tests/Dockerfile` must be built and published in advance.
- [Multi-step tasks](https://www.harborframework.com/docs/tasks/multi-step) are not supported.
