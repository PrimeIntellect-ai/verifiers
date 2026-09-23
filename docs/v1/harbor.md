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

When loading an individual task directory in a custom taskset, pass `image` and/or `verifier_image` to `parse_task` to override that task's declared images:

```python
from pathlib import Path

from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask
from verifiers.v1.tasksets.harbor.taskset import parse_task

config = HarborConfig()
task_dir = Path("/path/to/task")
data = parse_task(
    task_dir,
    idx=0,
    harbor_config=config,
    image=f"registry.example.com/solver/{task_dir.name}:latest",
    verifier_image=f"registry.example.com/verifier/{task_dir.name}:latest",
)
task = HarborTask(data, config.task)
```

Both overrides apply before image validation, so supplying a pre-built image needs no `ignore_dockerfile` flag. Omitted overrides preserve the task's declared images. `verifier_image` applies only to a separate verifier and must contain the complete `/tests` suite, including `/tests/test.sh`. A verifier that uses a fresh solver environment inherits the overridden solver image unless `verifier_image` is supplied. The task's `task.toml` remains unchanged.

To create and reuse images for your tasks, build the Dockerfile with Docker, push it to a registry, and pass the resulting image reference to `parse_task`.

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

`--env.taskset.artifact-max-bytes` sets the total artifact archive budget per solver runtime (default: 32 MiB), including the `/logs/artifacts/` convention directory. Increase it for tasks that transfer trained checkpoints or VM disk files. The budget also applies when scoring is deferred to a separate verifier.

Prime VM bounded reads stream binary data. Collected archives remain in host memory for grading and are excluded from persisted traces.

`artifacts = [...]` and `[[verifier.collect]]` are read from `task.toml` ([Harbor Docs](https://www.harborframework.com/docs/run-jobs/results-and-artifacts)). Collect hooks run in the agent's box from the task's `finalize`, which is Harbor's own ordering — after the agent phase, before collection — and declared paths plus the `/logs/artifacts/` convention dir are then carried into the grading box and restored at their original paths ("no translation", as in Harbor).

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

- Image `ENTRYPOINT`s are replaced with a keepalive, so `[environment.healthcheck]` cannot depend on entrypoint-based setup or services
- Switching to a different verifier-phase network policy for a *shared* verifier ([Harbor Docs](https://www.harborframework.com/docs/tasks/network-policy)); a separate verifier's own policy is applied
- Building a verifier image from `tests/Dockerfile`, which Harbor does when a declared `[verifier.environment]` names no `docker_image`. A separate verifier image itself is supported — it just has to be pre-built and pullable (see above), because verifiers never builds images
- Sidecar services, and the sidecar artifacts and collect hooks that go with them ([Harbor Docs](https://www.harborframework.com/docs/tasks#sidecar-artifacts-and-collect-hooks))
- Multi-step tasks ([Harbor Docs](https://www.harborframework.com/docs/tasks/multi-step))
