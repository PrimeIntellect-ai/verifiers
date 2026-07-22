# Evaluation

To evaluate any taskset, use the `eval` entrypoint:

```bash
uv run eval primeintellect/terminal-bench-2
```

You can also use `.toml` files for configuration:

```toml
model = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"

[sampling]
temperature = 1.0

[env.taskset]
id = "primeintellect/terminal-bench-2"

[env.agent.harness]
id = "codex"
version = "0.116.0"
```

Validate the config by using `uv run eval @ config.toml --dry-run`. To run the evaluation, use `uv run eval @ config.toml`.

Use dotted arguments to set values using the CLI, e.g. `--sampling.temperature 0.5`. CLI arguments overwrite toml arguments when both are present.

The output from evaluations are written into `outputs/<env>--<model>--<harness>/<uuid>/` by default, where `<env>` is the taskset, prefixed by the paired env id when `--env.id` sets one (use `output_dir` to overwrite the folder). The folder contains the used `config.toml`, all the episodes in `traces.jsonl`, as well as logs of the run and workers in `eval.log`.

## Common config values

- `model` — the model id to evaluate, e.g. `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B`
- `sampling` — generation params passed to the model, e.g. `sampling.temperature`
- `env.taskset.id` — pick the taskset (or the positional `eval <taskset-id>`)
- `env.agent.harness.id` — pick the agent's harness (`[env.agent.harness]` in TOML)
- `num_tasks` — how many tasks to evaluate. Not setting a value means all tasks; an
  infinite taskset (a procedural generator, e.g. `wordle-v1`) requires it
- `num_rollouts` — rollouts per task
- `max_concurrent` — caps how many rollouts are in flight at once
- `verbose` — log at debug instead of info
- `shuffle` — randomizes the order of tasks (fixed seed); a no-op on an infinite taskset

## Resuming evaluations

`--resume <output-dir>` re-runs only the rollouts a previous run left missing or errored, appending to that run's own `traces.jsonl`. It reloads the run's saved `config.toml` verbatim, so it takes no other arguments. Good rollouts are kept, while errored ones are dropped and redone.

## Disabling tools

Almost every harness comes with a `disabled_tools` list, which can be used to disable one or multiple tools:

```toml
[env.agent.harness]
disabled_tools = ["shell_tool"]
```

The names of these tools are set by the respective harness. Consult the relevant documentation for the given harness for the relevant name(s). Some harnesses do not offer support to disable tools.

## Runtime network policies

Runtime configs expose `network_access`. Prime and Modal apply it as an all-or-nothing
sandbox setting at provisioning time. Docker additionally supports URL-level rules and a
trusted setup phase before enforcement; the same policy vocabulary can extend to other
runtimes when their sandbox APIs support it.

### Docker URL policies

Docker harnesses can keep trusted setup online, then restrict the agent to declared
HTTP(S) destinations:

```toml
[env.agent.harness.runtime]
type = "docker"
network_access = false
allow = ["https://*.wikipedia.org"]
block = ["https://upload.wikimedia.org"]
```

`network_access = false` is deny-by-default: the interception URL and every MCP URL are
added automatically, then `allow` adds user destinations. `block` can narrow that user
allowlist. With `network_access = true`, networking stays open unless `block` is
non-empty, in which case matching destinations are denied. User block rules win over
user allow rules; framework interception and MCP routes always remain reachable.
Host-loopback and link-local destinations are reserved for those framework routes, so
neither the default-allow posture nor a user `allow` rule exposes unrelated host
services or cloud metadata endpoints.

Filtered Docker runtimes are single-rollout. Reusing one would require reopening trusted
setup networking to processes left by the previous agent, so each rollout gets a fresh box.

Rules may be bare host patterns (`*.example.com`) or URL origins
(`https://example.com:8443`). A scheme or port in a rule narrows the match; URL paths
are ignored. `*.example.com` includes `example.com` itself. HTTPS proxy tunnels use
port 443 by default; an `allow` entry with an explicit HTTPS origin authorizes another
port. Both the CONNECT authority and the TLS ClientHello SNI must satisfy the policy.

The enforcement shape follows
[Docker Sandboxes network isolation](https://docs.docker.com/ai/sandboxes/security/isolation/):
HTTP(S) leaves through a policy proxy and direct non-HTTP egress is removed. As in
[Docker's policy evaluation](https://docs.docker.com/ai/sandboxes/governance/concepts/),
user deny rules win over user allows; `network_access` selects Verifiers' default allow
or deny posture.

Per-task `TaskData.network_allow` and `TaskData.network_block` entries are merged into
the Docker runtime lists. Task and evaluator restrictions compose: either may disable
public access, all allow and block entries are retained, and block entries win. Any
explicit task URL policy requires Docker's framework-aware policy support.

The restriction begins after task and harness setup and remains active through agent
execution, finalization, and scoring. Debug actions apply it after task setup as well.
The policy proxy runs in the Verifiers process, not a sidecar. Linux puts its listening
socket on container loopback and removes every non-loopback route. macOS keeps one route
to the host and limits it to the proxy port. One-shot helpers place the listener and
apply the cut, then exit, so the harness remains the only running container. This
prevents direct proxy bypass, peer-container access, arbitrary host access, and non-HTTP
egress.

Colocated MCP servers remain available on container loopback. The in-process proxy dials
host-local interception and MCP endpoints directly; shared and external MCP URLs pass
through the same policy without requiring a sidecar or public tunnel.
