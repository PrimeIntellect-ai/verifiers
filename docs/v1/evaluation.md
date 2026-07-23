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

[env.agent.runtime]
type = "docker"
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

## Skills

Harnesses whose program supports SKILL.md skills natively (e.g. Claude Code, Codex) take a `skills` list of local skill folders, each uploaded into the program's skill discovery directory in the agent's runtime as `<skills dir>/<folder name>`:

```toml
[env.agent.harness]
skills = ["path/to/my-skill"]
```

Setting `skills` on a harness without native skill support fails up front.

## Runtime network policies

Modal exposes a provider-native `network_access` switch. Prime and Docker use `allow`
and `block` lists after a trusted setup phase; Prime enforces host-level rules in the
platform, while Docker supports URL-level rules through a host-side proxy.

### Prime host policies

Prime VM sandboxes (`vm = true`) take either a host-level `allow` list or a `block`
list:

```toml
[env.agent.harness.runtime]
type = "prime"
vm = true
allow = ["*.wikipedia.org", "1.1.1.1"]
```

Entries are exact hostnames, leftmost-label `*.` wildcards, IPv4 addresses, or IPv4
CIDRs; schemes, ports, paths, and IPv6 are not supported. The default `allow = ["*"]`
keeps egress unrestricted. An empty `allow` list permits only the interception and MCP
route hosts, which Verifiers adds automatically before enforcement.

Setup stays online. Immediately before the agent starts, Verifiers replaces the
sandbox's policy and waits up to 60 seconds for the platform to report it applied; the
rollout fails closed if that acknowledgement never arrives. The provider policy governs
new connections and does not revoke connections already established during trusted
setup. Filtered Prime runtimes are therefore single-rollout.

Prime's API accepts only one effective policy mode: a concrete `allow` list cannot be
combined with `block`. A denylist cannot exempt framework hosts, so do not block an
interception or MCP route host.

### Docker URL policies

Docker harnesses can keep trusted setup online, then restrict the agent to declared
HTTP(S) destinations:

```toml
[env.agent.runtime]
type = "docker"
allow = ["https://*.wikipedia.org"]
block = ["https://upload.wikimedia.org"]
```

Docker defaults to unrestricted with `allow = ["*"]` and no block entries. An empty
`allow` list enables deny-by-default filtering and permits only the interception URL and
every MCP URL, which are added automatically before user entries. Adding a block entry
also enables filtering and narrows the wildcard. User block rules win over user allow
rules; framework interception and MCP routes always remain reachable. Under every
filtered policy, non-global destinations—including host-loopback, private, and link-local
addresses—are reserved for framework routes, so user `allow` rules cannot expose host/LAN
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
user deny rules win over user allows.

Per-task `TaskData.network_allow` and `TaskData.network_block` entries are merged into
Docker or Prime runtime lists. The task's default `network_allow=["*"]` is neutral and
leaves the evaluator policy intact. Docker combines concrete task/runtime lists and
retains every block entry. Prime requires `vm = true`, accepts host-level entries, and
rejects a task/runtime combination that would require both an allowlist and a blocklist.
Other runtimes reject non-neutral task network policies.

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
