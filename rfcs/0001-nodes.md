# RFC 0001: Nodes, one placement model for a rollout

- **Status:** implemented for a task's nodes and tool servers; the agent's side (split execution, a harness or user on a node) is still a proposal.
- **Builds on:** #2862, the runtime plugins and capability flags, and #2876, tool servers on Prime VMs of their own. Both are on main.
- **In this PR:** `SUPPORTS_PRIVATE_NETWORK` on the backend contract, with Prime's LAN behind it; `TaskData.nodes` and `ToolsetConfig.on`; and the planner (`verifiers/v1/runtimes/nodes.py`) wired into every rollout and `vf-validate`, which provision what it plans. See [In this PR](#in-this-pr).

## Summary

Code says *what* runs: the task, the harness, the tool servers, and a service's process. Config says *where*. A rollout's config declares named **nodes**. A node is a #2862 runtime with its own egress policy, and each part of the rollout names its node with `on`. From that, verifiers derives the rest:

- the machines to provision;
- which nodes share a machine;
- the connections between nodes, and how each is realized;
- the lifecycle;
- the refusals, made before anything boots.

Compose becomes an input format and one way of realizing nodes. It is no longer a mode the author picks.

The model replaces five things:

- `deployment.mode`;
- `Task.deployment_spec()` returning a `ComposeSpec`;
- `ToolsetConfig`'s `colocated`, `runtime` and `service`;
- split execution's `mode = "split"` (#2803);
- how each deployment strategy wires connections.

An env that declares nothing keeps today's rollout.

## Why

### "Where" lives in three places, with three vocabularies

| Knob | What it decides | Code or config |
| --- | --- | --- |
| `agent.runtime`, `agent.deployment.mode` | Where the workspace and the harness run | Config |
| `ToolsetConfig.colocated` / `runtime` / `service` / `url` | Where a tool server runs | Config, on the toolset |
| `Task.deployment_spec()` returns a `ComposeSpec` | What else runs, and on which machine | Code |

Who may reach whom is declared nowhere. Each strategy derives it separately: Compose networks, LAN rules (#2876), loopback, the tunnel, `expose`.

### Four needs, four mechanisms

| Need | Example | Mechanism today |
| --- | --- | --- |
| A tool backend needs egress the workspace must not have | MCP Atlas: its MCP servers call Wikipedia and OpenStreetMap, while the workspace stays offline | The tool server on a Prime runtime of its own, a VM on the rollout's LAN (#2876) |
| Agent-submitted code needs less than the workspace has | forth-lang: no egress, a separate kernel | The tool server on a Prime runtime of its own with `allow=[]` (#2876) |
| Backends must be reachable only through the tool server | EnterpriseOpsGym: the gyms sit on an internal network behind `tools` | A Compose service (#2872) |
| The agent loop and its credentials must stay out of the workspace | Split execution | A nested container (#2803) |

Prime refuses `expose`, so the private network is the only path between Prime machines. A tool server on a Prime runtime of its own runs as a VM on the rollout's private network, under its own policy (#2876). As Mika suggested, it keeps `runtime` and adds no class.

## Principles

1. **Code says what runs; config says where and how** (Mika). A task may configure placement but not write it.
2. **Egress belongs to a machine.** Prime enforces it per VM. Containers on one machine isolate processes, files and credentials, never egress.
3. **Nothing reaches the workspace unless the task declares it.** The agent's machine is the least trusted.
4. **Refuse before provisioning when a runtime lacks a capability,** as #2862 does, naming the runtime and the gap.
5. **Declaring nothing means today's rollout:** one workspace, the harness inside it, and tool servers on the evaluator.

## The model

### Nodes

```python
vf.TaskData(
    prompt=...,
    nodes={
        "atlas": vf.NodeConfig(
            runtime={
                "image": "prime/prime/mcp-atlas:1.2.5",
                "cpu": 4,
                "memory": 10,
                "allow": ["*"],
            },
            process=vf.ProcessConfig(
                command=["bash", "-lc", "<start atlas>"],
                readiness=["sh", "-c", "<list tools>"],
            ),
            reachable_from=["main"],  # the default
            isolated=False,  # the default
        ),
    },
)
```

- **`runtime`:** the fields that differ from the workspace's runtime. Everything else follows the workspace: labels, team, idle timeout. Setting `type` starts from that runtime's defaults instead. A node's `allow` / `block` replace the workspace's policy whole.
- **`process`:** optional. It is a long-running process besides any tool server placed on the node, with a readiness check. Its environment reaches only that node. Without a `command`, a node packed as a Compose service runs its image's own; a machine of its own needs one. Without a process, a node runs nothing of its own and idles.
- **`reachable_from`:** the nodes that may open connections to this one. The framework adds the connections its own parts need: the harness to its tools, and to where those tools execute.
- **`isolated`:** gives the node a machine of its own even when it could share one. forth-lang uses this for agent code.

**Two nodes always exist:**

- `main`, the workspace. It runs on the agent's runtime, resolved for the task by `resolve_runtime_config`. By default nothing may reach it.
- `host`, the evaluator. Nothing is provisioned for it; it keeps its own egress, and nothing executes agent commands there.

**Nodes per task row are data, not code.** EnterpriseOpsGym's gyms differ per row, so the taskset writes them into the row's data (`TaskData.nodes`). Later, config could override them, and `deployment_spec()` could go away.

### Placing the parts

| Part | Where it is placed | Default |
| --- | --- | --- |
| The harness's agent loop (model credentials, conversation) | `agent.deployment.harness_on` (proposed) | `main` |
| The harness's native tools (shell, files, kernels) | `agent.deployment.execution_on` (proposed) | Same as `harness_on` |
| A task's MCP server (`vf.Toolset`) | `ToolsetConfig.on` (in this PR) | `host` |
| A service | A node's `process` (in this PR) | none |
| A simulated user | `user.on` (proposed) | Where it runs today |
| Shared grading | `main`, switched to the task's `scoring.network` | Unchanged |
| Isolated grading | A fresh verifier node, after `main` stops | Unchanged |

The agent config may add nodes for its own parts, such as the harness's node in split execution. They are merged with the task's, and a name used by both is an error.

A tool server that only needs a node of its own keeps writing `runtime` on its toolset config, as Mika asked. `tools.runtime = PrimeConfig(...)` is shorthand for an implicit `isolated` node named after the server, with that runtime, plus `on` that node, beside a workspace without Compose services whose runtime has a private network. Elsewhere, the evaluator provisions the server's runtime at a published port, as before.

### What verifiers derives

1. **Connections.** For each node T and each S in T's `reachable_from`, S may connect to T. On top of that:
   - the harness connects to where its native tools execute;
   - the harness connects to every MCP server's node.
2. **Machines.** Nodes with the workspace's runtime and egress policy share the workspace's machine, unless one of them is `isolated`. Every other node gets a machine of its own.
   - A machine holding one node boots that node's image directly. Atlas needs no Docker.
   - The workspace's machine, holding several, runs them as containers: the task's Compose project, or one generated for its declared nodes. That needs `SUPPORTS_COMPOSE`; without it, they get a machine each on a private network. With neither, the plan is refused.
3. **How each connection is realized:**

   | Between | Realization | Requires |
   | --- | --- | --- |
   | The same node | Loopback | none |
   | Nodes on one machine | The task's Compose networks, or a container network per connection | `SUPPORTS_COMPOSE` |
   | Machines | A private network, with one rule per connection | `SUPPORTS_PRIVATE_NETWORK` on both |
   | A node to the evaluator | The tunnel, as the model endpoint uses today | none |
   | The evaluator to a node | `expose` | `SUPPORTS_PORT_EXPOSURE` |

4. **Lifecycle:**
   - The network and its rules exist before anything boots.
   - Machines boot in parallel, and processes pass their readiness checks.
   - A keepalive covers machines with idle timeouts.
   - `main` and the harness's execution stop before service evidence is collected.
   - The solver's machines are released before grading.
5. **Refusals,** made before anything is provisioned. For example: "nodes 'main' and 'atlas' run on separate machines, but the docker runtime cannot join machines on a private network; use a supported runtime".

### Today's rollouts in this model

Each of these is a case in `tests/unit/test_nodes.py`.

| Rollout | Nodes | Plan |
| --- | --- | --- |
| An ordinary task, with nothing declared | `main` | One machine |
| A search tool with an API key | The MCP server `on = "host"` | One machine; `main` reaches the evaluator through the tunnel; the key never leaves the evaluator |
| MCP Atlas | `main` (`allow=[]`), `atlas` (`allow=["*"]`) | Two machines on a private network, and no Docker |
| EnterpriseOpsGym (read from its Compose file) | `main`, `tools`, `calendar`, `gmail`, all `allow=[]`; gyms reachable only from `tools` | One machine, packed, with container networks. `main` cannot reach a gym. |
| forth-lang | `forth`, with `allow=[]` and `isolated` | Two machines. Without `isolated`, `forth` would share the workspace's machine. |
| Split execution | `harness_on = "harness"`, `execution_on = "main"` | One machine with the agent loop and the workspace in separate containers. VM to VM when `main` is `isolated`. |
| A harness on the evaluator | `harness_on = "host"` | Refused on Prime, which cannot expose a port. On Docker, an exposed port. |

## How it fits #2862

PR #2862 is the layer below: what a runtime is and what it can do. Nodes sit on top: which runtimes a rollout has and how they connect.

- **Runtimes:** a node's `runtime` is a #2862 `RuntimeConfig`, custom plugins included.
- **Capabilities:** the planner reads #2862's flags (`SUPPORTS_COMPOSE`, `SUPPORTS_PORT_EXPOSURE`, and network policy through `NetworkPolicyConfig`). This PR adds one: **`SUPPORTS_PRIVATE_NETWORK`**, with a backend method behind it:

  ```python
  @classmethod
  @asynccontextmanager
  async def private_network(
      cls,
      members: Mapping[str, RuntimeBackend],
      rules: Collection[tuple[str, str]],
  ) -> AsyncIterator[None]:
      """A network of `members`' own, on which each `(source, target)` rule lets the
      source open connections to the target, and nothing else may open any. Members
      are backends of this runtime that have not started; each joins as it starts and
      is reached at its `network_host`."""
  ```

  Members join as they start because Prime joins a VM to its network at create. Prime's implementation creates the LAN network with its rules and tags each member; a VM on it keeps itself inside its idle timeout, since LAN traffic is not activity to the platform. A plugin can implement the method too. The conformance suite (`tests/unit/test_runtimes.py`) boots two machines with one rule and checks that the connection opens one way only, and that a runtime that does not declare a private network refuses one. Tool servers on runtimes of their own (#2876) now go through this method.
- **Processes:** `exec`, #2862's one process primitive, carries a stdio transport between nodes on one machine.
- **Local runtimes:** Docker enforces a policy through its egress proxy and packs nodes through local Compose, which needs `trust_compose`. Neither Docker nor subprocess has a private network, so plans that need several machines run on Prime, and the local runtimes refuse them with a message.

## How split execution fits

Split execution *is* placement. A harness has two halves, and nodes say where each runs.

| The split stack today | In this model |
| --- | --- |
| `deployment.mode = "colocated"` | `harness_on = execution_on = "main"` |
| `deployment.mode = "split"`: the loop on the Prime VM, tools in a nested container (#2803) | `harness_on = "harness"`, `execution_on = "main"`, packed on one machine |
| A workspace that needs a VM's features (Docker inside, privileges) | `main` is `isolated`, so the halves are VM to VM over the private network |
| nano-rlm #218's transports: stdio, or authenticated TCP on a private network | stdio over `exec` within a machine; TCP across machines |
| Codex's executor in the workspace (#2850) | The same `execution` connection, with Codex's stdio launch |
| Harness capability checks before provisioning (#2803) | The harness declares the transports it supports; the planner refuses an `execution` connection none of them can use |

**Egress under split:** the loop's node and `main` normally share one policy, the task's plus the framework's routes. Containers keep the loop's credentials away from the workspace. The workspace can reach the model route but holds no secret for it. The internal-network swap in #2803 therefore goes, and allowlists work under split.

**Three mechanisms become one:** the loop driving its tools, the harness calling MCP servers, and tool servers calling services are all a connection between nodes. One planner and one set of realizations replace `deployment.mode`, the `ToolsetConfig` placement fields and the deployment spec.

## Prior art: Harbor's Agent Sandbox Protocol (harbor#3023)

ASP standardizes one edge, from an agent loop to the sandbox its tools run in. It leaves provisioning to the orchestrator. In this model that edge is the `execution` connection between `harness_on` and `execution_on`.

**We don't adopt ASP.** verifiers controls both sides of that edge: it provisions the machines, and it launches the harness. Our split harnesses already speak a tool-call protocol over stdio and TCP (nano-rlm #218, and Codex in #2850).

ASP and its review thread still record what the `execution` connection will need as it matures:

| From ASP and its review | What it means for the `execution` connection |
| --- | --- |
| The agent loop is placed independently of the task's environment (jjbuck) | `harness_on` is chosen independently of `main`'s runtime |
| The orchestrator provisions; the agent never does | Only the planner provisions; the harness receives a connection |
| Connection details reach the harness through a trusted path, not by discovering a file (jjbuck) | The connection's endpoint and credential go in the harness's launch config |
| Proxy tool calls rather than POSIX I/O; Prime has TCP between sandboxes but no SSH (Damian) | What nano-rlm #218 does: stdio within a machine, TCP across machines |
| Define how operations behave, not only how to connect: cancellation kills the remote process group, stdout and stderr stay separate, files are binary-safe (jjbuck, cauebraz) | Requirements for our worker protocol, worth a shared conformance test across harnesses |
| Revoke access before the verifier runs (jjbuck, cauebraz) | The lifecycle's order: stop execution and `main`'s processes, revoke the connection's credential, collect evidence, then grade in a fresh node |
| MCP servers are placement-sensitive (jjbuck, stbenjam) | `mcp_servers.<name>.on`, with reachability derived and checked before boot |
| An untrusted harness may belong inside the sandbox (kajogo777) | `harness_on = "main"` stays the default. Each run picks its threat model. |

**Where ASP could help later: harnesses we don't control.** Split execution needs the harness to cooperate, by sending its tool I/O elsewhere. nano-rlm and Codex do that through their own protocols. If a third-party harness adds ASP, verifiers could serve the sandbox side for it, with no per-harness work:

- **Provide an SSH server in `main`,** for example a static binary such as dropbear, so it doesn't depend on the task image.
- **Let the harness's node reach it** on the private network. Prime has TCP between sandboxes, which SSH runs over, though not an SSH gateway of its own.
- **Pass a generated `.asp.json`** in the harness's launch config.

pi already has an ASP extension, and Harbor plans terminus-slim on ASP. This would be one more transport for the `execution` connection, chosen when a harness declares ASP support, and the model needs no change for it. The cost is an SSH server in the workspace, which the agent can see (alexgshaw's contamination point). That's acceptable only for harnesses we can't split any other way. Untested.

## What it replaces

| Today | In this model |
| --- | --- |
| `agent.deployment.mode` (`colocated`, and `split` in #2803) | `agent.deployment.harness_on` / `execution_on` |
| `Task.deployment_spec()` returning a `ComposeSpec` | Nodes in task config or row data. A Compose file is an input format: `nodes_from_compose`. |
| `ToolsetConfig.runtime` on Prime (#2876) | The server on a node of its own, with that runtime's policy |
| `ToolsetConfig.colocated` / `runtime` / `service` | `task.tools.on` (`topology_from` shows the mapping) |
| `ToolsetConfig.url` | An MCP server whose source is a URL |
| Harbor's MCP proxy toolset (`tasksets/harbor/toolset.py`) | `mcp_servers` entries with a command or URL source |
| `_ColocatedDeployment`, `_ContainerDeployment`, `_PrivateNetworkDeployment` | Realizations a plan picks (in this PR); later, one deployment that realizes any plan |

## Compatibility

- **Envs that declare nothing:** a plan with one `main` node, the harness inside it and tool servers on the evaluator. That's today. In prime-envs, about 100 of the 108 envs define no toolset.
- **Python toolsets** (wikispeedia, browsecomp_plus, bfcl_v3, general_agent, ptx_remote): the class is unchanged. `colocated` and `service` stay as the older spellings of `on`. `runtime` keeps its #2876 meaning: a node of the server's own, named after it, so `deployment.target(name)` still reaches it.
- **Harbor (22 envs) stays a task adapter.**
  - An image becomes `main`'s image.
  - `environment/docker-compose.yaml` becomes nodes through `nodes_from_compose`, all under the task's policy, so they pack onto one machine as today.
  - `task.toml` MCP servers become `mcp_servers` sources.
  - `environment_mode = "separate"` is unchanged.
  - Harbor can't declare a per-service egress policy. A later Harbor field or a Compose extension (`x-vf-allow`) could add one.
- **Our three multi-node envs** (EnterpriseOpsGym, Atlas, forth-lang, in prime-envs) move to nodes.

## Order of work

On main:

- **#2862:** the runtime layer.
- **#2876:** the private network, which every multi-machine plan needs, and the model's first slice. A tool server on a Prime runtime of its own runs on a node of its own: `ToolsetConfig.runtime` is that node's runtime, with no new class. Atlas (prime-envs#999) and forth-lang (prime-envs#989) use it.

In this PR:

1. **`SUPPORTS_PRIVATE_NETWORK` and `private_network()`,** implemented by `PrimeBackend`, with the conformance test.
2. **The planner, wired in.** `TaskData.nodes` and `ToolsetConfig.on`. Every rollout and `vf-validate` builds its topology (`task_topology`), plans it before provisioning, and the plan picks the realization.

Next:

1. **Rebase the split stack (#2803 → #2831 → #2850)** as `harness_on` / `execution_on`, with nano-rlm #218's transports as the `execution` bindings. That brings the agent's side into the topology.
2. **One deployment for any plan,** replacing the three realizations. That covers a Compose project beside machines of its own, and declared nodes beside the task's own Compose project.
3. **Optional:**
   - an ASP transport, once a harness we want supports ASP;
   - capability-based packing beyond policy equality.

## Decisions taken in this PR

Each one is reversible. The alternative is noted.

1. **Names:** `on`, `nodes`, `NodeConfig` and `ProcessConfig`, and `reachable_from` on the target. Alternatives: `placement`, `hosts`, `reaches` on the source.
2. **Per-row nodes:** a `TaskData.nodes` field, defaulting to None so existing task hashes don't change. Alternative: a taskset hook.
3. **Packing:** only onto the workspace's machine. Two nodes that share a policy but not the workspace's get a machine each, which costs a VM but needs no Compose host on the private network. Alternative: pack them onto one machine of their own.
4. **Declared nodes on the workspace's machine:** a generated Compose project, one service per node, with a network of its own per service for egress and one per connection. Container networks are symmetric, so a target can also open connections to its source. Alternative: per-connection iptables rules on the host VM.
5. **A packed node's image:** its own container, the workspace's image unless it names one. Its `cpu` and `memory` are not applied; the workspace's machine keeps its own size.
6. **Tool servers with a runtime of their own:** a machine of their own named after the server, where #2876 placed them, so `deployment.target(name)` keeps working. Elsewhere they keep their published port.
7. **A machine of its own's name:** the workspace reaches it at its `network_host` (`<id>.sandbox.internal` on Prime), and task hooks look it up with `deployment.host(name)`. Alternative: an `/etc/hosts` alias in the workspace, so every node is reached by its name.
8. **Refused for now:** declared nodes beside the task's own Compose project, and a Compose project beside machines of their own (its host VM cannot join the private network yet).

## Open questions

1. **Resources on a packed machine.** Sum the nodes' `cpu` and `memory`, or take the largest? EnterpriseOpsGym sizes its host by hand today.
2. **Config overrides of a row's nodes,** for example to give one node another image in an eval config.
3. **Sharing across rollouts:** for example one instance of Atlas's stateless servers for many rollouts. Out of scope here.
4. **Policy on local Compose.** Docker packs nodes through local Compose, which enforces no network policy. To refuse a restricted packed plan on Docker, the planner needs that stated: a flag of its own, or policy enforcement as part of `SUPPORTS_COMPOSE`?
5. **Exposure on Prime.** A Prime VM can publish an HTTP port to the internet through a private-network rule from `internet` to its tag and port, with no authentication. That could back `SUPPORTS_PORT_EXPOSURE` for Prime VMs, and with it a harness or tool server reached from the evaluator. Tool servers would first need to authenticate their callers.

## In this PR

`verifiers/v1/runtimes/nodes.py`:

- the config models (`NodeConfig`, `ProcessConfig`, and `TopologyConfig`, which flattens a rollout's placements into one model);
- `topology_from()` / `task_topology()`, which build a rollout's topology from its Compose project, `TaskData.nodes` and its toolsets' placements;
- `plan()`, which returns machines and connections or raises `PlacementError`, from the capabilities the backends declare;
- `compose_project()`, the Compose project for declared nodes on the workspace's machine;
- `nodes_from_compose()`.

`runtimes/strategies.py` picks the realization from the plan and refuses what it cannot realize before provisioning. A machine of its own boots its node's runtime and starts its process. The private network carries one rule per planned connection, and a machine keeps its state channel only when it runs a tool server. The unit tests plan every rollout above from config. `tests/integration/test_nodes.py` runs a node on a machine of its own, and a node packed beside the workspace, live on Prime.
