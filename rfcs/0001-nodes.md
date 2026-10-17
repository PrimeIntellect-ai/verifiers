# RFC 0001: Nodes, one placement model for a rollout

- **Status:** draft for discussion.
- **Base:** #2862, the runtime plugins and capability flags.
- **Prototype:** this PR, `verifiers/v1/runtimes/nodes.py`. It plans from config and provisions nothing.

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
- the choice between `ComposeSpec` and a service-VM spec (#2876);
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
| `Task.deployment_spec()` returns a `ComposeSpec` (or a service-VM spec, #2876) | What else runs, and on which machine | Code |

Who may reach whom is declared nowhere. Each strategy derives it separately: Compose networks, LAN rules (#2876), loopback, the tunnel, `expose`.

### Four needs, four mechanisms

| Need | Example | Mechanism today |
| --- | --- | --- |
| A tool backend needs egress the workspace must not have | MCP Atlas: its MCP servers call Wikipedia and OpenStreetMap, while the workspace stays offline | A service VM on a LAN (#2876) |
| Agent-submitted code needs less than the workspace has | forth-lang: no egress, a separate kernel | A service VM with `allow=[]` (#2876) |
| Backends must be reachable only through the tool server | EnterpriseOpsGym: the gyms sit on an internal network behind `tools` | A Compose service (#2872) |
| The agent loop and its credentials must stay out of the workspace | Split execution | A nested container (#2803) |

A fifth mechanism, `ToolsetConfig.runtime`, can no longer reach anything on Prime: #2862 makes Prime refuse `expose`. The private network is the only path between Prime machines.

## Principles

1. **Code says what runs; config says where and how** (Mika). A task may configure placement but not write it.
2. **Egress belongs to a machine.** Prime enforces it per VM. Containers on one machine isolate processes, files and credentials, never egress.
3. **Nothing reaches the workspace unless the task declares it.** The agent's machine is the least trusted.
4. **Refuse before provisioning when a runtime lacks a capability,** as #2862 does, naming the runtime and the gap.
5. **Declaring nothing means today's rollout:** one workspace, the harness inside it, and tool servers on the evaluator.

## The model

### Nodes

```toml
[env.taskset.task.nodes.atlas]
runtime = { image = "prime/prime/mcp-atlas:1.2.5", cpu = 4, memory = 10, allow = ["*"] }
process = { command = ["bash", "-lc", "<start atlas>"], readiness = ["sh", "-c", "<list tools>"] }
reachable_from = ["main"]   # the default
isolated = false            # the default
```

- **`runtime`:** the fields that differ from the workspace's runtime. Everything else follows the workspace: labels, team, idle timeout. Setting `type` starts from that runtime's defaults instead. A node's `allow` / `block` replace the workspace's policy whole.
- **`process`:** optional. It is a long-running process besides any tool server placed on the node. Its environment reaches only that node.
- **`reachable_from`:** the nodes that may open connections to this one. The framework adds the connections its own parts need: the harness to its tools, and to where those tools execute.
- **`isolated`:** gives the node a machine of its own even when it could share one. forth-lang uses this for agent code.

**Two nodes always exist:**

- `main`, the workspace. It runs on the agent's runtime, resolved for the task by `resolve_runtime_config`. By default nothing may reach it.
- `host`, the evaluator. Nothing is provisioned for it; it keeps its own egress, and nothing executes agent commands there.

**Nodes per task row are data, not code.** EnterpriseOpsGym's gyms differ per row, so the taskset writes them into the row's data, and config can override them. `deployment_spec()` goes away.

### Placing the parts

| Part | Where it is placed | Default |
| --- | --- | --- |
| The harness's agent loop (model credentials, conversation) | `agent.deployment.harness_on` | `main` |
| The harness's native tools (shell, files, kernels) | `agent.deployment.execution_on` | Same as `harness_on` |
| An MCP server: a `vf.Toolset`, an external command, or a URL | `mcp_servers.<name>.on` (a task's toolset: `task.tools.on`) | `host` |
| A service | A node's `process` | none |
| A simulated user | `user.on` | Where it runs today |
| Isolated grading | A fresh verifier node, after `main` stops | Unchanged |

The agent config may add nodes for its own parts, such as the harness's node in split execution. They are merged with the task's, and a name used by both is an error.

### What verifiers derives

1. **Connections.** For each node T and each S in T's `reachable_from`, S may connect to T. On top of that:
   - the harness connects to where its native tools execute;
   - the harness connects to every MCP server's node.
2. **Machines.** Nodes with the same runtime type and egress policy share a machine, unless one of them is `isolated`.
   - A machine holding one node boots that node's image directly. Atlas needs no Docker.
   - A machine holding several runs them as containers. That needs `SUPPORTS_NESTED_CONTAINERS`; without it, they get a machine each on a private network. With neither, the plan is refused.
3. **How each connection is realized:**

   | Between | Realization | Requires |
   | --- | --- | --- |
   | The same node | Loopback | none |
   | Nodes on one machine | A container network per set of nodes that may reach each other | Nested containers |
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
- **Capabilities:** the planner reads #2862's flags (`SUPPORTS_NESTED_CONTAINERS`, `SUPPORTS_PORT_EXPOSURE`, and network policy through `NetworkPolicyConfig`). It adds one: **`SUPPORTS_PRIVATE_NETWORK`**, with a backend method behind it:

  ```python
  async def private_network(
      self, members: dict[str, RuntimeBackend], rules: list[tuple[str, str]]
  ) -> dict[str, str]:
      """Join `members` (this runtime's type) on a network on which each (source, target)
      rule may open connections and nothing else may; return each member's hostname."""
  ```

  #2876's LAN code (create the network, set the rules and wait until they apply, then create each member with its tags) becomes Prime's implementation. A plugin can implement the method too. #2862's conformance kit gets the matching test: a declared private network runs, and an undeclared one is refused.
- **Processes:** `exec`, #2862's one process primitive, carries a stdio transport between nodes on one machine.
- **What #2862 removed:** local Compose and Docker network policies. That leaves packing and private networks to Prime. The local debugging runtimes run one-node topologies, and refuse multi-node ones with a message.

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
| A service-VM spec (#2876) | A node with a different policy, or `isolated` |
| `ToolsetConfig.colocated` / `runtime` / `service` | `task.tools.on` (`topology_from` shows the mapping) |
| `ToolsetConfig.url` | An MCP server whose source is a URL |
| Harbor's MCP proxy toolset (`tasksets/harbor/toolset.py`) | `mcp_servers` entries with a command or URL source |
| `_ColocatedDeployment`, `_ContainerDeployment`, the service-VM strategy | One deployment that realizes a plan |

## Compatibility

- **Envs that declare nothing:** a plan with one `main` node, the harness inside it and tool servers on the evaluator. That's today. In prime-envs, about 100 of the 108 envs define no toolset.
- **Python toolsets** (wikispeedia, browsecomp_plus, bfcl_v3, general_agent, ptx_remote): the class is unchanged. `colocated`, `runtime` and `service` stay as deprecated aliases of `on` for a release.
- **Harbor (22 envs) stays a task adapter.**
  - An image becomes `main`'s image.
  - `environment/docker-compose.yaml` becomes nodes through `nodes_from_compose`, all under the task's policy, so they pack onto one machine as today.
  - `task.toml` MCP servers become `mcp_servers` sources.
  - `environment_mode = "separate"` is unchanged.
  - Harbor can't declare a per-service egress policy. A later Harbor field or a Compose extension (`x-vf-allow`) could add one.
- **Our three multi-node envs** (EnterpriseOpsGym, Atlas, forth-lang; all unmerged) move to nodes.

## Order of work

1. **#2862:** the runtime layer.
2. **#2876:** the private network, which every multi-machine plan needs. Whether its public service-VM class survives is open (Mika); this model doesn't need it.
3. **`SUPPORTS_PRIVATE_NETWORK` and `private_network()` on `PrimeBackend`,** with the conformance test.
4. **Wire the planner into `Deployment`,** replacing the strategies. Add `on` and nodes to config, and turn the toolset fields into aliases.
5. **Rebase the split stack (#2803 → #2831 → #2850)** as `harness_on` / `execution_on`, with nano-rlm #218's transports as the `execution` bindings.
6. **Optional:**
   - an ASP transport, once a harness we want supports ASP;
   - capability-based packing beyond policy equality.

## Open questions

1. **Resources on a packed machine.** Sum the nodes' `cpu` and `memory`, or take the largest? EnterpriseOpsGym sizes its host by hand today.
2. **Names:** `on` or `placement`; `nodes` or `hosts`; `mcp_servers` for the placed thing, with `vf.Toolset` kept for the class.
3. **Edges declared on the target or the source:** `reachable_from` on the target, or `reaches` on the source?
4. **Per-row nodes:** a `TaskData.nodes` field, or a taskset hook that returns them?
5. **A packed node's image:** its own container, or the machine's when it is the only node with an image?
6. **Sharing across rollouts:** for example one instance of Atlas's stateless servers for many rollouts. Out of scope here.

## The prototype in this PR

`verifiers/v1/runtimes/nodes.py` contains:

- the config models (`NodeConfig`, `ProcessConfig`, `TopologyConfig`);
- `plan()`, which returns machines and connections or raises `PlacementError`;
- `capabilities()`, read from the backend's flags;
- `topology_from()`, which turns today's toolset knobs and a Compose project into a topology;
- `nodes_from_compose()`.

`TopologyConfig` flattens the proposed config surface into one model: `mcp_servers` maps each server to its node. "Placing the parts" says where each field would be set. Nothing calls it; `Deployment` still uses its strategies. The tests plan every rollout above from config alone, along with the refusals. Planning Atlas on unmodified #2862 is refused, because no backend declares a private network yet. That is step 3 above.
