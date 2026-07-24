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

## External candidate optimization (Weco)

[Weco](https://github.com/wecoai/weco-cli) optimizes a candidate artifact by rewriting it
and re-running an eval command. `weco-eval` is the verifiers-side adapter: one fixed v1
evaluation of the configured taskset + harness (it is not a harness itself) whose stdout
ends in a parseable `reward: <mean>` line. The candidate must be a *declarative* local
file the taskset or harness actually loads in each fresh evaluation process — a prompt,
template, or config, never Python the taskset imports (code candidates execute inside the
`weco-eval` process itself, which no output sealing or container around the harness can
make safe; supporting them would need a separate scoring trust domain). Use Weco's
`--sources` for a deliberately separated multi-file candidate surface — verifiers neither
receives nor manages Weco's source paths:

```bash
weco run --source <candidate-artifact> \
  --eval-command "uv run weco-eval <taskset-id> -n 20" \
  --metric reward --goal maximize \
  --steps 10 --eval-timeout 1800 --apply-change --output plain --no-open \
  --additional-instructions weco-instructions.md
```

Author `weco-instructions.md` yourself — what may change and what behavior must remain —
and pass the *path*: inline instruction text that starts with `-` breaks argument parsing,
and text that happens to equal an existing filename is read as one. Check the file exists
before launching (Weco silently treats a missing path as literal instruction text).
Instructions steer the optimizer, so they must come from you, never
from candidate or task output. `--metric` may
be the aggregate `reward` or one of the emitted `reward/<name>` / `metric/<name>`
components; the selected objective must stay fixed across the run. Any errored rollout
fails the step — `weco-eval` exits non-zero without metric lines rather than scoring a
partial eval (configure `--env.agent.retries.*` to absorb transient provider errors). The
flags after `--goal` keep headless runs bounded and noninteractive: an explicit
`--steps` budget (Weco defaults
to 100 steps — thousands of rollouts at `-n 20`), an `--eval-timeout` so a pathological
candidate can't hang the run forever, `--apply-change` to write the winner back without
the interactive confirmation `weco run` otherwise ends with, plain output, and no browser
tab.

### System-prompt convenience

When the candidate is the taskset's system prompt, `--system-prompt-path` overrides every
selected task's prompt with the file contents, and `--init-prompt` seeds the file from the
taskset's own baseline. Seeding scans the same task selection the eval scores
(`-n`/`--shuffle`; an infinite taskset requires `-n`) and is refused when those tasks carry
differing system prompts:

```bash
uv run weco-eval <taskset-id> \
  --system-prompt-path prompt.txt --init-prompt -n 20

weco run --source prompt.txt \
  --eval-command "uv run weco-eval <taskset-id> --system-prompt-path prompt.txt -n 20" \
  --metric reward --goal maximize \
  --steps 10 --eval-timeout 1800 --apply-change --output plain --no-open \
  --additional-instructions weco-instructions.md
```

Describing the task and its required output format in your instructions file keeps the
task's intent visible to the optimizer even after it has rewritten `prompt.txt` (Weco
already sees the current prompt as the source file). Each run snapshots the exact
evaluated prompt to `<run-dir>/system_prompt.txt` and points its saved `config.toml` at
that snapshot, so a run replays against the candidate it actually scored even after Weco
restores `prompt.txt`; an explicit `-o` gains a per-run leaf so successive candidates
never overwrite each other.

### Benchmark integrity

Weco may edit only the intended candidate surface. Keep outside `--source`/`--sources`: the
eval command and its config, `@reward`/`@metric` implementations, reference answers and
dataset selection, tests/validators/correctness gates, and held-out tasks or data. Keep
fixed across candidates: taskset configuration and seed, the selected tasks, harness, model
and sampling, rollout count, and the scoring implementation. Evaluate the winner on a
disjoint held-out split or selection — a higher optimization reward on the selected tasks is
not by itself an improvement, and candidate artifacts that define or generate tasks must
be optimized against a frozen task-quality metric with fixed validity gates, or Weco can raise reward by weakening
the task.

Keep held-out answers and scoring assets *inaccessible* to the candidate's runtime, not
merely unwritable — a tool-capable harness will read whatever files a candidate prompt asks
it to. Note that `--apply-change` trusts the file set Weco's service returns (its source
allowlist is not yet enforced client-side), and that seeded prompts containing Markdown
backticks can trip current weco-cli handling of uploaded content. Expect data egress: `weco run`
uploads the source contents, additional instructions, the evaluation command, each step's
evaluation output, and any `--api-key` provider keys you pass to the Weco service. The
`--source` list is not a security boundary on its own, so stick to declarative candidates;
if you ever experiment beyond them, run the entire Weco + evaluator stack inside an
isolated container or VM with scoped credentials and controlled data/network access — a
merely "disposable" directory still inherits your environment variables, credentials, and
filesystem, and in-process candidate code could tamper with scorers or data in memory
regardless of sandboxing.

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
[env.agent.runtime]
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
