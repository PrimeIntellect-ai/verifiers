# Harbor Swarm

Run a Harbor source-code task with four solvers and one coordinator in a persistent Worlds repository. Each agent gets its own runtime. Worlds stores communication, branches, PRs, and commit-bound submission approvals; Verifiers owns agent execution and grading.

A fresh task runtime supplies the initial workspace. Agents exchange UTF-8 file changes through the forge API, so collaboration requires no Git executable in their sandboxes. Main is protected by independent PR review. Any participant can propose main for submission; all participants must approve the exact proposal, and only the coordinator requests acceptance. The controller runs the configured public check in a fresh task runtime before accepting it.

After solving, credentials are revoked and the accepted commit is captured. On deadline or when all agents finish without agreement, current main is captured with `accepted=false` and an explicit termination reason. Harbor's native isolated-verifier lifecycle restores that snapshot into a separate grading runtime and runs the task's original verifier once for the team. The first trace records the source snapshot and verifier metrics; every trace receives the same rewards. The world remains available in the local viewer.

At the coordination deadline, the controller captures main immediately and revokes participant access. A participant failure does not discard that artifact: grading still runs, while the failed traces and episode retain their error status. Configure each role's native `retries` policy to restart transient sandbox failures under the same world account. A replacement sandbox must recover committed work from the persistent world. The shared deadline remains unchanged.

## Run

SwarmTask delivers bounded Worlds inbox summaries at supported user-message and completed-tool boundaries, without changing the harness. Agents can also call `inbox` explicitly. Directed mentions, DMs, assignments, and PR reviews persist until explicitly acknowledged with `ack_inbox`; reading or delivering a summary does not acknowledge it. Delivery resumes from pending notifications when a participant restarts. Worlds owns the inbox; the runner owns when agents execute.

SwarmEnv enrolls participants in general before launching them and registers available short role aliases. The sandbox helper supports `--help` and `describe_tools` for authenticated schema discovery; omitted JSON defaults to `{}`, while `-` explicitly reads stdin. Agents can use Worlds issue claims, reviewer requests, PR supersession and merge queues. CI commands run inside their sandboxes, with agent attestations attached to exact merge candidates. Worlds stores signed attribution receipts and never executes repository tests on the host.

Install the package with `uv pip install -e environments/harbor_swarm`. Set `WORLDS_ADMIN_TOKEN` in the controller environment and supply an authenticated Worlds server. `agent-url` must reach that same server from the sandboxes.

```bash
uv run eval harbor-swarm \
  --env.taskset.task-dir /path/to/harbor/task \
  --env.taskset.workspace /app/zig-git \
  --env.taskset.editable 'src/*.zig' \
  --env.taskset.public-check 'zig build' \
  --env.world.agent-url https://your-world-tunnel.example \
  --env.review-timeout 3600 \
  --model internal/glm-5.3-fast -n 1 -r 1 --no-push
```

Run with `--dry-run` first to resolve configuration. The default roles use Prime runtimes and have no turn or token caps. Task-authored agent and verifier timeouts are retained; explicit agent timeout settings take precedence. `review-timeout` bounds coordination including public checks. Setup and teardown are separate from that deadline. Configure `--env.verifier.runtime.*` to place the independent verifier and `--env.agent.timeout.scoring` to override its task deadline.

For the native single-agent baseline, use `harbor-swarm-baseline` with the same task directory and workspace settings. It runs Harbor's original task and isolated grading, without world orchestration. Both variants retain bounded verifier logs and reward files in trace info. Keep model, harness, resources, and solving time comparable; record total token usage separately because five agents can consume more inference than one.

For experiments that score partial work at a fixed deadline, set `timeout.rollout_as_stop=true` on every solving role (the baseline uses `agent`). This marks the trace as truncated and runs artifact collection and grading. Other failures remain errors. Prime VM harness processes stop before collection. The swarm grades committed main at the coordination deadline, so agents should integrate changes throughout the run.

## Supported task contract

- A pullable agent image and a declared separate Harbor verifier.
- Exactly one artifact directory matching `workspace`; artifact exclusions also apply when capturing the pristine seed.
- A small UTF-8 source workspace, bounded by Worlds to 1,000 regular files and 2 MiB; no symlinks, binaries, or submodules.
- No collect hooks or task MCP services. The adapter rejects unsupported transfer semantics explicitly.
- `editable` identifies source paths allowed to differ from the pristine seed. Protected-file edits fail the public check and cannot be graded as a valid submission.
- Only the world host is added to the task's network allowlist. The isolated verifier receives no world credentials or agent-added network access.

An in-place Harbor task needs an explicit adapted manifest declaring a separate verifier and the source artifact to transfer. Preserve the original instructions and grader files, record their source revision and hashes, and validate grading in a fresh copy of the original image. Report adapted benchmark results with their solving budget and resource allocation.
