# Harbor Swarm

Run a Harbor source-code task with four solvers and one coordinator in a persistent Worlds repository. Each agent gets its own runtime. Worlds stores communication, branches, PRs, and commit-bound submission approvals; Verifiers owns agent execution and grading.

A fresh task runtime supplies the initial workspace. Agents exchange UTF-8 file changes through the forge API, so collaboration requires no Git executable in their sandboxes. Main is protected by independent PR review. Any participant can propose main for submission; all participants must approve the exact proposal, and only the coordinator requests acceptance. The controller runs the configured public check in a fresh task runtime before accepting it.

After solving, credentials are revoked and the accepted commit is captured. On deadline or when all agents finish without agreement, current main is captured with `accepted=false` and an explicit termination reason. Harbor's native isolated-verifier lifecycle restores that snapshot into a separate grading runtime and runs the task's original verifier once for the team. The first trace records the source snapshot and verifier metrics; every trace receives the same rewards. The world remains available in the local viewer.

## Run

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

For the native single-agent baseline, use `harbor-swarm-baseline` with the same task directory and workspace settings. It exports Harbor's single-agent environment and original task, without world orchestration. Keep model, harness, resources, and solving time comparable; record total token usage separately because five agents can consume more inference than one.

## Supported task contract

- A pullable agent image and a declared separate Harbor verifier.
- Exactly one unfiltered artifact directory matching `workspace`.
- A small UTF-8 source workspace, bounded by Worlds to 1,000 regular files and 2 MiB; no symlinks, binaries, or submodules.
- No collect hooks or task MCP services. The adapter rejects unsupported transfer semantics explicitly.
- `editable` identifies source paths allowed to differ from the pristine seed. Protected-file edits fail the public check and cannot be graded as a valid submission.
- Only the world host is added to the task's network allowlist. The isolated verifier receives no world credentials or agent-added network access.

FrontierSWE v2 requires its actual task package and pinned agent/verifier images, plus validation of its execution-user restrictions on the selected runtime. The public Git-to-Zig v1 task has an in-place verifier and is rejected. A coding fixture validates this adapter's integration; it does not establish a FrontierSWE result.
