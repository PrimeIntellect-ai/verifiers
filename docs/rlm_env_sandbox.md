# RLMEnv Sandbox Execution Backend Spec

## High‑level goals
- Add a sandbox execution backend for `RLMEnv` that supports both Bash and Python REPLs.
- Keep **local execution** as the default and preserve current behavior/performance.
- Keep **sub‑LLM calls and root tools local**, routed through a Prime Tunnel when the worker runs in a sandbox.
- Allow full filesystem context upload to sandboxes (zip/tar, upload, extract).
- Simplify sandbox execution where possible (no filesystem jail, no PTY‑simulated Bash).
- Keep rollout isolation (no overlap between rollouts) and persist environment variables + filesystem within a rollout.
- Preserve `retain_filesystem_after_rollout` semantics for both local and sandbox backends.

## Non‑goals
- Changing the RLM prompt scaffolding format or tool semantics.
- Replacing the existing local backend or its worker protocol.
- Rewriting the interception server or sub‑LLM tool loop.

## Architectural overview
`RLMEnv` stays a `StatefulToolEnv` and selects an execution backend:
- **Local backend** (existing): persistent local worker + FIFO I/O.
- **Sandbox backend** (new): persistent worker inside a Prime Sandbox.

Selection is done via a new `execution_backend` parameter:
- `execution_backend="local"` (default)
- `execution_backend="sandbox"`

To avoid inheriting `SandboxEnv` (which would eliminate local mode and introduce tool conflicts),
`RLMEnv` delegates to executor classes. The sandbox executor reuses sandbox patterns and errors
via a shared utility mixin in `verifiers/utils/sandbox_exec_utils.py`.

## Execution backend separation
Shared between backends:
- Tool split/deduplication (`tools`, `root_tools`, `sub_tools`)
- Interception server and sub‑LLM routing
- System prompt scaffolding and docs generation
- Trajectory handling and RLM metrics
- Stop conditions and answer extraction

Backend‑specific:
- Worker lifecycle and code execution transport
- Filesystem provisioning into the worker environment
- Removal of filesystem jail in sandbox mode
- Tunnel setup (sandbox only)

## New sandbox executor utility
Add `verifiers/utils/sandbox_exec_utils.py` containing a small mixin that imports from
`verifiers/envs/sandbox_env.py`:
- `ThreadedAsyncSandboxClient`
- `CreateSandboxRequest`
- `SandboxCreationError`
- `SandboxNotReadyError`
- `CommandTimeoutError` (from `prime_sandboxes`, same as `SandboxEnv`)

The mixin provides:
- client initialization + retry wrapper
- create / wait / execute / delete helpers
- teardown for the sandbox client

RLM’s sandbox executor uses this mixin; `RLMEnv` does not inherit `SandboxEnv`.

## Sandbox backend worker model
Sandbox worker is still a single Python process using FIFO IPC, but:
- **No filesystem jail** (sandbox provides isolation).
- **Bash execution** is done via `bash -lc` in a subprocess, no PTY.
- **Persistence across turns** is limited to:
  - `PWD`
  - `RLM_CONTENT`
  - `RLM_READY`
  - any explicit files on disk

State persistence is stored in a small JSON file (e.g., `rlm_env_state.json`) inside the sandbox.
Each Bash execution:
1. loads state, `cd` to `PWD`
2. exports `RLM_CONTENT`, `RLM_READY`
3. defines root tools as shell functions
4. runs user code
5. writes updated state back

Python REPL path stays close to the existing worker code (persistent namespace).

## Filesystem handling (sandbox)
Keep the local filesystem staging logic to generate metadata and prompt scaffolding, then:
- tar/zip the staged context directory
- upload to sandbox (see `harbor_env.py` upload pattern)
- extract into a sandbox working directory (e.g., `/tmp/rlm_fs/<rollout_id>`)
- set `rlm_fs_root` to the **sandbox path** in state for accuracy in prompts

Context size checks (symlink checks, size limits) remain in local staging.

## Interception + tunnel
For sandbox backend:
- Start the local interception server as usual.
- If `interception_url` is **not** provided, start a Prime Tunnel and use its URL.
- Set worker env vars:
  - `RLM_INTERCEPTION_URL`
  - `RLM_ROOT_TOOL_URL`

If `interception_url` is provided, the tunnel is skipped.

## Error handling (reuse where possible)
Reuse sandbox errors:
- `SandboxCreationError` for sandbox creation failures
- `SandboxNotReadyError` for wait‑for‑creation failures

Use existing `vf.SandboxError` for other sandbox‑API failures (upload, exec, I/O).
Keep `RLMCodeExecutionTimeout` for REPL execution timeouts and recovery logic.

## Init args / configuration changes
Add:
- `execution_backend: Literal["local","sandbox"] = "local"`
- `interception_url: str | None = None` (sandbox only; skips tunnel)
- `sandbox_*` fields (mirroring `SandboxEnv` defaults):
  - `sandbox_docker_image`, `sandbox_start_command`
  - `sandbox_cpu_cores`, `sandbox_memory_gb`, `sandbox_disk_size_gb`, `sandbox_gpu_count`
  - `sandbox_timeout_minutes`
  - `sandbox_environment_vars`
  - `sandbox_team_id`, `sandbox_advanced_configs`, `sandbox_labels`
  - `sandbox_client_max_workers`, `sandbox_client_max_connections`,
    `sandbox_client_max_keepalive_connections`

No existing args become fully redundant; however:
- `disallowed_modules` / `disallowed_builtins` are **ignored in sandbox** mode.
- `pip_install_packages` still applies in sandbox (install inside the worker).
- `retain_filesystem_after_rollout` applies to both local and sandbox.

## State fields (rollout‑level)
Add/ensure:
- `rlm_rollout_dir` (local staging dir)
- `rlm_fs_root` (path inside local or sandbox)
- `rlm_fs_source` (context_dir path, if any)
- `rlm_fs_metadata` (size/file count)
- `rlm_fs_has_data`
- `retain_filesystem_after_rollout`
- `sandbox_id` (sandbox backend only)
- `interception_base_url` (tunnel or override)

Reward functions can use these to clean up after judging (see `rlm_secrets`).

## Performance considerations
- Persistent worker per rollout avoids per‑turn startup costs.
- Single upload per rollout for filesystem context.
- No PTY/interactive shell overhead in sandbox.
- Optional sandbox reuse (future‑proofing) can reduce cold‑start overhead.

## Robustness
- FIFO protocol includes sequence numbers to detect desync (reused).
- Sandbox worker restart on timeout if configured.
- Tunnel failure surfaces as a sandbox error with explicit guidance.
- Cleanup always attempts to delete sandbox unless `retain_filesystem_after_rollout`.

## Testing updates
Add tests (mocking sandbox + tunnel):
- Sandbox backend setup uses tunnel unless override provided.
- Context upload path is invoked and sandbox `rlm_fs_root` is set.
- Bash worker script in sandbox mode is valid Python and omits jail.
- `retain_filesystem_after_rollout` keeps local staging and does not delete sandbox if configured.

## Migration / backward compatibility
- Default remains local execution; no behavior change unless `execution_backend="sandbox"`.
- Public API only adds optional parameters; existing code continues to work.
