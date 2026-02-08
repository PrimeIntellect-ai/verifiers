# Prime CLI x Verifiers CLI Unification Design

Date: 2026-02-08

## Goal
Make `prime` the stable user-facing CLI while letting `verifiers` own command behavior, argument contracts, and environment orchestration so feature drift cannot happen as `verifiers` evolves.

## Scope
This design covers:
- `prime eval run`
- `prime gepa run` (new)
- `prime env build` (new)
- `prime lab setup` parity and interactivity
- shared environment resolution/install behavior for eval/gepa
- shared platform auth/config primitives used by verifiers

## Current Drift (Observed)
### `prime eval run` contract drift
- Prime mirrors eval options manually in `packages/prime/src/prime_cli/commands/evals.py` and `packages/prime/src/prime_cli/commands/env.py`.
- Prime forwards stale flags (`--save-every`, `--max-concurrent-generation`, `--max-concurrent-scoring`) that are not present in current `verifiers/cli/commands/eval.py` contract.
- Prime maps `--independent-scoring` to `-R`, but in current eval command `-R` is `--resume`.
- Prime does not expose newer eval options (`--resume`, `--tui`, `--debug`, `--max-retries`) and therefore cannot stay in sync without manual updates.
- Prime treats any slash-containing argument as a slug; this conflicts with config paths like `configs/eval/*.toml`.

### setup drift
- Prime `lab setup` calls `run_setup(..., vf_rl=...)`, but current setup module no longer accepts `vf_rl`.
- `prime lab setup` has no interactive flow for agent selection/multi-agent workspace scaffolding.

### duplicated install/platform logic
- Environment install and private-env build/cache flows are implemented in prime (`commands/env.py`) while verifiers has a separate, simpler install path (`verifiers/utils/install_utils.py`).
- Auth/config logic exists in prime (`prime_cli/core/config.py`, `prime_cli/core/client.py`) and separately in verifiers (`verifiers/utils/client_utils.py` partial fallback only).

### command export mismatch
- Prime imports selected verifiers internals directly and shells out to command-specific modules; there is no single integration contract.

## Target Architecture
### 1) Verifiers becomes CLI source-of-truth
Create a new internal package in verifiers:
- `verifiers/cli/`
  - `apps/` (command-level interfaces)
  - `workflows/` (preflight/plan/execute logic)
  - `orchestration.py` (generic multi-step runner)
  - `env_resolution.py` (local/remote/user-slug resolution)
  - `platform/` (auth/config/api primitives)
  - `plugins/prime.py` (prime-consumable exports)

Keep legacy command wrappers as compatibility layers that call into `verifiers/cli/*`.

### 2) Prime consumes verifiers plugins instead of mirroring flags
Prime should load verifiers-exported command modules and invoke them as subprocess modules from existing Typer groups.

Proposed plugin contract:
- `verifiers.cli.plugins.prime:get_plugin()`
- Returns a typed object with:
  - `api_version`
  - `eval_module`
  - `gepa_module`
  - `install_module`
  - `init_module`
  - `build_module`
  - `setup_module`
  - `build_module_command(module_name, args)`

Prime keeps platform-specific commands (`eval list/get/push/samples`, `env list/push/pull/status`, `login`, `config`) and injects host hooks.

### 3) Shared workflow pattern for multi-step commands
Introduce a generic execution model in verifiers:
- `prepare` phase (parse args/config, resolve env refs, build install/run plan)
- `run_pre_steps` phase (env install/pull/build steps)
- `run_atomic` phase (single-process eval/gepa run)
- `run_post_steps` phase (host hooks, e.g., Prime eval upload)

This gives one reusable pattern for `eval`, `gepa`, and future chained commands.

### 4) Platform auth/config primitives in verifiers
Add a minimal auth/config layer in verifiers that understands Prime conventions:
- env vars: `PRIME_API_KEY`, `PRIME_TEAM_ID`, `PRIME_USER_ID`, `PRIME_API_BASE_URL`, `PRIME_INFERENCE_URL`
- config file: `~/.prime/config.json`
- request headers: `Authorization: Bearer ...`, `X-Prime-Team-ID` when set

Add a small API client for verifiers workflows:
- `whoami()` for slug resolution
- environments hub detail fetch (`/environmentshub/{owner}/{name}/@{version}`)
- optional team lookup for future team-slug fallback

## Environment Resolution + Auto-Install Design
### Canonical behavior for eval/gepa
Users should not pre-install environments for `prime eval run` or `prime gepa run`.

For each requested `env_id`:
1. If explicit slug (`owner/name[@version]`): resolve remotely, ensure installed.
2. If ownerless (`name`):
   - Check installed module availability.
   - Check local path (`<env_dir_path>/<name_with_underscores>`).
   - If unresolved, resolve against current user slug (`<user_slug>/<name>`) and install.

Install strategy:
- local path: editable install
- public remote: wheel/simple-index install
- private remote: authenticated pull + local wheel build + cached install

Apply this logic consistently for:
- single env invocation
- config-file invocation (`[[eval]]` entries)
- GEPA env resolution

## Method Migration Map (Prime -> Verifiers)
Move reusable logic from prime into verifiers modules, keep UI/output in prime.

| Prime source method | New verifiers home | Notes |
| --- | --- | --- |
| `validate_env_id` | `verifiers/cli/platform/env_id.py` | shared parsing for `owner/name@version` |
| `is_valid_url`, `process_wheel_url` | `verifiers/cli/platform/url_utils.py` | shared validation |
| `normalize_package_name` | `verifiers/cli/platform/package_utils.py` | dedupe with existing verifiers util |
| `fetch_environment_details` | `verifiers/cli/platform/environments_hub.py` | authenticated and reusable |
| `get_install_command`, `_build_install_command` | `verifiers/cli/install/commands.py` | central install command generation |
| `execute_install_command` | `verifiers/cli/install/executor.py` | command execution policy |
| `_safe_tar_extract`, `_validate_path_component` | `verifiers/cli/install/security.py` | safe extraction + path checks |
| `_get_env_cache_dir`, `_get_version_from_pyproject`, `_pull_and_build_private_env` | `verifiers/cli/install/private_env.py` | private env pull/build/cache for all callers |
| `_is_environment_installed`, `_install_single_environment` | `verifiers/cli/install/resolve_and_install.py` | used by eval/gepa preflight |
| eval preflight from `run_eval` | `verifiers/cli/workflows/eval.py` | keep prime-only upload hook external |
| Prime config reading (`Config` subset) | `verifiers/cli/platform/auth.py` | minimal, no prime dependency |
| Prime API client header behavior (`APIClient` subset) | `verifiers/cli/platform/api_client.py` | minimal typed client |

Keep prime-only:
- rich UI/table rendering and command UX text
- login challenge flow and team-selection interaction
- eval list/get/push/samples APIs

## Command Ownership After Refactor
### Verifiers-owned command behavior
- argument schema for eval/gepa/build/setup
- env resolution/install policy
- workflow planning/execution

### Prime-owned behavior
- command grouping/navigation (`prime eval`, `prime env`, `prime lab`)
- platform post-run actions (eval upload, links)
- account/auth commands and context selection UX

## `prime lab setup` Interactive Flow
Add interactive mode (default when TTY and no explicit options):
1. Ask coding agent(s): `codex`, `claude`, `cursor`, `opencode` (multi-select)
2. Ask whether to scaffold all selected agent folders
3. Execute base lab setup
4. Create selected `.<agent>/skills/` folders

Keep:
- `--prime-rl` flag-only (do not prompt)

Add non-interactive controls:
- `--agents codex,cursor`
- `--no-interactive`

## Global `uv tool` Dev Workflow (Prime + Local Verifiers)
Recommended commands:
- For developing both repos together:
  - `uv tool install --force -e /Users/williambrown/dev/prime-cli/packages/prime --with-editable /Users/williambrown/dev/verifiers`
- For stable prime + local verifiers only:
  - `uv tool install --force prime --with-editable /Users/williambrown/dev/verifiers`

This ensures `prime` always invokes local editable verifiers without reinstall churn.

## Parallel PR Plan
### PR A: verifiers (foundational)
- Add `verifiers/cli/` core modules and plugin export.
- Move reusable install/auth/env-resolution logic from prime equivalents.
- Refactor existing scripts to wrappers.
- Add `eval` + `gepa` workflow planning/execution interfaces.

### PR B: prime-cli (integration)
- Replace manual eval wrapper flags with verifiers plugin registration.
- Add `prime gepa run` via verifiers plugin.
- Add `prime env build` via verifiers plugin.
- Rework `prime lab setup` to verifiers plugin + interactive prompts.
- Keep existing eval list/get/push/samples commands.

### PR C: cleanup + parity
- Remove dead/stale flags and compatibility shims.
- Update docs/help text in both repos.
- Add plugin API version checks and explicit compatibility errors.

## Testing Strategy
### Verifiers
- unit tests for env resolution order and install planning
- unit tests for private env authenticated pull/build/cache
- integration tests for workflow step chaining (`prepare` -> `install` -> `run`)

### Prime
- contract tests ensuring `prime eval run` options come from verifiers export
- integration tests for ownerless env resolution to user slug
- regression test for config path handling (`configs/eval/*.toml`)
- integration test for `prime gepa run` and `prime env build` registration

### Cross-repo contract checks
- smoke matrix in CI that installs prime from source with editable verifiers and runs:
  - `prime eval run --help`
  - `prime gepa run --help`
  - `prime env build --help`

## Resolved Decisions
1. Ownerless environment precedence is local-first.
2. Ownerless remote fallback checks both personal and active-team owners; when both match, prime prompts for owner choice.
3. Eval upload stays prime-owned for now (no migration into verifiers in this phase).
4. Prime command surfaces remain full Typer commands while forwarding verifiers args/contracts without mirroring.
5. Plugin API mismatches warn and continue with compatibility fallbacks.

## Key Insight for Clean Execution
The refactor should separate concerns into three layers:
- command contract (verifiers-owned)
- platform/runtime hooks (prime-owned)
- composition/orchestration (verifiers-owned, hook-aware)

If this boundary is enforced, future verifiers CLI evolution no longer requires prime flag rewrites, and prime remains the stable user entrypoint.
