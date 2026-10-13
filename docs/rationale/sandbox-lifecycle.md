# Rationale: sandbox, tunnel and image lifecycle

Covers P7 (sweep labelled sandboxes at launch), V18 (tunnel and sandbox cleanup by run
label) and V17 (image build, push, cache by digest). Three branches, one per repo:

| Repo | Branch | Commit | What |
| --- | --- | --- | --- |
| verifiers | `feat/sandbox-lifecycle` (on `c17b296f`) | `1e63922d` | tunnel labels, `runtimes.sweep`, `runtimes.prime_images` |
| prime-rl | `feat/sandbox-sweep` (on `c3ef3b1f`) | `affdb66d` | sweep the run name when `rl`/`eval` resume |
| flow | `feat/pipeline-sweep` (on `29de8a1`) | `929a1d0` | sweep the root's label before a remote launch opens anything |

## Problem

A run that dies without its teardown (SIGKILL, OOM, host reboot, a hung drain killed by
hand) leaves three kinds of Prime resources behind:

- Sandboxes, which bill until their idle timeout. The default is 1 h, but long seats
  raise it: hillclimb runs most roles with `DURABLE_LIFETIME_TIMEOUT_SECONDS` and the
  solver and judge with 86,400 s (`hillclimb/agents/episode.py:75-91,122`). A crash there
  bills for up to a day per box.
- Tunnels, which count against the team's tunnel cap (2,000, shared by every user) until
  they expire. verifiers opens one per interception server with no labels, and its stop is
  one 5 s DELETE with errors ignored (`prime_tunnel.Tunnel.sync_stop`).
- Images: each pipeline that builds Dockerfiles into the Prime registry shells out to the
  `prime images` CLI and polls its JSON list output.

## Evidence

- hillclimb sweeps sandboxes at launch and on `hillclimb teardown`: a stable per-host,
  per-data-dir label (`launch.teardown_label`), passed to the env server as
  `HILLCLIMB_SANDBOX_LABEL` and stamped with `set_base_sandbox_labels`
  (`hillclimb/env.py:217-219`), swept by `launch.sweep_sandboxes` (bulk delete per label,
  failures logged as the manual command) before `os.execve` into prime-rl `eval`
  (`hillclimb/launch.py:65-93,165-168`, `hillclimb/cli.py:287-295`). It never sweeps
  tunnels.
- repoforge (flow PR #25) monkeypatches `prime_tunnel.Tunnel` with a subclass that adds
  `$VF_RUN_ID` as a label and retries the delete, and sweeps the flow label at pipeline
  setup and in `python -m repoforge sweep <run>` (`pipelines/repoforge/repoforge/tunnels.py`,
  `pipeline.py:107-108`, `worlds.py:83-84`). Its operator notes: "verifiers tunnel stop is a
  single 5 s DELETE with errors swallowed; label and sweep tunnels" (line 27), "Leftover
  tunnels swept (60 runs had some)" (line 286), team tunnels at 1,180/2,000 (line 60).
- Image builds, done twice the same way:
  - hillclimb `scoring/images.py` (`CandidateImages`): tag `sha-<sha256(salt+Dockerfile)[:24]>`,
    reuse when the registry lists the VM_SANDBOX artifact COMPLETED, otherwise `prime images
    push` and poll `prime images list --output json` until COMPLETED, FAILED or a 2,400 s
    deadline; delete with `prime images delete`. It depends on the host CLI version
    (`launch.py:189-192` puts the venv last on PATH because the venv's CLI differs).
  - repoforge `world.py:314-370`: tag = sha256 over pin, rendered Dockerfile and override;
    reuse if listed, else `prime images push`, then poll until COMPLETED or failed.

## What already existed

- verifiers labels every Prime sandbox with the process-wide base labels
  (`set_base_sandbox_labels`), the runtime's `PrimeConfig.labels` and the run scope
  (`$VF_RUN_ID`) (`runtimes/prime.py:137-143,272`). Docker containers carry
  `verifiers.run=<scope>`. Tunnels carried nothing. No sweep existed anywhere in verifiers.
- prime-rl stamps `$PRL_RUN_NAME` as a base label in env servers and the orchestrator
  (`entrypoints/env_server.py:19-21,36-37`, `orchestrator/orchestrator.py:214-216`).
  `$PRL_RUN_ID` (= `$VF_RUN_ID`) is minted fresh per launch, so it is useless for finding a
  previous attempt's boxes. prime-rl never sweeps.
- flow sets `$VF_RUN_ID` to `Flow.label`, a digest of host and root, under the root's
  `flow.lock` (`flow.py:181,214-240`). Every box a flow opens therefore already carries a
  label that is stable across relaunches and owned by exactly one live launch. flow never
  sweeps.
- The Prime SDK has everything needed: `AsyncSandboxClient.bulk_delete(labels=...)`
  (matches sandboxes with ALL the labels, so one call per label), `TunnelClient`
  `list_tunnels_page(labels=...)` and `delete_tunnel`, `prime_tunnel.Tunnel(labels=...)`,
  and `AsyncImageClient` (`list`, `initiate_build`, `start_build`). Image delete is not in
  the SDK; it is `DELETE /images/{name}/{tag}`, as the CLI does.

## Why these layers

- **Labels and the sweep belong in verifiers.** verifiers opens the sandboxes and tunnels
  and decides their labels; the sweep is the inverse of that and must agree with it. It is
  plain Prime API, with no pipeline knowledge. The tunnel label fix can only live there:
  repoforge's monkeypatch of `prime_tunnel.Tunnel` exists because verifiers constructs the
  tunnel.
- **Who sweeps which label belongs to the launcher that owns the label**, because a sweep
  cannot tell a leftover from a box in use. Only a launcher knows that no other live run
  carries its label:
  - flow: its label is per host and root and is held under `flow.lock`, so at launch every
    box with it is a dead launch's. This is what repoforge does by hand; flow now does it
    for every flow pipeline.
  - prime-rl: the run name is the only stable label. A resume names the run it continues,
    so at a resume launch every box with that name is the dead attempt's. A fresh run has
    nothing to sweep (an auto name has a random suffix; an explicit name needs a fresh or
    `--clean` run dir).
  - hillclimb launches through prime-rl `eval` without resume (timestamped run names) and
    keeps its own stable label, so it calls the verifiers sweep itself.
- **Image builds belong in verifiers**, next to the Prime runtime that boots them.
  `PrimeConfig.image` takes the ref the builder returns; harbor tasksets reject
  Dockerfile-only environments today because verifiers cannot build them
  (`tasksets/harbor/taskset.py:10-14,602-642`), and this is the building block for that.
  Using the SDK instead of the CLI removes the host-CLI version dependency.

## What was built

verifiers (`1e63922d`):

- `runtimes.prime.run_labels()`: base labels plus run scope, the labels every Prime box
  and tunnel of the process carries. `PrimeTunnel.expose` now passes them to
  `prime_tunnel.Tunnel(labels=...)`.
- `runtimes.sweep`: `sweep_sandboxes(labels)`, `sweep_tunnels(labels)` (re-checks the label
  and the owner client-side, skips terminated tunnels), `sweep(labels) -> SweepReport`.
  Never raises; a failed label maps to -1 and logs the manual command (`prime sandbox
  delete --label`, `prime tunnel stop --label`). The credentials check it needs is
  `utils.prime.prime_authenticated()`.
- `runtimes.prime_images`: `find_image(client, name, tag) -> PrimeImage` and
  `delete_image(name, tag)`, both in the configured team's registry. An `ensure_image`
  that built a Docker context into the registry by content digest was built here too and
  later removed: no pipeline adopted it (hillclimb still builds through the CLI).
- Tests: `tests/v1/test_sandbox_lifecycle.py`, 16 cases against SDK fakes.

prime-rl (`affdb66d`): `utils/sandboxes.sweep_run_leftovers(run_name)`, called by `rl`
when `resume` is set (not on dry runs) and by `eval --resume` before env servers start.
No-op without Prime credentials. Tests: `tests/unit/utils/test_sandboxes.py`.

flow (`929a1d0`): `Flow._serving` sweeps `[self.label]` before it opens the interception
(the tunnel) when any seat's runtime is remote and Prime credentials exist. Tests:
`tests/test_flow_sweep.py`.

## What we did not build

- A shutdown sweep in prime-rl. Clean shutdowns already tear down per runtime, and the
  crash case cannot run one; the resume sweep covers the crash.
- A confirmed/retried tunnel delete on stop (repoforge's `LabelledTunnel.sync_stop`). The
  labels make every lost delete sweepable at the next launch; the SDK's async `stop` does
  retry but blocks the loop on `process.wait`, which is why verifiers runs `sync_stop` in a
  thread.
- Sweeps for Docker, Modal and E2B. No pipeline reported leftovers there.
- hillclimb's `prime images build-vm` fallback for a VM artifact that never appears. The
  SDK has no call for it and the installed CLI (0.9.3) has no such command; the deadline
  still bounds the wait.
- `.dockerignore` handling and image garbage collection. Both pipelines assemble an explicit
  context; the digest tag makes stale images harmless.
- A sweep CLI. `prime sandbox delete --label` and `prime tunnel stop --label` exist.

Caveat carried over from hillclimb: the registry can still list a deleted image's VM
artifact as COMPLETED, so a caller that deletes images must not reuse the tag. Salt the
digest per owner (hillclimb salts with the episode id) or do not delete.

## What the pipelines can delete afterwards (separate changes)

- hillclimb: `launch.sweep_sandboxes` (use `verifiers.v1.runtimes.sweep.sweep`, which also
  sweeps tunnels) in `launch.run` and `cli._teardown`. `scoring/images.py` keeps its
  per-episode event and retain bookkeeping but drops `publish`, `_wait_vm_ready`,
  `_artifact_status`, `_published_reference`, `_run`, `push_argv`, `list_argv` and the CLI
  delete for `ensure_image(context, name, salt=episode_id)` and `delete_image`, and with them
  the venv-last PATH workaround if nothing else needs the host CLI. The
  `HILLCLIMB_SANDBOX_LABEL` plumbing stays unless hillclimb moves to a stable prime-rl run
  name with `--resume`.
- repoforge: all of `repoforge/tunnels.py` except a two-line dead-run sweep
  (`sweep([run_label(root)])` under the lock); the `tunnels.install()` and
  `sweep_label(self.label)` calls in `pipeline.py` and `worlds.py`; `world.prime_status`
  and `world.push_prime` become `ensure_image(ctx, name, tag=fingerprint)`.

## Integration notes

- prime-rl's and flow's changes import `verifiers.v1.runtimes.sweep`, so they need the
  verifiers commit first (prime-rl `deps/verifiers` pin, flow's verifiers dependency).
- Not run against the live Prime API. The SDK calls match prime-sandboxes 0.4.2 and
  prime-tunnel 0.1.12 as installed in the hillclimb venv.
