# Agent plumbing fixes pipelines patch locally

Survey items V5–V9, V11 and V24 of the pipeline unification plan (9 Oct 2026). Base:
`feat/pipeline-integration` @ c17b296f. Upstream checked at `origin/main` 6e69354f
(12 Oct 2026).

## Verdicts

| Item | Verdict | Where |
| --- | --- | --- |
| V5 prompt in a file, not argv | Present in c17b296f (d7c881ff), not on main; it also covers sft-gen's prompt-as-messages-file workaround. Tests added (4e56c3ec). | `harnesses/utils/launch.py` |
| V6 dead Prime job probe | Present in c17b296f (463174c9), not on main. Tests added (4e56c3ec). | `runtimes/prime.py` `_job_alive` |
| V7 Prime 502/503 repeats | Present in c17b296f (564d3e0b), not on main. Tests added (4e56c3ec). | `runtimes/prime.py` `_repeat`, `PrimeRuntime._held` |
| V8 restored-box creation wait | Present in c17b296f (fd09173a), not on main. Tests added (4e56c3ec). | `runtimes/prime.py` `_RESTORE_WAIT_ATTEMPTS` |
| V9 uv interpreter per user | Present in c17b296f (fa47445b); **superseded on main** by e2253343 (#2753), which keys the interpreter by per-user script dir the same way and also bootstraps curl as root. Drop fa47445b when the integration branch rebases. | `runtimes/base.py` `prepare_uv_script` |
| V11 retries | **Built** (43f376ee). `BashHarnessConfig.model_retries` (default 10). | `harnesses/bash`, `utils/core.py` |
| V11 group kill on timeout | **Built** (588181e8). `BashHarnessConfig.command_timeout`; process group killed, partial output kept. | `utils/core.py` `run_bash` |
| V11 bad output encoding | Present in c17b296f (23c74e73: `decode(errors="replace")`); main still uses `text=True`, which turns one bad byte into `error: ...` and loses the output. Covered by the new `run_bash` tests. | `utils/core.py` `run_bash` |
| V11 resumable conversation | **Built** (c95d8301). `Harness.resume` drops a trailing unanswered tool call. | `harness.py` |
| V11 skill listing | **Built** (a7d156a6). `BashHarness` supports `skills`. | `harnesses/bash` |
| V11 retry of a mid-stream error event (sft-gen) | Not built (see below). | — |
| V24 per-request model override | Not built (see below). | — |

## V11: the bash chat program

### Problem and evidence

- **Model-call retries.** The chat program builds `AsyncOpenAI` with the SDK default
  of 2 retries. The interception server deliberately does not retry a failure before
  its stream commits (`KEEPALIVE_GRACE_SECONDS`): it returns the status so that the
  harness SDK retries. So the program is the only retry layer for a 429/5xx burst, and
  two retries end the rollout.
  - hillclimb `environments/hillclimb/hillclimb/harnesses/bash.py` rewrites
    `CHAT_PROGRAM_SOURCE` at import to add `max_retries=10` (raises if the anchor text
    changes).
  - sft-gen `sft_gen/bash.py` forks the harness (`sft-gen-bash`) for
    `request_max_retries=200` through an env var spliced into the program source.
- **Timeout kills only `bash`.** `subprocess.run(timeout=3600)` kills the `bash -c`
  process; a child (`sleep`, a hung server, `grep -R /` over `/proc`) keeps running and
  all output is lost (`TimeoutExpired` becomes `error: Command ... timed out`). The
  limit is fixed at one hour.
  - hillclimb replaces `run_bash` with a `Popen(start_new_session=True)` +
    `os.killpg` version that returns the output so far plus a note, and reads the limit
    from `HILLCLIMB_BASH_TIMEOUT` (role agents: 300 s).
- **Resumable conversation.** `Harness.resume` relaunches on the trace's branch. A turn
  cancelled mid-command (episode cancel, rollout timeout, box loss) leaves the branch
  ending in an assistant message whose tool calls have no results; providers reject that
  conversation, so the resume fails on its first model call.
  - hillclimb `RoleBashHarness.resume` + `resumable_conversation()` pop that message.
  - The trace cannot hold a partial set of results: tool results only reach the trace
    in the next request, which is sent after every call is answered.
- **Skill listing.** `BashHarness` has `SUPPORTS_SKILLS = False`, so `skills` is rejected.
  - hillclimb `RoleBashHarness`: installs skill folders in a per-instance directory
    (two sessions on one box must not mix) and lists `- name (path): description` in the
    system prompt.
  - sft-gen `RetryBashHarness.skills_fragment`: installs host folders, finds every
    `SKILL.md` under host and runtime roots, lists the paths through a prompt template.

### What we built

One commit each, so they split into PRs:

1. `model_retries` on `BashHarnessConfig` (default 10, hillclimb's tested value; with
   the SDK's backoff capped at 8 s that rides out about a minute of errors), passed as
   `--max-retries`. The program keeps the SDK default when the flag is absent, so
   `NullHarness` is unchanged. `chat()` already bounds its stream retries by
   `client.max_retries`, so those scale too. Safe to raise: SDK retries carry
   `x-stainless-retry-count`, which the interception's body-digest replay guard keys on.
2. `command_timeout` on `BashHarnessConfig` (default 3600, as before), passed as
   `--command-timeout`. `run_bash` starts each command in its own session, kills the
   group on timeout, and returns the output so far plus
   `error: command exceeded Ns and was killed`. A command that ends normally leaves its
   background children (`server &`) running, as before.
3. `Harness.resume` drops trailing assistant messages that carry tool calls. Applies to
   every harness using the default relaunch-on-the-conversation resume.
4. `BashHarness.SUPPORTS_SKILLS = True`: setup installs `config.skills` in a
   per-instance directory; launch lists each `SKILL.md` found there (name and
   description from its front matter, else the folder name) in the system prompt.

### What we will not build

- **Retrying a mid-stream `APIError` in the program** (sft-gen 9c6040d). In c17b296f the
  interception reruns a committed turn that failed 5xx/429 up to 5 times
  (`COMMITTED_TURN_ATTEMPTS`), and a relayed provider stream that ends without its
  terminal event becomes a `ProviderError`, which takes the same path. The error events
  that still reach the program are either the rerun budget exhausted or a deliberate
  400/409 (`rollout stopped`, `rollout concluded`); retrying those in the program only
  delays the stop. If sft-gen still sees error events on its current pin, the fix belongs
  in the interception's rerun predicate.
- **A prompt template for the skill listing** (sft-gen `skills_prompt`). The listing
  states where skills are and what each does; the task's own prompt can add policy.
- **An env-var timeout** (`HILLCLIMB_BASH_TIMEOUT`). Config fields reach the program
  over argv like the other flags, and the agent's commands do not inherit them.

## V24: per-request model override

Claim: "Interception forces one model per rollout, so FW's text seat can't call a vision
helper."

Evidence: data-flywheel (`/data/pep-pipelines/environments/data-flywheel`) solves this
with per-seat model profiles: `examples/web-app-clone/models.toml` puts the
investigator seat alone on a vision model (`test_examples.py::
test_the_web_apps_vision_investigator_profile_resolves_...`). Its helpers are subagents
on the caller's own model; its verifiers fork (`origin/flywheel`) adds only
`wrap_client` to the interception, no model override. No other pipeline calls a second
model from inside a seat.

Existing support: each agent of a v1 episode has its own `AgentConfig.model`, and
`message_agent` lets a seat steer a sibling seat; a host-side `vf.Toolset` can hold its
own client (as `vf.JudgeConfig` does) and call any model.

Not built. An override in the interception would let the box pick any model on the
rollout's credentials, and would put turns of two models on one trace graph that
training reads as one policy.

## What hillclimb can delete once these land

- `hillclimb/harnesses/bash.py`: `MODEL_CALL_RETRIES`, `_CLIENT`, `_RUN_BASH`,
  `_GROUP_RUN_BASH`, `_patched` and the import-time rewrite of `CHAT_PROGRAM_SOURCE`;
  `COMMAND_TIMEOUT_ENV` (set `command_timeout = 300` on role harness configs instead of
  the env var); `resumable_conversation` and `RoleBashHarness.resume`; `SKILLS_PROMPT`,
  `_skill_listing`, `setup`, `resolve_prompt` and `SUPPORTS_SKILLS`. That is all of
  `RoleBashHarness`; `load_role_harness` reduces to `vf.load_harness(config)` with
  `command_timeout` set for roles.
- `tests/test_resilient_bash.py` (all four tests move upstream).
- In `tests/test_skills.py`, the `RoleBashHarness` assertions (around lines 541–553)
  become assertions on the stock `BashHarness`.
- One wording difference to keep in the role prompts: hillclimb's `SKILLS_PROMPT` adds
  "Where a card names `skill_dir`, use that skill's folder."; the upstream listing does
  not.

sft-gen can replace `sft-gen-bash` with the stock `bash` harness and
`model_retries = 200`: the prompt file is V5, the uv upgrade is on main (bf2834ca), and
skills are listed natively. It keeps, or moves into its task prompt, its
`skills_prompt` wording and the `_check_result` exit logging; the mid-stream `APIError`
retry is dropped (see above).
