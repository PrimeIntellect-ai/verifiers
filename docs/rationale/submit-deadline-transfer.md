# Submit tool, soft deadline, large sandbox transfers (V14, V15, V16)

Three items from the pipeline survey. Base: verifiers `feat/pipeline-integration`
(c17b296f). Upstream `origin/main` (6e69354f, 2026-10-12) was checked too.

| Item | Verdict | Built |
| --- | --- | --- |
| V14 standard submit tool | Building blocks exist; the pattern is undocumented | Example + docs section, no new API |
| V15 soft deadline that still finalizes | Already in verifiers since #2693 (2026-09-27) | Nothing |
| V16 large file transfer to and from sandboxes | Real gap in `PrimeRuntime.read`/`write` | Parts, per-request timeout, retries |

## V14: submit tool

**Problem.** Pipelines need the agent to finish by handing in a typed result. The
pipeline has to validate it, stop the run, remind an agent that stops without
submitting, and tell "never submitted" apart from "ran out of time".

**Evidence (the versions found):**

1. hillclimb `hillclimb/agents/episode.py` `run_contract_lane` and the turn/validate/repair
   loop. The result is the reply JSON or a contract file. A pydantic contract is checked,
   and repair feedback goes back into the same interaction for up to `attempts` turns. It
   is not a tool. `sources/core/skills/_shared/check_submission.py` is a pre-submit gate
   script, not a submit.
2. sft-gen `sft_gen/audit_tools.py` `submit`: an in-sandbox FastMCP tool that uses
   jsonschema plus a per-kind validator and writes `audit-result.json`.
   `sft_gen/task_qc.py` `seat_attempt` sends up to `resubmits` (5) "resubmit" turns while
   the file is missing.
3. sft-gen `sft_gen/task_fix.py`: the same server with the `fixsubmit`/`rechecksubmit`
   descriptions.
4. flow #27 task-audit `examples/task-audit/audit_pipelines/agents.py`: a `vf.Toolset`
   `submit(result, report)` that writes to state, plus `@vf.stop report_submitted`. Its
   `finalize` raises when nothing was submitted.
5. flow #13 taskforge `pipelines/taskforge/taskforge/tools.py` `Decide.decide`: checks the
   outcome against an enum and writes to state. `seats.py` `finish_decision` sends NUDGES
   turns.
6. flow #25 repoforge `repoforge/tools.py` `Decide` and `seats.py` `finish_decision`:
   DECISION_REMINDERS=2, and it checks `terminated`, `stop_condition` and `last_error`.
   `tasks.py` `Undecided`/`OutOfTime` uses `trace.is_timeout`.
7. flow #13 productivity/simforge `visit.py`: a report file with a placeholder, a
   `@vf.stop out_of_time`, and a `MissingHandoff` that is excused when the budget ran out.
   (A vendored eighth version: taskforge `vendor/evo_runner` `submit_result`.)

**Existing support.** `vf.Toolset` with `@vf.tool` covers the tool. A pydantic argument
gives the MCP input schema, and a bad call returns a `ToolError` to the agent. This was
checked against `MCPServer` in the test. `vf.State` holds the result, `@vf.stop` ends
the run, `agent.interaction` plus `turn()` drives reminders, and `trace.is_timeout`
marks an agent timeout.

**Why no shared Toolset.** The versions agree on the shape but not on its parts:

- Payload: free JSON plus a report, an enum outcome with role rules, a per-solution
  verdict map, or a file in the box.
- Where the result lives: state, a box file, or the reply.
- Reminder text: static, missing-solution lists, or error-specific repair.
- Missing result: fail the call, excuse it on timeout, or keep partial state.

A generic class would need hooks for each of these, and those hooks cost more than the
roughly 30 lines it would replace. The 2026-10-05 review reached the same conclusion
("use Toolset + example"). The repeated cost is rediscovery. Three of the seven reminder
loops miss a stop check (taskforge checks only `terminated`). Two pipelines carry stale
beliefs about timeouts (see V15).

**Built.** `examples/submit_tools.py` holds the typed `Verdict` payload and the
`SubmitTools.submit` tool. It is a separate module because a tool server starts as
`python -m <module>`: the module's `__main__` must serve the toolset, so it cannot also
be the script that runs the agent. A first live run hit exactly that trap.
`examples/submit_tool.py` holds the `@vf.stop submitted` stop, `run_until_submitted`
(bounded reminders that skip any run that ended) and a `finalize` with
`Undecided`/`OutOfTime`. A live run (glm-5.2, null harness, subprocess runtime) ended
`stop_condition="submitted"` with the verdict on `trace.info`.
A section "A submit tool" was added to `docs/v1/tasksets.md`.
`tests/v1/test_submit_tool_example.py` covers the schema reaching the agent, rejection of
a bad call, the stop, both finalize endings, and the reminder bounds.

**Not built.** A `vf.Submit`/`SubmitToolset` class, a reminder helper on `Interaction`,
and jsonschema validation (use a pydantic model).

## V15: soft deadline that still runs finalize

**Problem as stated.** "Hard timeouts lose the work."

**Existing support.** verifiers #2693 (509983f9, 2026-09-27) changed this. An expired
stage deadline (`setup`, `agent`, `finalize`, `scoring`) now calls
`Rollout.timeout(stage)` and then `Trace.record_timeout`. The result is
`is_timeout=True`, `stop_condition="<stage>_timeout"`, `ok` stays true, and nothing goes
into `errors`. A timed-out agent rollout therefore still runs `finalize` and scoring with
its box alive (`rollout.py` `step`, `timeout`, `close`). repoforge relies on this
(`tasks.py` `OutOfTime`: "Verifiers finalizes a session after an agent timeout").

**Why the item came up.** productivity and simforge (flow #13, `visit.py` `GRACE`) lock
verifiers 0.3.2.dev138, which is older than #2693. Their comment says "verifiers' hard
timeout ... fails the rollout and skips `finalize`". That is no longer true. Their
`@vf.stop out_of_time` budget plus `GRACE` reproduces what `TaskTimeout(agent=...)` now
does. The one difference is that their stop refuses the next model call instead of
cancelling a tool call that is in flight. No evidence shows work lost to that difference.

**Remaining gap, not built.** An env-level `--env.timeout.episode` expiry still cancels
in-flight agent rollouts without their `finalize`. That is the outer watchdog, and no
pipeline reported losing work to it.

**Built.** Nothing. Stale workarounds to clean up in their own repos:

- hillclimb `hillclimb/agents/episode.py` `accept_solver_timeout` matches a HarnessError
  "agent timeout: rollout exceeded its" message that verifiers no longer emits, so it is
  dead code.
- productivity and simforge `GRACE` comments, once they bump verifiers.

## V16: large file transfer to and from sandboxes

**Problem.** `PrimeRuntime.write` and `read` send each file as one gateway request with
the SDK's default 300 s timeout. They retry only on gateway-marker errors
(`_repeatable`), not on timeouts or dropped connections. Upstream main behaves the same.

**Evidence.**

- flow #13 productivity `visit.py`: "about 2 GB always, 512 MB when the gateway drops
  the connection". Its commits show a run of fixes: 2b3b0e5 set a 600 s timeout, 7cf3943
  set 1800 s, 872adcc added zstd, 692b8bf added 128 MB parts with retries on dropped
  connections, and bab5390 excluded `node_modules`. The fixes reach into
  `runtime._client.upload_bytes/download_file` because verifiers exposes neither a timeout
  nor retries.
- simforge `visit.py`: 512 MB parts through `runtime.write/read` with `split`/`cat`.
- repoforge `tasks.py` `put_tree`/`get_tree`: a single tar per call, so it is exposed to
  the same limit.

**Why this layer.** Every pipeline moves files through `Runtime.read/write`, and the
limit belongs to the Prime gateway. Fixing it in `PrimeRuntime` fixes all callers. They
can then drop their `_client` reach-ins and hand-rolled chunking.

**Built** (`verifiers/v1/runtimes/prime.py`):

- Files over 128 MiB (`_TRANSFER_PART_BYTES`) cross in parts.
  - **Write:** the parts are uploaded beside the target. A box-side
    `cat ... > tmp && size check && mv` joins them, so a failed write never leaves a
    truncated target. The parts are removed on every exit.
  - **Read:** each part is cut in the box with `tail -c | head -c` one at a time, then
    downloaded and removed, so the box holds at most one extra part. A file that changes
    size during the read fails it.
- Every upload and download request gets `PrimeConfig.transfer_timeout`, 600 s by
  default. Evidence for that default: the SDK's 300 s lost visits, and 512 MB parts took
  over 600 s under load. A 128 MiB part at that throughput takes about 150 s.
- A request that fails in transit is tried 3 times, with backoff of 5 s and then 10 s.
  "In transit" means `sandbox_error_kind == "io"`, which covers timeouts and dropped or
  reset connections, plus the existing gateway-5xx markers. A lost box, a missing file,
  an account refusal and a 4xx are final.
- An unbounded `read` now starts with one `wc -c` exec to choose between a single
  download and parts. A missing file fails at that step with the same `unknown` kind as
  before, and no download is attempted. The cost is one extra RPC per read. Reads are not
  on a hot path: the most frequent caller is the MCP port-file poll, once per second.

**Tests.** `tests/v1/test_prime_transfer.py` maps the box onto a local directory and runs
the join and cut scripts in a real shell. It covers:

- single-request small transfers that carry the timeout;
- byte-exact large transfers that leave no part files;
- a failed join that keeps the old file;
- retries on dropped connections and timeouts;
- no retry after a lost box;
- a missing file failing before any download;
- a file that shrinks during a read.

`tests/v1/test_sandbox_errors.py::test_prime_io_and_exec_failures_are_typed` was adapted
to the new path, with the same kinds and messages.

**Not built.**

- Compression. It depends on the content and needs `zstd` in the image, so the caller
  passes compressed bytes when it helps.
- `put_tree`/`get_tree` directory helpers. The three versions differ in their excludes
  and prefixes, and none of them needs more than `write`/`read` once those are reliable.
- Parallel part transfers. Gateway load is the failure mode.
- Changes to the modal, e2b and docker runtimes. There is no evidence of failures there.
- Running the join and cut scripts as `runtime.user`. This base uploads as the default
  user. Upstream #2753 makes uploads act as the user, and on rebase the parts' join needs
  the same treatment.
