# Bump a Harness

Each agent harness pins the upstream program it installs. We try to keep these
pins current. A bump can change how the program replays a conversation, so
re-verify the harness after every bump.

## 1. Find the pins

The pins are in `verifiers/v1/harnesses/<harness>/harness.py`:

- The `version: PinnedVersion` field of the harness config.
- Module constants for companion packages, such as `ACP_VERSION`,
  `MCP_VERSION`, or `PRIME_AGENT_VERSION`.

Update each pin to a released upstream version. Bump one harness per PR.

## 2. Recheck the replay overrides

Some harnesses override `Harness.replay_view`. The override declares the
cosmetic rewrite that the upstream client applies when it replays an earlier
assistant turn, such as dropping whitespace-only text. The view receives the
dialect the harness speaks (`chat_completions`, `responses` or
`anthropic_messages`), because a client can rewrite differently per wire
format. The trace graph applies the view when it matches a replayed turn to its
sampled node. If the view is wrong, the trace forks.

Each override has a comment with a link to the upstream source line at the
pinned tag. For each link:

1. Open the same file at the new tag.
2. Confirm that the rewrite is still there and has not changed.
3. Update the link to the new tag and line.

If the rewrite is gone, remove the override. If the rewrite changed, update the
view in `verifiers/v1/harnesses/utils/replay.py`. A view may only normalize
cosmetic differences. If the program now changes what the model sees, such as
dropping a turn, do not hide it with a view. That is a real branch.

Some integration tests let a harness fork by design. `FORKS` in
`tests/integration/grid.py` links the upstream reason for each one. Recheck those
links the same way.

Closed-source programs (claude-code) have no source link. Verify them with the
tests in step 3.

## 3. Run the harness tests

Run the integration cells of the harness on prime sandboxes. Do not use docker.
The mark is the harness id with `_` for `-`:

```bash
uv run pytest tests/integration -n 8 -m "<harness_mark> and prime"
# e.g. -m "claude_code and prime"
```

The tests assert `trace.num_branches == 1`. `test_multi_turn.py` also runs
every other `dialect` the harness config can select. If a cell forks, dump its trace and
compare the sampled node with its unsampled sibling under the same parent. The
difference is the new replay rewrite. Fix it as in step 2.

## 4. Open the PR

Open a PR with the bump. List each pin you changed and each replay override you
rechecked, with the new upstream links.
