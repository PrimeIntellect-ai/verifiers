# Multi-Agent Architecture Reference

Target audience: new reader who needs to be productive in ~15 minutes. This page names the pieces, draws the data path, defines every concept, and points at files for follow-up.

## 1. Framing

`verifiers` natively supports single-actor rollouts through `MultiTurnEnv`: one environment talking to one model across turns. Multi-agent debate (and cousins — RPS, PD, proposer-solver, self-play) has a different shape: N speakers share a single transcript, some turns happen simultaneously, and training needs one rollout *per speaker*, not one per episode. This stack extends `verifiers` with four pieces stacked around that shape:

1. A **functional kernel** (`multi_agent_kernel.py`) owning transcript evolution as a pure reducer, with immutable state and per-utterance parse quarantine.
2. A **`MultiAgentEnv` base class** (`multi_agent_env.py`) that runs the slot-scheduled rollout loop on top of `vf.Environment`, including atomic commit for simultaneous slots.
3. A **`MultiAgentRubric` base** (`rubrics/multi_agent_rubric.py`) that owns the rollout/scoring error boundary and writes a typed `MARScore` (Member-Attributed Reward) as the single source of truth.
4. A **bridge** (`multi_agent_bridge.py`) that projects one episode-level `RolloutOutput` into a list of per-member training rollouts the trainer's advantage path and interleave code can consume unchanged.

The goal is that adding a new N-agent env is a two-method subclass of `MultiAgentEnv` plus a `build_marscore` method on `MultiAgentRubric`, with everything else (atomicity, prefix-cache keys, error quarantine, wandb projection, per-member training decomposition) shared.

## 2. Sequence diagram

End-to-end, from a dataset row to a list of trainable `MemberRollout`s and onward into advantage estimation + the trainer.

```
dataset row (prompt, answer, example_id)
        │
        ▼
MultiAgentEnv.rollout(input)                [verifiers/envs/multi_agent_env.py]
        │
        ├── init_state → state + KernelState(slot_index=0)
        │
        └── while not is_completed(state):
                slot = schedule.current_slot(kernel)         [multi_agent_kernel.py]
                ├── len(slot.agents) == 1 → _run_sequential_slot
                └── len(slot.agents) >  1 → _run_simultaneous_slot   [3-phase atomic]
                        │
                        └── per call: get_model_response(state, prompt,
                                            request_context=ModelRequestContext(
                                                lineage_key=agent,
                                                usage_tracker=branch_tracker))
                                        │
                                        └── apply_action → Utterance
                                            (parse_channels split into raw/public/private)
                                            → commit to state["_kernel"].transcript
                                            → append TrajectoryStep to state["trajectory"]
        │
        ▼
DebateRubric.score_rollout(state)             [verifiers/envs/debate_rubric.py]
  (base MultiAgentRubric owns boundary:
   prompt_too_long / existing error / vf.Error → errored MARScore)
  subclass builds: state["mar_score"] = MARScore(members=[...], episode_scalar=..., episode_metrics={...})
        │
        ▼
state_to_output(state)                        [verifiers/utils/save_utils.py]
  project state["mar_score"] → output["reward"] + flat top-level metrics (wandb)
  project state["mar_score"] → output["mar_score"] (bridge)
        │
        ▼
RolloutOutput  ──►  [wire boundary: JSON / IPC]  ──►  orchestrator
        │
        ▼
rollout_to_member_rollouts(output)            [verifiers/multi_agent_bridge.py]
  read output["mar_score"] → group steps by extras["member_id"] → one MemberRollout per member
        │
        ▼
list[MemberRollout]
        │
        ├──► compute_rae_advantages(rollouts, rae_state)  [prime-rl/.../multi_agent_advantage.py]
        │      per-(task, example_id, member_id) EMA baseline
        │
        └──► trainer (pretokenize_rollout_trajectory → interleave_rollout → TrainingSample)
              completion_mask respects step["extras"]["parse_error"] for quarantined turns
```

The left column is roughly where the component lives (env-layer → rubric-layer → save/bridge-layer → orchestrator/trainer). The wire boundary is the serialization point: the bridge reads the projected `output["mar_score"]` dict, not the live pydantic `MARScore`, which keeps orchestrator code free of `verifiers.types` dependencies beyond the `MemberRollout` TypedDict.

## 3. Concept glossary

Cross-links use backticks matching the glossary headers; every row points at the file where the definition lives.

| Concept | Where it lives | What it is |
| --- | --- | --- |
| `SlotProgram` | Protocol, `verifiers/envs/multi_agent_kernel.py` | Schedule interface; `current_slot(kernel) -> TurnSlot \| None`. In-tree impl: `StaticSchedule`. |
| `TurnSlot` | Frozen dataclass, `verifiers/envs/multi_agent_kernel.py` | One schedule step: `slot_id`, `agents: tuple[str, ...]`, `phase: str`. `len(agents) == 1` → sequential, `> 1` → simultaneous barrier. Rejects empty + duplicate-agent tuples at construction. |
| `KernelState` | Frozen dataclass, `verifiers/envs/multi_agent_kernel.py` | Immutable episode state. Carries `slot_index`, `transcript: tuple[Utterance, ...]`, simultaneous-slot `pending: MappingProxyType[str, Utterance]` buffer, cached `_active_slot: TurnSlot \| None`. |
| `Utterance` | Frozen dataclass, `verifiers/envs/multi_agent_kernel.py` | Committed agent output with three channels: `raw_content` (verbatim), `public_channel` (opponent-visible, think-stripped), `private_channel` (author's stripped think body or `None`). Per-utterance `parse_error: str \| None` carries quarantine info. |
| `parse_channels` | Function, `verifiers/envs/multi_agent_kernel.py` | One-pass split into `(public, private)`; always strips native `<think>`/`<thinking>` regardless of configured `think_tag` (security boundary — native reasoning tokens are discarded, never surfaced as private). |
| `apply_action` | Function, `verifiers/envs/multi_agent_kernel.py` | Pure reducer. `KernelState × program × member × raw × tokens → ActionResult`. Raises `KernelProtocolError` on protocol violations (wrong agent, duplicate submit, finished episode); per-utterance quarantine on `ContentParseError`. |
| `MultiAgentEnv` | Class, `verifiers/envs/multi_agent_env.py` | Generic N-agent rollout loop on top of `vf.Environment`. Owns the sequential/simultaneous slot dispatch, atomic 3-phase commit for simultaneous, stop-condition priority, lineage-keyed prefix cache. Marks `rollout` / `_run_sequential_slot` / `_run_simultaneous_slot` as `@final`. |
| `ModelRequestContext` | Dataclass, `verifiers/envs/request_context.py` | Per-inference-call metadata: `lineage_key` (prefix-cache partition key) + `usage_tracker` (per-agent-branch accounting). Passed as a kwarg to `get_model_response`; NOT stored on `State`. |
| `MARScore` / `MemberScore` | Pydantic models, `verifiers/types.py` | Single source of truth for episode scoring. Base `MultiAgentRubric` writes `state["mar_score"]`; `state_to_output` one-way-projects it to legacy `output["reward"]` + flat top-level metrics (via `MARScore.to_wandb_flat()`) at the wire boundary. Construction-time validators reject empty members list and duplicate `member_id`. |
| `MultiAgentRubric` | Abstract class, `verifiers/rubrics/multi_agent_rubric.py` | Base rubric that owns the rollout/group error boundary. Subclass implements `build_marscore(state) -> MARScore`; base handles short-circuits (`prompt_too_long`, pre-existing `state["error"]`) and scoring-time `vf.Error` → errored MARScore via `build_errored_marscore`. Non-`vf.Error` exceptions still propagate. |
| `rollout_to_member_rollouts` | Function, `verifiers/multi_agent_bridge.py` | Pure projection from one `RolloutOutput` (with `mar_score`) + tagged trajectory to `list[MemberRollout]`. One rollout per member, with the member's own steps + per-member reward from `mar_score`. |
| `MemberRollout` | TypedDict, `verifiers/types.py` | `RolloutOutput`-compatible dict with `member_id` + per-member `reward`. The unit the trainer's advantage + interleave path consumes. RAE baselines key on `(task, example_id, member_id)`. |

Two adjacent things that frequently get conflated with the above and are worth naming explicitly:

- `ContentParseError`, `KernelProtocolError` live in `verifiers/errors.py`. The first is per-utterance quarantine (parse failure on model output — rollout continues with `public_channel=""`, `parse_error` set). The second aborts the slot — it signals a framework invariant violation (wrong agent, duplicate submit, submission after episode end) that the rollout loop does not try to recover from.
- `StateUsageTracker.fork` / `StateUsageTracker.merge` in `verifiers/utils/usage_utils.py` are what makes the simultaneous-slot usage accounting atomic. Each agent in a simultaneous slot charges a forked child; only the publish phase merges children back into the parent, so a cancelled slot drops its child forks and leaves the parent at the pre-slot snapshot.

## 4. Monotonic prompt invariant

> **Invariant.** For a fixed member `A`, `build_prompt(state, A, slot_{N+1})` MUST be a structural extension of `build_prompt(state, A, slot_N)`: append at most a `[user, assistant-prefill]` pair at the tail, never modify or remove any prior message. This is what lets vLLM's prefix cache (via `OpenAIChatCompletionsTokenClient.get_prompt_ids`) reuse the shared token prefix across all of `A`'s turns. Violation = O(T²) tokenization over a T-turn episode and silent cache thrash. Tested in `prime-rl/tests/unit/orchestrator/test_multi_agent_env.py::test_build_prompt_monotonic_across_slots` and the debate counterpart in `test_debate_env.py`.

Two practical consequences when writing `build_prompt` on a subclass:

1. The visibility policy (what opponents see vs. what the author sees) has to be applied in a way that is stable across turns. Rendering "what A saw at slot N+1" must include, byte-identically, whatever A saw at slot N plus only the newly-committed turn (rendered per A's visibility policy). The default `visibility_policy` — opponents see `public_channel`, author sees full `raw_content` — is chosen to satisfy this: author-side rendering always reproduces earlier author turns verbatim regardless of whether the opponent was under `public_only` at the time.
2. `_prepare_prompt` runs `fold_consecutive_user_messages` → `maybe_normalize_messages` after your `build_prompt`. The fold is idempotent and monotone on valid inputs; the normalizer is a pass-through in the monotone case. If you find you need to insert a message in the middle of the transcript to "patch things up", you have violated the invariant — fix upstream (prompt structure, visibility rules, or the schedule), don't rewrite history.

The `lineage_key` the env passes into `ModelRequestContext` is the prefix-cache partition key. It is just the member id — concretely, the cache bucket for A's prompts is keyed by `lineage_key="A"`, distinct from B's cache bucket. Without a `lineage_key`, the cache would partition by first-match, and agents with overlapping prefixes would cross-contaminate.

## 5. Atomicity

> **Invariant.** For simultaneous slots: all-or-nothing writes to `state["_kernel"]`, `state["trajectory"]`, and the parent `StateUsageTracker`. Three-phase protocol — **(1) fan-out** under `asyncio.TaskGroup` (first raise cancels siblings) → **(2) stage** into a local kernel + pre-built `TrajectoryStep` list + per-agent `extract_fields` → **(3) publish** (trajectory appends + kernel assignment + tracker merge, no `await`, no raises after this point). A doomed slot drops its per-agent usage forks; the parent tracker stays at its pre-slot snapshot.

The reason this is not a one-liner: each agent's model call is async, can fail independently, and — critically — TaskGroup wraps concurrent raises into an `ExceptionGroup`. The loop catches the group, flattens nested groups recursively via `_flatten_exception_group`, then picks *one* exception to re-raise with priority `OverlongPromptError > vf.Error > anything else`. Suppressed peer exceptions are logged with the slot id. This is deliberate: using chained `except* OverlongPromptError:` / `except* vf.Error:` would cause BOTH branches to fire on a mixed doomed slot (one agent overlong, the other generic vf.Error), which re-wraps both re-raises into a new `ExceptionGroup` at the call site, which the outer `except vf.Error:` in `rollout` does not match — the effect was a P0: a doomed slot would crash the whole rollout instead of being caught and recorded. The flatten-and-pick-one path exists specifically to keep that match clean.

`extract_fields` runs in phase 2, not phase 3, deliberately: it is user-provided code that may raise. Raising from there must discard the staged buffers cleanly instead of partially publishing, so it lives in the "staged, not yet published" phase.

## 6. Known gotchas / pitfalls

- `state["mar_score"]` MUST be set by the rubric or `rollout_to_member_rollouts` will `KeyError` on the wire side. Single-agent legacy envs use the `state["metrics"]` path instead — `state_to_output` handles both (`mar` branch vs. legacy branch) and the branch is gated on `state.get("mar_score") is not None`.
- `step["extras"]["parse_error"]` being truthy means the completion tokens MUST be masked in training. The kernel's per-utterance quarantine zeros the `public_channel` but NOT the `raw_completion` — the raw model output is still present on the `Utterance` and the `Response`, and training would otherwise learn from malformed markup. The trainer's `completion_mask` respects this flag; any new training path must too.
- Member ≠ agent. "Member" is a roster entry, static per episode (e.g. `["debater_a", "debater_b", "judge"]`). "Agent" is a per-turn participant, a subset of members scheduled for that specific slot (e.g. `("debater_a", "debater_b")` for a simultaneous opening, `("judge",)` for a final verdict). The rubric writes one `MemberScore` per **member**; the schedule defines `slot.agents`.
- `ModelRequestContext.lineage_key` is required for prefix-cache correctness in multi-agent rollouts. Without it, the cache partitions by first-match and cross-contaminates across speakers (silent performance regression, not a crash). The env passes it on every `get_model_response` call — if you introduce new inference-triggering code paths, thread it through.
- `MARScore` construction-time validators reject empty `members` and duplicate `member_id`. That is load-bearing: drift between the rubric writer and the bridge reader is structurally impossible after a `MARScore` is constructed. Don't route around the validators by building the dict by hand.
- Native `<think>` / `<thinking>` blocks are ALWAYS stripped from the public channel, even when your pack configures a different private-channel tag. Their content is DISCARDED — not surfaced as `private_channel`, not surfaced to rubrics, not surfaced anywhere. `parse_channels` treats native reasoning blocks as a third-party model artifact the pack author did not opt into. If your pack wants reasoning in the private channel, set `think_tag="thinking"` (or `"think"`) so the configured-tag pass consumes it.
- In a simultaneous slot, if one agent is fast and another is slow, the slow one can still be cancelled by a sibling's raise — that is intentional, and the cancelled tokens are discarded from accounting via the `StateUsageTracker.fork`/`merge` pair. If you add new bookkeeping state that must be atomic with kernel commit, do the same fork/merge dance.
- `KernelState` is frozen; `state["_kernel"]` is reassigned, never mutated. If you see code that looks like it's mutating the kernel in place (e.g. `state["_kernel"].transcript = ...`) it's a bug — that attribute is read-only and the reassignment via `dataclasses.replace` + `state["_kernel"] = ...` is load-bearing for the atomicity story.

## 7. Where to look next

- **Add a new multi-agent env.** Subclass `MultiAgentEnv`, implement `build_prompt(state, member_id, slot)` (respect the monotonic invariant) and `render_completion(state)`. Optional overrides: `extract_fields`, `visibility_policy`, `resolve_agent`. Reference: `verifiers/envs/debate_env.py`.
- **Add new multi-agent scoring.** Subclass `MultiAgentRubric`, implement `build_marscore(state) -> MARScore`. The base class already handles error boundaries. Reference: `verifiers/envs/debate_rubric.py` (W+G+M scoring).
- **Add dynamic / adaptive scheduling.** Implement the `SlotProgram` protocol (`current_slot(kernel) -> TurnSlot | None`). `StaticSchedule` is the in-tree static-tuple implementation; a dynamic schedule can condition on `kernel.transcript`, `kernel.slot_index`, or external state. Reference: `verifiers/envs/multi_agent_kernel.py`.
- **Add per-call telemetry / routing hints.** Extend `ModelRequestContext` with new ephemeral fields; thread them from the `get_model_response` call sites in `_run_sequential_slot` and `_run_simultaneous_slot`. Keep it ephemeral — anything that belongs on durable rollout state lives on `State`, not `ModelRequestContext`.
- **Advantage / trainer side.** `src/prime_rl/orchestrator/multi_agent_advantage.py` holds `compute_rae_advantages` (role-conditioned EMA baselines keyed by `(task, example_id, member_id)`). `src/prime_rl/orchestrator/multi_agent_bridge.py` is a thin compatibility shim that re-exports `rollout_to_member_rollouts` from verifiers. The trainer's tokenization/interleave path respects `step["extras"]["parse_error"]` for masking.
- **Tests.** Kernel + env invariants: `prime-rl/tests/unit/orchestrator/test_multi_agent_env.py` (monotonic prompt, atomicity, stop-condition priority, parse quarantine). Rubric boundary: `prime-rl/tests/unit/orchestrator/test_multi_agent_rubric.py`. Bridge projection: `prime-rl/tests/unit/orchestrator/test_multi_agent_bridge.py`. Advantage EMA: `prime-rl/tests/unit/orchestrator/test_multi_agent_advantage.py`. Debate-specific: `test_debate_env.py`, `test_debate_prompts.py`, `test_debate_fields.py`.

## 8. Lifecycle of one turn (worked example)

A walk-through of a single sequential slot, with every state mutation named. Concrete scenario: a two-debater debate where `slot = TurnSlot(slot_id=3, agents=("debater_a",), phase="rebuttal")`, the kernel already has two transcript utterances from `slot_id=0, 1, 2`, and no pending buffer. The loop has just decided the slot is non-null and `len(slot.agents) == 1`, so it calls `_run_sequential_slot(state, slot)`. Top to bottom:

1. `_prepare_prompt(state, "debater_a", slot)` calls the subclass's `build_prompt`, then runs `fold_consecutive_user_messages` and `maybe_normalize_messages`. The returned `Messages` list is what goes on the wire. The subclass is responsible for the monotonic-extension property here; the env does not validate it at runtime (the cost would dominate the tokenization savings the invariant buys).

2. `resolve_agent("debater_a")` returns `(client_override, model_override)` from `self.agent_overrides`, falling back to `(None, None)` for default routing. Self-play and fixed-opponent setups use this to pin different speakers to different endpoints.

3. `self._get_usage_tracker(state, create_if_missing=True)` returns the parent `StateUsageTracker` attached to `state`. Sequential slots charge directly against the parent — no fork/merge dance, because there is no concurrency.

4. `get_model_response(...)` with `request_context=ModelRequestContext(lineage_key="debater_a", usage_tracker=parent_tracker)` issues the chat-completions call. The client uses `lineage_key` as the prefix-cache partition key, so `debater_a`'s prompts hit one cache bucket and `debater_b`'s another — even when the in-context messages overlap. The usage tracker is incremented from the response before return.

5. `apply_action(state["_kernel"], self.schedule, "debater_a", content, token_count, think_tag=self.think_tag)` runs the pure reducer. Inside: `parse_channels(raw_content, think_tag)` splits into `(public, private)` or raises `ContentParseError`. On `ContentParseError`, the reducer quarantines the utterance: `public_channel=""`, `private_channel=None`, `parse_error=str(exc)`, but does NOT raise — the rollout continues. On `KernelProtocolError` (wrong agent, duplicate submit, episode finished) it DOES raise and the outer rollout loop catches it as `vf.Error`, recording on `state["error"]`.

6. For `len(slot.agents) == 1`, the reducer commits immediately: `new_state = replace(state, slot_index=slot_index+1, transcript=transcript + (utterance,))`. `state["_kernel"] = result.new_state` lands this on the State dict.

7. `self.extract_fields(utt.public_channel, "debater_a", slot)` runs next — it reads the public channel (never `raw_content`, never `private_channel`) and returns a `dict[str, Any] | None` of structured fields like `{"answer": "A", "argument": "..."}`. Default implementation returns `None`. Rubric scoring consults `step["extras"]["fields"]` during grading.

8. `self._build_step(state, prompt, response, utt, fields)` composes a `TrajectoryStep` via `parse_response_message` + `parse_response_tokens`, setting `extras = {"member_id": "debater_a", "phase": "rebuttal"}`. If `fields is not None`, `extras["fields"] = fields`. If `utt.parse_error is not None`, `extras["parse_error"] = utt.parse_error` — and this is the quarantine flag the trainer reads to mask completion tokens.

9. `state["trajectory"].append(step)`. This is the only mutation of `trajectory` in the sequential path; kernel transcript and trajectory list stay 1:1 after commit.

10. The outer `while not await self.is_completed(state):` loop re-checks stop conditions. Priority order: `has_error` (100) > `schedule_exhausted` (50) > `prompt_too_long` (10). If `schedule_exhausted` fires (the schedule returned `None` for the new `slot_index`), the loop exits, `render_completion` populates `state["completion"]`, and `rollout` returns. Otherwise the next slot dispatches.

Simultaneous slots (covered in §5) follow the same shape but stage all three mutations (`trajectory.append`, kernel assignment, tracker merge) at the end so they land atomically or not at all.

## 9. Rubric error boundary in detail

`MultiAgentRubric.score_rollout` is the choke point between env errors and scoring errors. Here's the decision tree it runs:

```
score_rollout(state):
    if state.get("prompt_too_long"):
        # rollout layer flagged prompt overflow before any real work
        state["mar_score"] = build_errored_marscore(
            state, error_type="prompt_too_long", error_phase="rollout")
        return

    if state.get("error") is not None:
        # rollout layer already recorded a vf.Error during rollout
        state["mar_score"] = build_errored_marscore(
            state, error_type=type(existing_error).__name__, error_phase="rollout")
        return

    try:
        state["mar_score"] = await self.build_marscore(state)
    except vf.Error as error:
        # scoring-time failure — record and convert
        state["error"] = error
        state["mar_score"] = build_errored_marscore(
            state, error_type=type(error).__name__, error_phase="scoring")
    # non-vf.Error exceptions (KeyError, RuntimeError, assertion failures)
    # propagate loud — these are programming bugs, not graceful degradation.
```

Two consequences:

- **`mar_score` is always populated after `score_rollout` returns normally.** The bridge can assume this. There is no "missing MARScore → fall through to legacy metrics" branch on the bridge side for multi-agent envs; `state_to_output` has one such branch only to keep single-agent envs working through the same serializer.
- **`vf.Error` is the graceful-degradation channel; anything else is a bug.** A dataset schema violation (e.g. `state["answer"]` missing) should raise `KeyError`, not be swallowed. Debate's `build_marscore` relies on this explicitly — `state["answer"]` lookup raises `KeyError` with a targeted message rather than defaulting to empty string. If the rubric needs a judge client and the pack declared one, it raises `RuntimeError` when the judge step never ran (early termination, malformed verdict) rather than silently falling back to answer-grading, which would generate fake training signal.

The errored MARScore itself is uniform: `reward=0.0` for every member, `episode_scalar=0.0`, `episode_metrics={"errored_rollout": 1.0, "error_type": ..., "error_phase": ...}`. Subclasses can override `build_errored_marscore` to add domain-specific error metadata — `DebateRubric` does this to include per-member `parse_error_count` on the error path, so we can see which member blew up even when scoring was impossible.

## 10. Subclassing recipe: new multi-agent env in ~50 lines

Concrete steps to add a new N-agent env, e.g. prisoner's dilemma with two players + a settler. The shared machinery carries most of the weight; the domain code is small.

**Step 1: pick a roster and a schedule.**
```python
members = ["player_a", "player_b", "settler"]
schedule = StaticSchedule(slots=(
    TurnSlot(slot_id=0, agents=("player_a", "player_b"), phase="choose"),
    TurnSlot(slot_id=1, agents=("settler",),              phase="settle"),
))
```
The first slot has two agents → simultaneous (barrier until both submit). The second has one → sequential.

**Step 2: subclass `MultiAgentEnv` and implement `build_prompt` + `render_completion`.**
```python
class PDEnv(MultiAgentEnv):
    async def build_prompt(self, state, member_id, slot):
        # must be monotonic: slot_N+1 extends slot_N byte-for-byte.
        # Typical pattern: render the shared context up to the current slot,
        # then append a member-specific instruction message for this slot.
        visible = [
            utt for utt in state["_kernel"].transcript
            if self.visibility_policy(utt, member_id) != "hidden"
        ]
        return (
            [{"role": "system", "content": self.system_prompt(member_id)}]
            + [render(utt, member_id) for utt in visible]
            + [{"role": "user", "content": self.prompt_for_slot(slot, member_id)}]
        )

    async def render_completion(self, state):
        # what gets saved to state["completion"] — typically a human-readable
        # rendering of the whole transcript.
        state["completion"] = [render(utt, viewer_id=None) for utt in state["_kernel"].transcript]
```

**Step 3: optional — override `extract_fields` to surface structured data.**
```python
async def extract_fields(self, public_channel, member_id, slot):
    # parse choice from the public channel; returns None if nothing to extract.
    match = re.search(r"CHOICE:\s*(COOPERATE|DEFECT)", public_channel)
    if match is None:
        return None
    return {"choice": match.group(1)}
```
`step["extras"]["fields"]` is where this lands for the rubric to consume.

**Step 4: subclass `MultiAgentRubric` and implement `build_marscore`.**
```python
class PDRubric(MultiAgentRubric):
    async def build_marscore(self, state):
        # pull the latest choice per debater from extras["fields"]
        choices = {}
        for step in reversed(state.get("trajectory", [])):
            mid = step["extras"].get("member_id")
            if mid in ("player_a", "player_b") and mid not in choices:
                fields = step["extras"].get("fields") or {}
                if "choice" in fields:
                    choices[mid] = fields["choice"]

        rewards = payoff_matrix(choices.get("player_a"), choices.get("player_b"))
        return MARScore(
            members=[
                MemberScore(member_id="player_a", reward=rewards["a"]),
                MemberScore(member_id="player_b", reward=rewards["b"]),
                MemberScore(member_id="settler",  reward=0.0),
            ],
            episode_scalar=(rewards["a"] + rewards["b"]) / 2,
            episode_metrics={"mutual_cooperation": float(
                choices.get("player_a") == choices.get("player_b") == "COOPERATE"
            )},
        )
```
The base class's `score_rollout` wraps this in the error boundary; a `vf.Error` from `build_marscore` becomes an errored MARScore, anything else propagates loud.

**Step 5: wire the env + rubric into the env server.** This is env-dir scaffolding (a `load_environment` function, a `pyproject.toml` env-entrypoint) — see `verifiers/envs/debate/` for the canonical example. The rollout path, atomicity, bridge projection, and training path all come for free; nothing PD-specific lives in the framework.

## 11. Relation to single-actor `MultiTurnEnv`

`MultiAgentEnv` is a sibling of `MultiTurnEnv`, not a subclass. The reason, stated once and cross-referenced from `multi_agent_env.py`'s module docstring: `MultiTurnEnv.rollout` is `@final` and shaped for a single conversation (one env turn → one agent turn → one env turn), with `env_response` returning the env's next turn and `is_completed` terminating the loop. Retrofitting N speakers sharing a transcript into that shape would mean either (a) wedging a slot-loop inside `env_response` and lying to the rest of the `MultiTurnEnv` machinery about what a turn is, or (b) unfreezing `@final` on a base class dozens of envs depend on. Both are worse than having a sibling class with its own `@final rollout`. The cost is one class that does not inherit from the other, and some duplicate wiring (stop conditions, trajectory construction, usage tracking). The benefit is that single-actor and multi-agent envs have isolated rollout loops and isolated invariants.

Concretely, the stop-condition decorator (`@vf.stop(priority=...)`) and the `State` TypedDict are shared. The rollout loop, the schedule abstraction, the kernel, the channel-splitting, the atomic simultaneous commit, and the `MemberRollout`-producing bridge are multi-agent specific and have no counterpart in `MultiTurnEnv`.

## 12. Invariant summary

One-line recap of every load-bearing invariant across the stack, in the order they matter:

1. **Monotonic `build_prompt`** — per member, slot N+1 extends slot N byte-for-byte. Violation → O(T²) tokenization. §4.
2. **Simultaneous-slot atomicity** — all-or-nothing writes to kernel + trajectory + parent tracker. 3-phase protocol: fan-out → stage → publish. §5.
3. **`parse_channels` always strips native `<think>`** — even when `think_tag` is configured to something else. Native reasoning tokens are discarded, never surfaced. §3.
4. **Per-utterance parse quarantine** — `ContentParseError` → utterance with `parse_error` set, rollout continues; trainer masks those completion tokens. §6.
5. **Kernel protocol errors abort the slot** — wrong agent, duplicate submit, finished episode → `KernelProtocolError` → caught as `vf.Error` by the rollout loop, recorded on `state["error"]`. §3, §9.
6. **`lineage_key` is required for prefix-cache correctness** — without it the cache cross-contaminates across speakers. §4.
7. **`MARScore` is always populated after `score_rollout` returns** — the bridge can assume it. `vf.Error` during scoring converts to an errored MARScore; non-`vf.Error` exceptions propagate loud. §9.
8. **`MARScore` construction validators reject empty members + duplicate `member_id`** — drift between rubric writer and bridge reader is structurally impossible. §3, §6.
9. **Member ≠ agent.** Member = roster entry (static); agent = per-turn participant (subset). Rubric writes one `MemberScore` per **member**; schedule defines `slot.agents`. §6.
10. **`KernelState` is frozen.** Mutations happen via `dataclasses.replace` + reassignment of `state["_kernel"]`, never in-place. §6.

If you find yourself wanting to relax any of these, first check that the existing tests exercising the invariant would still be satisfied. The listed tests in §7 are the canonical specs — if a change makes them fail, the change is wrong until shown otherwise.

## 13. `parse_channels` behavior, worked examples

`parse_channels(raw, tag)` is the single function that splits raw model output into `(public_channel, private_channel)`. The rules are enumerated in the docstring but easier to read as a table. Assume `tag="thinking"` throughout:

| Input | `public` | `private` | Notes |
| --- | --- | --- | --- |
| `"hello world"` | `"hello world"` | `None` | No tags, passthrough. |
| `"<thinking>reason</thinking>verdict: A"` | `"verdict: A"` | `"reason"` | Configured tag consumed. |
| `"<think>reason</think>verdict: A"` | `"verdict: A"` | `None` | Native tag stripped, content DISCARDED. |
| `"<thinking>a</thinking><think>b</think>done"` | `"done"` | `"a"` | Pack-configured tag → private; native → discarded. |
| `"<thinking>a</thinking><thinking>b</thinking>"` | quarantine | quarantine | Multiple configured tags → `ContentParseError`. |
| `"<thinking>a<thinking>b</thinking></thinking>"` | quarantine | quarantine | Nested → `ContentParseError`. |
| `"<thinking>unclosed"` | quarantine | quarantine | Unbalanced → `ContentParseError`. |
| `"</thinking>stray"` | quarantine | quarantine | Stray closer → `ContentParseError`. |
| `"abc<thinking>r</thinking>xyz"` (`tag="think"`) | `"abcxyz"` | `None` | `tag` aliases native; one pass handles both. Inner content discarded because native-tag handling discards. |

"Quarantine" means `apply_action` wraps the `ContentParseError` raise: the `Utterance` is constructed with `public_channel=""`, `private_channel=None`, `parse_error=<str-of-exc>`, then commits normally. The `TrajectoryStep` gets `extras["parse_error"]`. The rubric's errored-path counter (`_count_parse_errors` in `debate_rubric.py`) tallies these per member, and downstream the trainer masks the completion tokens via `completion_mask`.

The stripping-native-always rule exists because reasoning-model artifacts (`<think>`, `<thinking>`) are third-party: the model emits them whether or not the pack author asked for them. Leaking native reasoning into opponent view would be a capability leak; surfacing it as `private_channel` on a pack that did not opt in would mislead downstream consumers into thinking the author intended it as private reasoning. Discard is the safe default.

## 14. Atomicity failure modes, enumerated

The 3-phase simultaneous-slot protocol in §5 handles five failure modes. Enumerating explicitly:

**(a) Single agent raises `vf.OverlongPromptError` during generation.**
TaskGroup catches it, cancels siblings, re-raises as `ExceptionGroup`. Flatten + pick `OverlongPromptError` (highest priority). Rollout loop's `except vf.OverlongPromptError:` matches → `state["prompt_too_long"] = True`, `state["is_truncated"] = True`. Parent tracker unchanged (child forks dropped). Trajectory + kernel unchanged. Next iteration of the rollout-loop's `while not is_completed` sees `prompt_too_long=True`, exits, scoring produces an errored MARScore.

**(b) Single agent raises generic `vf.Error` during generation.**
Same path as (a), but `pick` returns the `vf.Error`, which matches `except vf.Error:` → `state["error"] = e`. Next loop iteration sees `error != None` via `has_error` stop (priority 100), exits.

**(c) Multiple agents raise different `vf.Error` subclasses concurrently.**
This is the P0 that motivated the flatten-and-pick-one implementation. If we used chained `except* OverlongPromptError: re-raise; except* vf.Error: re-raise`, BOTH branches would fire, re-wrapping both re-raises into a new `ExceptionGroup` at the call site, which the outer `except vf.Error:` in `rollout` does NOT match — the rollout would crash with an uncaught `ExceptionGroup`. Current implementation catches the whole `BaseExceptionGroup`, flattens nested groups, picks by priority, logs the suppressed peers with the slot id, and re-raises the single chosen exception. Outer handler matches as expected.

**(d) All agents succeed, but staging (phase 2) raises.**
`extract_fields` or `_build_step` (token parsing) can raise. Because phase 2 runs after phase 1 completed but before the publish step, the local `staged_kernel` and `staged_steps` are discarded, `state["_kernel"]` and `state["trajectory"]` are unchanged. Parent tracker: the per-agent forks never merge back in. The raise propagates to the rollout loop's `except vf.Error:` / bubble.

**(e) All agents succeed, staging succeeds, but `apply_action` in phase 2 returns an inconsistent commit count.**
The sanity check `if len(committed_utts) != len(slot.agents)` raises `vf.Error` with the slot id in the message. This should never happen if `apply_action` is correct — it's a defensive assertion against future bugs in the kernel contract. Same cleanup path as (d).

The publish step (phase 3) is deliberately written with no `await` and no constructs that can raise — just list appends, dict assignment, and tracker `merge` calls. If phase 3 ever starts including code that could raise, the atomicity property is broken; reviewers should flag it.

## 15. Visibility policy, concretely

The default `visibility_policy(utt, viewer_id)`:

- Viewer is the utterance author → `"full"` (sees `raw_content`).
- Viewer is anyone else → `"public_only"` (sees `public_channel`).

There is also a `"hidden"` mode, which subclasses opt into — `DebateEnv` returns `"hidden"` for judge utterances when rendering debater views in mid-debate rebuttals, so debaters don't see the judge's private reasoning before producing their own.

Concretely, a subclass's `build_prompt` uses this to filter the visible slice of `state["_kernel"].transcript`:
```python
visible = [
    (utt, self.visibility_policy(utt, viewer_id=member_id))
    for utt in state["_kernel"].transcript
]
rendered = [
    render(utt, mode) for utt, mode in visible if mode != "hidden"
]
```
The monotonic invariant (§4) applies to the rendered sequence: `rendered` at slot N+1 must extend `rendered` at slot N byte-for-byte from member A's perspective. Satisfied when:

- Your `render` function is deterministic: same `(utt, mode)` → same bytes across calls.
- Your `visibility_policy` is stable for already-committed utterances: A's view of utterance U at slot N must equal A's view of U at slot N+k for all k ≥ 0.
- You don't reorder the transcript between slots.

All three are the case for the default policy. If you override visibility, the second point is the one most likely to break — e.g. a policy that depends on the current slot's phase will show different things for the same old utterance at different times, violating monotonicity. Fix: derive visibility from the utterance's own phase/slot_id, not the viewer's current phase.

## 16. File reference

Top-to-bottom list of files implicated by this doc, with one-line purpose statements:

- `verifiers/envs/multi_agent_kernel.py` — `TurnSlot`, `KernelState`, `Utterance`, `ActionResult`, `SlotProgram` protocol, `StaticSchedule`, `parse_channels`, `apply_action`.
- `verifiers/envs/multi_agent_env.py` — `MultiAgentEnv` base class, rollout loop, sequential/simultaneous dispatch, 3-phase atomic commit, stop-condition hooks, `_build_step`, `_prepare_prompt`.
- `verifiers/envs/request_context.py` — `ModelRequestContext` dataclass.
- `verifiers/envs/debate_env.py` — `DebateEnv` concrete subclass (reference implementation for how to extend `MultiAgentEnv`).
- `verifiers/envs/debate_rubric.py` — `DebateRubric` concrete subclass (reference for how to extend `MultiAgentRubric`, W+G+M scoring).
- `verifiers/envs/debate/prompts.py` — `DebatePrompts` declarative prompt pack.
- `verifiers/envs/debate/fields.py` — field extraction DSL (FieldSpec, EnumScoring, classify_enum).
- `verifiers/envs/debate/parsing.py` — XML field extraction used by `DebateEnv.extract_fields`.
- `verifiers/rubrics/multi_agent_rubric.py` — `MultiAgentRubric` base class with error boundary.
- `verifiers/types.py` — `MARScore`, `MemberScore`, `MemberRollout`, `RolloutOutput`, `TrajectoryStep`, all shared TypedDicts + pydantic models.
- `verifiers/multi_agent_bridge.py` — `rollout_to_member_rollouts` projection function.
- `verifiers/utils/save_utils.py` — `state_to_output` wire-boundary serializer; the `mar` branch projects `MARScore` to `output["reward"]` + flat wandb keys + `output["mar_score"]`.
- `verifiers/utils/usage_utils.py` — `StateUsageTracker` with `fork`/`merge` used by the atomic commit path.
- `verifiers/errors.py` — `ContentParseError` (per-utterance quarantine), `KernelProtocolError` (framework invariant violation), `OverlongPromptError`, base `Error`.
- `src/prime_rl/orchestrator/multi_agent_advantage.py` — `compute_rae_advantages`, `RAEState` (EMA baselines keyed by `(task, example_id, member_id)`).
- `src/prime_rl/orchestrator/multi_agent_bridge.py` — thin compatibility shim re-exporting from verifiers.
- `tests/unit/orchestrator/test_multi_agent_env.py` — monotonic prompt test, atomicity tests, stop-condition priority.
- `tests/unit/orchestrator/test_multi_agent_rubric.py` — rubric error boundary tests.
- `tests/unit/orchestrator/test_multi_agent_bridge.py` — bridge projection tests.
- `tests/unit/orchestrator/test_multi_agent_advantage.py` — RAE advantage/EMA baseline tests.
- `tests/unit/orchestrator/test_debate_env.py`, `test_debate_prompts.py`, `test_debate_fields.py` — debate-specific end-to-end + unit tests.

## 17. Common mistakes when building on this stack

Observed failure modes, with the fix upstream rather than the workaround:

**"My new env's prompt-cache hit rate dropped after I added a per-turn summary message."**
You inserted a message whose content depends on slot index into the middle of the rendered prompt. Even if it's appended at position `-2` instead of the end, the fact that its content changes across slots breaks the byte-identical-extension invariant. Fix: make the summary a function of committed transcript only, rendered at the tail of the previously-committed segment, never inserted in the middle.

**"`rollout_to_member_rollouts` is raising `KeyError: 'mar_score'`."**
Your rubric subclass either (a) does not inherit from `MultiAgentRubric`, so `score_rollout` is not writing the `mar_score` field, or (b) your `build_marscore` is raising a non-`vf.Error` exception that bypasses the base class's error boundary and leaves `mar_score` unset. Check whether `state.get("error")` is a `KeyError` / `AssertionError` / `RuntimeError` — those are meant to propagate, and they leave the state in a partial shape. The bridge cannot paper over this and will not try.

**"My simultaneous-slot usage counts are wrong — it's double-counting tokens."**
You are reading `state.usage_tracker` mid-slot. The per-agent fork pattern means the parent tracker is stale until the slot publishes. Read usage only at slot boundaries, or read the child trackers directly if you need mid-slot telemetry (but be aware a cancelled slot drops its children).

**"My test for atomicity keeps flaking — sometimes the trajectory has N steps, sometimes 0."**
The slot is failing nondeterministically (e.g. rate-limit errors surfacing intermittently). The test is asserting 0, but 0 vs. N is exactly the all-or-nothing property: you should see N on success, 0 on any agent raising. If you see anything between 0 and N, that's the P0 — file it.

**"The rubric is returning `episode_scalar=0.0` even when the judge picked a winner."**
Your `outcome_fn` is returning `None` because the latest judge step's `extras["fields"]` is missing `"decision"`. Check the debate `_default_outcome` — it breaks on the FIRST judge step encountered in reverse order (the latest), and returns `None` without falling back to earlier steps. That's intentional: a stale verdict is worse than no verdict. The fix lives upstream in field extraction or the judge prompt, not in the rubric.

**"I want to surface the judge's reasoning as `private_channel`."**
`parse_channels` always strips native `<think>` to `None`, regardless of `think_tag`. If you want reasoning in the private channel, have the judge emit a block with your configured tag (e.g. `<thinking>...</thinking>` when `think_tag="thinking"`). The configured-tag pass populates `private_channel`; the native-tag pass discards.

**"I want to add a new kind of ephemeral per-call metadata (e.g. priority hint, custom header)."**
Extend `ModelRequestContext` with the new field. Thread it from `_run_sequential_slot` and `_run_simultaneous_slot` at the `ModelRequestContext(...)` construction sites. Do NOT store it on `State` — that makes it durable rollout state, which has different serialization and lifetime semantics. The point of `ModelRequestContext` is that it's scoped to one inference call.

## 18. Glossary of Greek

Not all of these appear above; all are worth knowing if you're going to read across the codebase.

- **Member** — roster entry, static per episode. `env.members = ["debater_a", "debater_b", "judge"]`.
- **Agent** — per-turn participant, subset of members scheduled for one `TurnSlot`. `slot.agents = ("debater_a", "debater_b")`.
- **Member** — a single participant in an N-agent episode, identified by `member_id` (e.g. `debater_a`, `debater_b`, `judge`). The kernel schedules members; the rubric emits one `MemberScore` per member; the bridge emits one `MemberRollout` per member. The RAE advantage baseline partitions on `(task, example_id, member_id)`.
- **Slot** — one step in the schedule. Sequential (1 agent, commit immediately) or simultaneous (N agents, barrier commit).
- **Phase** — a human-readable label on the slot (`"opening"`, `"rebuttal"`, `"verdict"`). Used by DebatePrompts to pick the right prompt template. Not load-bearing for the kernel — it's metadata.
- **Transcript** — `kernel.transcript: tuple[Utterance, ...]`. Ordered, immutable, grows on commit.
- **Trajectory** — `state["trajectory"]: list[TrajectoryStep]`. One step per utterance, with prompt + completion + response + tokens + extras. This is what the trainer consumes.
- **Episode** — one full rollout from `init_state` to `render_completion`. Produces one `RolloutOutput`, one `MARScore`, one list of `MemberRollout`s (length = `len(members)`).
- **Lineage key** — prefix-cache partition key. Defaults to `agent_id` in multi-agent rollouts.
- **Branch tracker** — a forked `StateUsageTracker` used by one agent in a simultaneous slot. Merges back into parent on publish, dropped on slot failure.
- **Quarantine** — per-utterance flag set when `parse_channels` raised `ContentParseError`. Rollout continues, but downstream consumers know the channel split is unreliable.
- **Errored MARScore** — uniform zero-reward MARScore with `episode_metrics["errored_rollout"] = 1.0`. Produced by `MultiAgentRubric.build_errored_marscore` on the rollout/scoring error paths.
- **Wire boundary** — the point at which `State` is serialized to `RolloutOutput` (JSON dict). Happens in `state_to_output`. Beyond this point, pydantic models become dicts, objects become primitives, and the orchestrator-side code does not need to import `verifiers.types` beyond the `MemberRollout` TypedDict.
