## Description

This PR consolidates three separate callbacks (`on_start`, `on_progress`, `on_log`) into a unified event-based system, addressing issue #755. The new system uses TypedDict events with Literal discriminators for type-safe pattern matching, making it cleaner, more extensible, and easier to maintain.

### Motivation

The previous callback system had grown brittle with three separate callback parameters that felt "tailor-made" for specific use cases. This PR:
- ✅ Replaces 3 callbacks with 1 unified `on_event` handler
- ✅ Adds support for PR #632 (GroupCompleteEvent with State objects)
- ✅ Provides infrastructure for #753 (log streaming via LogStreamEvent)
- ✅ Makes the system more extensible for future event types

### Before / After

**Before:**
```python
await env.evaluate(
    client=client,
    model="gpt-4",
    on_start=lambda total: ...,
    on_progress=lambda all_outs, new_outs: ...,
    on_log=lambda msg: ...
)
```

**After:**
```python
async def on_event(event: EvalEvent):
    match event["type"]:
        case "start":
            print(f"Starting: {event['num_examples']} examples")
        case "progress":
            print(f"Progress: {event['completed_count']}/{event['total_count']}")
        case "complete":
            print(f"Done! Avg reward: {event['avg_reward']}")

await env.evaluate(
    client=client,
    model="gpt-4",
    on_event=on_event  # Single unified handler!
)
```

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [x] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Test improvement

**Note:** This is a breaking change that removes `on_start`, `on_progress`, and `on_log` parameters. All internal usages have been migrated to the new event system.

## Changes

### New Event Types (`verifiers/types.py`)
- **StartEvent** - Emitted once at start with resolved `num_examples` and `rollouts_per_example`
- **ProgressEvent** - Emitted after each rollout/group completes with `all_outputs` and `new_outputs`
- **GroupCompleteEvent** - For grouped scoring, includes full `State` objects (addresses PR #632)
- **LogEvent** - For log messages with level, source, and timestamp
- **LogStreamEvent** - Infrastructure for streaming logs to files (#753)
- **SaveEvent** - When results are saved (intermediate or final)
- **CompleteEvent** - When generation finishes with timing and metrics

### Event Emission (`verifiers/envs/environment.py`)
- Updated `generate()` and `evaluate()` signatures to use `on_event` parameter
- Added `_emit_event()` helper using `maybe_await()` for sync/async handlers
- Added `_run_group_with_states()` internal method to preserve State objects for GroupCompleteEvent
- Events emitted at all key points: start, progress, group complete, save, log, complete

### Event Consumption (`verifiers/utils/eval_utils.py`)
- Migrated `run_evaluations_tui()` to use `match/case` pattern for event handling
- All metric accumulation logic preserved
- Progress bar, display updates, and logging all driven by events

### Supporting Infrastructure
- **New file:** `verifiers/utils/event_utils.py` - LogStreamFileWriter for log tailing
- **Updated:** `verifiers/utils/eval_display.py` - Comments updated for new event system

## Testing

Comprehensive test coverage demonstrates the system works correctly:

### Unit Tests (`tests/test_event_system.py` - 10 tests)
- ✅ Event type structure validation
- ✅ LogStreamFileWriter functionality (file creation, appending, custom paths)
- ✅ All tests pass

### E2E Scenarios (`tests/test_event_system_e2e.py` - 4 scenarios)
Standalone executable script with realistic integration scenarios:
- ✅ Scenario 1: Simple independent scoring
- ✅ Scenario 2: Grouped scoring with multiple rollouts (tests GroupCompleteEvent)
- ✅ Scenario 3: Intermediate saves (tests SaveEvent emission)
- ✅ Scenario 4: Progress tracking with metrics
- ✅ All scenarios validate event order, data completeness, and counts
- Run with: `uv run python tests/test_event_system_e2e.py`

### Integration Testing
- [x] All existing tests pass when running `uv run pytest` locally (514 tests pass, 4 skipped external env tests)
- [x] New tests have been added to cover the changes (10 unit + 4 e2e + 5 bugfix + 2 immutability = 21 event system tests)
- [x] Verified with real `vf-eval` command - progress bar and TUI work correctly

### Manual Testing
Ran actual evaluation with `vf-eval` using a test environment:
```bash
$ uv run vf-eval test_config.toml
Processing 2 groups (2 total rollouts): 100%|██████████| 2/2 [00:00<00:00]
Evaluation completed in 1.94 seconds
```
✅ Progress bar updates correctly (requires ProgressEvent)
✅ Results display properly (requires CompleteEvent)

## Checklist

- [x] My code follows the style guidelines of this project as outlined in [AGENTS.md](https://github.com/PrimeIntellect-ai/verifiers/blob/main/AGENTS.md)
- [x] I have performed a self-review of my own code
- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation (MEMORY.md added)
- [x] My changes generate no new warnings (only pre-existing experimental uv warnings)
- [x] Any dependent changes have been merged and published (N/A)

## Bug Fixes (Post-Review)

Three issues were identified during code review and have been fixed:

### 1. Server Mode Bypass (HIGH Priority)
**Problem:** When `independent_scoring=False`, grouped scoring bypassed the server mode dispatch in `run_group()`, causing failures in server mode.

**Fix:** Added server mode detection. In server mode, properly routes through `run_group()`. In local mode, uses `_run_group_with_states()` to get State objects for GroupCompleteEvent.

### 2. Incorrect num_examples Calculation (MEDIUM Priority)
**Problem:** When `independent_scoring=True` with `rollouts_per_example > 1`, StartEvent reported incorrect `num_examples` (total rollouts instead of unique examples).

**Fix:** Added `configured_rollouts_per_example` parameter to `generate()`. Now correctly calculates: `num_examples = total_rollouts // rollouts_per_example`.

### 3. Missing Documentation (LOW Priority)
**Problem:** Changes to core user-facing methods weren't documented.

**Fix:** Added comprehensive event type documentation to `docs/reference.md` and usage examples to `docs/evaluation.md`.

**Test Coverage:** Added `tests/test_bugfix_event_system.py` with 5 tests covering all three fixes.

### 4. Mutable Reference in Events (MEDIUM Priority)
**Problem:** ProgressEvent and GroupCompleteEvent stored direct references to mutable lists (`builder.outputs`, `states`, `new_outputs`). When events were stored (e.g., in EventCollector), the lists would silently grow as more results were added, making `all_outputs` misleading.

**Fix:** Copy all list references when creating events: `list(builder.outputs)`, `list(states)`, `list(new_outputs)`.

**Test Coverage:** Added `tests/test_event_immutability.py` with 2 tests verifying events don't mutate after emission.

### 5. Unnecessary O(N²) Event Construction (MEDIUM Priority)
**Problem:** Changed `elif on_progress is not None:` to bare `else:`, which unconditionally creates ProgressEvent objects (including expensive list copies) even when `on_event=None`. This causes O(N²) allocations affecting production code like GEPA.

**Fix:** Use `elif on_event is not None:` to skip event construction when no handler is registered, matching the original callback pattern's performance characteristics.

## Additional Notes

### Design Decisions

**Why TypedDict over dataclasses?**
- Matches existing patterns in the codebase
- JSON-serializable by default
- Works well with Literal discriminators for type-safe pattern matching

**Why break backward compatibility?**
- The previous callback system was acknowledged as "tailor-made/brittle" (issue #755)
- Clean break is simpler than maintaining an adapter layer
- All internal usages migrated in this PR
- Better to do it now than carry technical debt

**State Preservation Strategy**
- Created internal `_run_group_with_states()` method to return both State objects and outputs
- Public `run_group()` API remains unchanged (returns only outputs)
- GroupCompleteEvent receives State objects without breaking existing callers

### Future Work

- Full subprocess log streaming implementation (infrastructure in place)
- Additional event types as needed (e.g., ErrorEvent for failures)
- TUI features that leverage State objects from GroupCompleteEvent

### Context

I'm relatively new to this codebase and the broader Prime Intellect ecosystem, so I focused on:
1. Understanding the existing patterns (TypedDict, maybe_await, etc.)
2. Following established conventions
3. Thorough testing to ensure no regressions
4. Clear documentation for future maintainers

Feedback welcome, especially on areas where I might have missed broader integration concerns!

## Related Issues

- Closes #755
- Addresses PR #632 (GroupCompleteEvent infrastructure)
- Partial implementation of #753 (log streaming infrastructure)
