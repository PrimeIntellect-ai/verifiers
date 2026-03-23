---
name: optimize-environments
description: Audit and optimize verifiers environments for async performance. Use when asked to profile, speed up, or review an environment for concurrency bottlenecks, event loop blocking, or scaling issues under high rollout counts.
---

# Optimize Environment Performance

## Goal
Find and fix synchronous bottlenecks in verifiers environment code so that rollouts scale efficiently under concurrency. The verifiers runtime runs all rollouts on a single async event loop — any sync operation blocks every concurrent rollout.

## Audit Workflow

### 1. Identify Async Entry Points
Locate all async methods in the environment (typically `setup_state`, `env_response`, `score`, `cleanup`, and any tool functions). These are the hot paths where sync operations cause the most damage.

### 2. Scan for Sync Offenders
Search for these patterns inside async methods, ordered by typical severity:

**Critical — blocks network I/O:**
- `time.sleep()` → replace with `await asyncio.sleep()`
- Sync HTTP clients (`requests`, `httpx.Client`, `urllib`) → replace with `httpx.AsyncClient` or equivalent
- Sync LLM clients (`OpenAI()`, `litellm.completion()`) → replace with `AsyncOpenAI()` or use `self.get_model_response()`

**High — blocks on disk or CPU:**
- `open()`, `tempfile.NamedTemporaryFile`, `Path.unlink()`, `Path.read_text()`, `shutil` → offload with `await asyncio.to_thread(...)` or use `verifiers.utils.path_utils.write_temp_file`
- `copy.deepcopy()`, `.model_copy()` on non-trivial objects → offload with `await asyncio.to_thread(...)`
- `json.dumps()`/`json.loads()`, `base64.b64encode()`, `msgpack.pack()` on large payloads → offload with `await asyncio.to_thread(...)`

**Medium — blocks GIL for compute:**
- Heavy computation, data parsing, static analysis, compilation → use `ProcessPoolExecutor`

### 3. Check for Shared Immutable Data
If the environment deep-copies an object with large immutable fields (dictionaries, corpora, config blobs):
1. Build a `deepcopy` memo dict that maps `id(immutable_field)` → `immutable_field` so the field is shared, not copied.
2. Compute the memo once in `__init__` after the object is initialized.
3. Pass `memo.copy()` to each `deepcopy` call.

### 4. Check Upload Patterns
If the environment encodes file content manually (base64, JSON) and sends it inline:
- Prefer the client's native async `upload_file()` method instead.
- Write to a temp file (via `asyncio.to_thread(write_temp_file, ...)`) and upload, rather than encoding large blobs on the event loop.

### 5. Check for GIL-Saturating Work
If any single operation takes >50ms of pure CPU time:
- Move it to a `ProcessPoolExecutor` via `loop.run_in_executor(executor, fn, *args)`.
- Common examples: running linters, compilers, parsers, or large data transforms in reward functions.

## Fix Patterns

### asyncio.to_thread() — the default fix
```python
# Offload any sync function
result = await asyncio.to_thread(sync_function, arg1, arg2)
```
The runtime scales the thread pool to match concurrency. No pool management needed.

### Shared deepcopy memo
```python
@staticmethod
def build_shared_memo(obj):
    memo = {}
    memo[id(obj.large_immutable_field)] = obj.large_immutable_field
    return memo

# __init__:
self.shared_memo = self.build_shared_memo(self.obj)

# hot path:
obj_copy = await asyncio.to_thread(deepcopy, self.obj, self.shared_memo.copy())
```

### ProcessPoolExecutor for CPU-bound work
```python
from concurrent.futures import ProcessPoolExecutor
executor = ProcessPoolExecutor(max_workers=4)

async def heavy_reward(data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, cpu_bound_fn, data)
```

## Findings Format
Report findings sorted by severity:
1. **Critical**: sync network I/O (HTTP, LLM clients, sleep) in async methods.
2. **High**: sync disk I/O, large deepcopy/serialization in async methods.
3. **Medium**: GIL-saturating CPU work inline.
4. **Low**: small sync operations that are technically blocking but negligible in practice.

## Verification
After applying fixes, verify with a concurrency stress test:
```bash
prime eval run <env> -m gpt-4.1-mini -n 64 -r 32 -c -1 -s
```
Compare wall-clock time and the event loop lag which is periodically logged from the env server.
