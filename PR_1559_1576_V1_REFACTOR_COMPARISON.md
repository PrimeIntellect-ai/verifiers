# PR 1559 and PR 1576 v1 Refactor Comparison

Source basis: PR 1559 head `5d1ef09e19cfb0866741eb6c0c40de269d58fdd2` and PR 1576 head `d4b054e32360c4a8274e5ab6f7ea8c4c4b298958`, inspected on June 15, 2026.

PR 1559 is a broad refactor of the existing v1 module and its environment ecosystem. It keeps a recognizably Verifiers-shaped user surface around `Taskset`, `Harness`, `Env`, `State`, lifecycle decorators, MCP `Toolset`s, and the existing `vf-eval` integration. It ports a large set of existing v1 examples and package integrations into that shape.

PR 1576 replaces the v1 module with the vf-nano architecture. It centers the implementation on `Environment`, `Taskset`, `Harness`, `Rollout`, `Episode`, `Trace`, explicit runtime/client abstractions, provider-native interception, and a new v1 CLI/server path. It also includes a v0 legacy bridge and examples that exercise training-relevant trace graphs, token/logprob preservation, shared tools, user simulators, and agent harnesses.

## Difficulty Legend

The "add difficulty" classifications are relative to the target PR's current architecture.

- `Straightforward`: local API or adapter work with little change to execution ownership.
- `Moderate`: requires touching several framework components, but the target architecture already has a clear home for the behavior.
- `Significant`: requires changing the execution model, output contract, or core object ownership.
- `Drop candidate`: functionality appears duplicative, stale, or not worth carrying unless there is a concrete downstream consumer.

## Unification Requirements

- Hidden arguments to tools and users are required.
- Read/write scratchpads are required for key environment patterns. The public `State` versus `Trace` decision is downstream of the hidden-argument and scratchpad model.
- Python 3.10 support is dropped. Heavy training-only dependencies such as `renderers`, `transformers`, and `torch` stay optional rather than becoming core install requirements.
- The old loader shape and empty runtime stubs are not part of the target API.
- Advantage computation belongs in prime-rl algorithm modules. The v1 environment surface supports user-extensible, env-configurable algorithms without forking prime-rl, and the rollout object includes fields for algorithm-provided advantages.
- Reward weights and decorator priorities are preserved. The forcing use case for reward priorities is expensive state-writing work, such as judge queries, that runs once and feeds multiple downstream reward components.
- Multiple toolsets are required. The mental model is `Toolset = MCP server`, where one toolset can expose multiple tools. Prefer the public name `Toolset` over `Tools`.
- Train/eval split is preserved.
- `init_group` is deferred behind a clean `expand_group` extension point.
- An init template is required for v0 envs, v1 taskset-only envs, v1 taskset-plus-harness envs, and optional tool/user stubs. Generated subcomponents are local by default, either in contained files or subfolders.
- End users run `prime eval run`, not `eval` or `vf-eval`. The Prime CLI bridge preserves one top-level UX across v0 and v1, displays full helper commands, supports auto-installation, and supports platform uploading.
- Rich system-prompt merge strategies are supported.
- Every existing v1 environment pattern needs a clean migration into the 1576-style implementation.
- The `environments/` folder remains the home for examples, including both tasksets and harnesses. Third-party harness authoring, especially for Python agent frameworks such as DSPy and LangChain DeepAgents, is clear and symmetric with taskset packaging.
- Independent harness execution with optional runtime reuse is required. The simple target API is roughly `harness.run(task="Write 'hello world' backwards.", model="openai/gpt-5")`.

## Shared Or Near-Equivalent Design Features

Both PRs provide the following design features, with equivalent or near-equivalent intent.

### New v1 Module Boundary

- Both replace the old experimental v1 implementation with a new `verifiers.v1` package that is intended to be the public v1 surface.
- Both expose first-class taskset and harness concepts rather than treating environments as one monolithic class.
- Both make environment packages load through typed `load_taskset(...)` and `load_harness(...)` entrypoints.
- Both keep a top-level environment loader that combines a taskset and optional harness into a runnable evaluation object.
- Both use Pydantic configs for public structured configuration.
- Both lean on explicit IDs for tasksets, harnesses, and environments.
- Both use strict config and data models in core user-facing types.
- Both remove large portions of the previous v1 module rather than layering a compatibility wrapper over every old path.

### Taskset And Harness Split

- Both make the taskset own task data and task-specific scoring logic.
- Both make the harness own the execution mechanism for an agent, program, command loop, framework adapter, or runtime-backed interaction.
- Both support a default/simple harness path for single-turn or plain chat-style tasks.
- Both support custom harness implementations for more complex agentic tasks.
- Both allow harnesses to run in a runtime rather than only in the Python process that loaded the environment.
- Both allow tasksets to be reused with different harnesses in principle.
- Both treat tasksets and harnesses as separately configurable components.

### Task Records

- Both introduce a typed task object instead of relying on arbitrary row dictionaries at the execution boundary.
- Both support text instructions/prompts.
- Both support system prompts at task or component level.
- Both support image or multimodal task metadata in some form.
- Both support per-task runtime resource hints in some form.
- Both support task identifiers that are stable enough to appear in outputs and traces.
- Both convert taskset data into canonical task objects before rollout execution.

### Rollout/Episode Execution

- Both make a rollout a first-class execution unit.
- Both have a per-rollout context object that carries task, runtime/client state, trace/state output, and cleanup responsibilities.
- Both capture stage timing.
- Both capture errors into the rollout output rather than relying only on thrown exceptions.
- Both support grouped execution for group rewards or multi-sample scoring.
- Both allow multiple rollouts for the same task to be scored together.
- Both have a cleanup/finalization path that runs even when generation or scoring fails.

### Scoring

- Both support reward decorators.
- Both support metric decorators.
- Both support group-level reward behavior.
- Both allow taskset-owned scoring to inspect the generated interaction.
- Both allow harness-owned execution metrics.
- Both support a numeric reward as the canonical optimization/evaluation target.
- Both support named metrics and named reward components, even though the output shape differs.
- Both support stopping based on generated content, task state, or harness conditions.

### Tool Use

- Both support MCP-style tools.
- Both support task-owned tools.
- Both support tool servers running in a runtime.
- Both support colocated tools that run next to the agent/harness.
- Both support remote tool servers via URL/headers.
- Both support sharing an expensive tool server across more than one rollout in some form.
- Both support passing tool server connection data into the agent execution path.
- Both include examples that exercise search/game/tool tasks.

### User Simulation

- Both support user simulators as a first-class part of v1.
- Both model user simulation through MCP-compatible server behavior.
- Both allow a taskset to decide whether a task has a user simulator.
- Both can use user simulation for multi-turn environments such as games or interactive tasks.
- Both have examples around TextArena/Wordle-style interaction.

### Runtimes

- Both provide runtime configuration objects.
- Both support subprocess execution.
- Both support Docker execution.
- Both support Prime sandbox execution.
- Both have at least a stub or implementation for Modal.
- Both allow runtime selection at harness/tool/taskset level in some form.
- Both support background process management.
- Both expose URLs/ports for agents and tools.
- Both include runtime read/write helpers.
- Both include cleanup behavior for processes or sandboxes.

### Provider/Model Interaction

- Both support OpenAI-compatible chat-completion-style model calls.
- Both include provider/client abstractions rather than hard-coding a single SDK call at every generation site.
- Both support multiple protocol or dialect families in the v1 module.
- Both preserve enough request/response structure to reconstruct messages.
- Both have a place to attach token usage and model timing.

### CLI And Evaluation Integration

- Both intend v1 to run through a standard eval path rather than ad hoc Python scripts.
- Both support TOML/Pydantic configuration loading.
- Both support taskset and harness config composition from CLI/config.
- Both can produce persisted eval outputs.
- Both include examples/configs for v1 environments.
- Both update package metadata and scripts around v1.

### Example And Package Coverage

- Both migrate or add example tasksets under the repository's package structure.
- Both include TextArena/Harbor-related v1 work.
- Both include Wiki/Search-style examples.
- Both include command/program/agentic harness examples.
- Both include tests that exercise local subprocess and runtime behavior.
- Both include tests or examples for multi-turn user simulation and tools.

## Shared Usage Patterns With Different UX

| Pattern | PR 1559 UX | PR 1576 UX | Practical Difference |
| --- | --- | --- | --- |
| Constructing an environment | `vf.Env(taskset, harness=None, runtime=None, advantage="rl")`, with generated `load_environment(config: vf.EnvConfig)` shims | `vf.Environment(config)` loads taskset/harness from `EnvConfig` plugin IDs or explicit configs | 1559 feels like an extension of existing Verifiers Python APIs; 1576 is config/plugin/server first. |
| Loading task data | `Taskset.load_tasks(split: Literal["train", "eval"])` | `Taskset.load_tasks()` | 1559 has train/eval split as a taskset API. 1576 currently treats split selection as outside or inside taskset config. |
| Task prompt fields | `Task.prompt` stores messages; `Task.system_prompt` stores a `SystemPrompt` | `Task.instruction` accepts text or messages; `Task.system_prompt` is optional text | 1559 makes prompt/transcript shape central. 1576 keeps a smaller task shape and lets harness prompt resolution adapt. |
| Task IDs | `task_id`, `row_id`, deterministic hash fallback, aliases for `id`/`example_id` | required `idx` plus plugin/runtime IDs elsewhere | 1559 optimizes for dataset/eval row continuity. 1576 optimizes for simple task indexing and trace serialization. |
| Per-task resources | `Task.resources`, `Task.image`, runtime config merging | `Task.workdir`, timeout fields, CPU/memory/GPU/disk fields | Both support resource hints, but 1576 exposes more per-stage timeout/resource fields directly on `Task`. |
| Core output object | `State` with transcript, extras, metrics, rewards, artifacts, token usage, and `to_output(...)` | `Trace` with message graph, branches, rewards, metrics, info, errors, token/logprob data | 1559 is closer to old `RolloutOutput`. 1576 is closer to a training trace and provider relay log. |
| Message history | Linear `State.transcript: list[Turn]` | Graph of `MessageNode`s with `Branch` views | 1559 is simpler for eval authors. 1576 supports branches, subagents, compaction, token attribution, and training masks. |
| Metrics and rewards | Decorators operate on `Task`, `State`, `Context`, groups; weighted reward components and optional advantages | Decorators operate mainly on `Trace` and runtime; group rewards operate over `list[Trace]` | 1559 scoring API is richer in lifecycle scope. 1576 scoring API is smaller and trace-oriented. |
| Lifecycle hooks | `@setup`, `@update`, `@cleanup`, `@teardown`, `@stop`, `@metric`, `@reward`, `@advantage`, with rollout/group stages and priorities | `Taskset.setup`, `Taskset.finalize`, `Taskset.validate`, `@stop`, `@metric`, `@reward`, `@group_reward`; harness `launch/run` | 1559 offers a decorator-heavy lifecycle. 1576 exposes fewer lifecycle concepts with stronger stage ownership. |
| Group rollouts | `Taskset.init_group(task, num_rollouts) -> tasks, states`; `Group` object runs and scores | `Environment.episode(task, ctx, n)` creates `n` `Rollout`s; `@group_reward` receives traces | 1559 can mutate or specialize each rollout's task/state before generation. 1576 assumes repeated rollouts for the same task, then group scoring. |
| User simulation | `User` subclasses `Toolset`; `@vf.user` hidden `respond` tool can bind task/state args and update state with `sets`/`extends` | `User` is a `Tools` server; `serve_user` returns `respond(message) -> (Messages, done)` and interception drives it | 1559 user simulation is integrated with state mutation. 1576 user simulation is simpler and program-facing. |
| Tools | `Toolset` classes with `@vf.tool`, `@vf.resource`, visibility config, hidden/bound args, automatic state updates | `Tools` dataclass with script/command/env/url/headers; harness/program receives MCP URLs | 1559 owns individual tool-call semantics. 1576 mostly brokers tool servers to programs. |
| Multiple toolsets | Multiple configured `Toolset`s can be attached through taskset config/task visibility and each can expose multiple tools | `Taskset.tools(task)` can return multiple `Tools` server specs | Both support the shape. The unified API uses the name `Toolset` and the mental model `one Toolset = one MCP server exposing many tools`. |
| Shared tools | Toolset `scope="env"` and env-scope registries | `ToolsConfig(shared=True)` and shared servers | Both support expensive shared tools. 1576 has a simpler explicit config; 1559 has richer parent/child registry behavior. |
| Tool visibility | Task-level `toolsets`, `tools`, `VisibilityConfig`, enabled/show/hide | Taskset methods decide `tools(task)`; harness capability flags determine support | 1559 has declarative visibility. 1576 expects taskset code to decide which tools to expose. |
| Dynamic tools | `setup` can return dynamic tool definitions/routes; registry exposes them | Not a first-class pattern; taskset can choose tools before serving, but not framework-owned dynamic per-state tools | 1559 supports dynamic tool availability during rollout. 1576 requires a proxy or reload mechanism for equivalent behavior. |
| System prompts | `SystemPrompt`, `SystemPromptConfig`, strategies such as harness-task composition | Task/harness prompt resolution, capability flag for whether harness appends system prompts | 1559 has a larger prompt composition abstraction. 1576 has fewer knobs and warning-based adaptation. |
| Independent harness execution | Harnesses can be run through an env/context and some examples call harness logic directly | Harnesses own `launch/run` over a task/runtime, but ergonomic standalone execution is not yet the main UX | The unified surface supports direct calls such as `harness.run(task="...", model="...")` and optional runtime reuse. |
| Harness capability declarations | Implicit through base methods, protocols, system-prompt strategies, and runtime config | Explicit flags such as `APPENDS_SYSTEM_PROMPT`, `SUPPORTS_TASK_TOOLS`, `SUPPORTS_USER_SIM`, `SUPPORTS_MESSAGE_INSTRUCTION` | 1576 makes harness compatibility fail early. 1559 relies more on composition and lifecycle methods. |
| Model protocols | Protocol classes such as OpenAI chat/completions/responses and Anthropic messages | Dialects for chat/responses/anthropic plus raw provider relay and training client rendering | 1559 is protocol-extensible. 1576 is provider-relay and training-data oriented. |
| Runtime selection | Runtime configs can be set on taskset, harness, toolset, user, env; provider conflict checks | Runtime configs for harness, tools, user, task needs; `Runtime` implementations in subpackages | Both have multi-level runtime config. 1576's runtime is more tightly tied to rollout/server/client execution. |
| End-user eval entrypoint | Existing `vf-eval` path and `verifiers/scripts/eval.py` integration | New `eval`, `validate`, `serve` v1 CLIs plus legacy v0 bridge | Neither PR exposes its internal eval command as the final user-facing path. The unified UX is `prime eval run` for both v0 and v1, with different supported flags where necessary. |
| Config plugin resolution | Existing v1 loaders and import helpers; generated package entrypoints | `EnvId` plugin IDs, built-ins, local modules, Hub install-on-demand, annotation-based config narrowing | 1576 has a more ambitious plugin story. 1559 is simpler and closer to current repository patterns. |
| Error handling | `State.capture_error`, state output errors, lifecycle cleanup | Typed `RolloutError` classes, retry classification, `Trace.errors` | 1576 has a stronger typed error/retry story. 1559 has state-compatible output capture. |
| Retry behavior | Eval-level `max_retries` and current eval machinery | `RetryConfig` for model, runtime calls, and rollout retries | 1576 has more granular retry semantics. |
| Validation without model calls | Mainly load/import/runtime smoke paths and tests | `Taskset.validate(task, runtime)` and `validate` CLI | 1576 has a first-class validation mode. |
| v0 compatibility | Keeps existing v0 compatibility by not replacing v0 eval core, but v1 output maps into current `RolloutOutput` | Explicit legacy bridge runs v0 envs and maps outputs into v1 `Trace` | 1576 has the clearer single new output story across v0/v1. |
| Training readiness | `@advantage`, token usage, and old output compatibility | Trace graph, sampled masks, token IDs/logprobs, routed experts, renderer model selection, train/eval clients | 1559 has env-level algorithm hooks. 1576 is closer to a training data plane, but heavy training dependencies must stay optional. |

## Usage Patterns Supported Only In PR 1559

| Pattern | 1559 Support | Add To 1576 | Recommendation |
| --- | --- | --- | --- |
| Existing v1 generated environment shape with `load_environment(config: vf.EnvConfig)` shims | Environment packages keep generated `load_taskset`, optional `load_harness`, and `load_environment` functions around `vf.Env`. | `Moderate` for a thin compatibility shim; `Significant` for exact old package behavior. | Add a transitional loader for existing package migration. The old generated shape is not the primary 1576 authoring API. |
| `Taskset.load_tasks(split=...)` | Split is part of the taskset API and examples use train/eval values directly. | `Moderate` | Add split support to 1576 as a standard taskset data-loading contract. The exact shape can be an optional `split` parameter or a standard config field consumed by `load_tasks()`. |
| Eval fallback when eval split is absent | 1559 preserves the expected empty-eval fallback warning path through current eval machinery. | `Moderate` | Port the behavior at the 1576 eval/task loading boundary. Keep missing eval data absent; normalize only at the runner boundary. |
| Direct Python construction with `vf.Env(taskset, harness)` | Authors can instantiate and run an environment without config/plugin resolution. | `Moderate` | Add a convenience constructor or `Environment.from_components(...)` for interactive use. Keep `Environment(config)` as the canonical server/CLI path. |
| `EnvRun` scoped execution object | `env.run()` returns an `EnvRun` that owns env-scope resources and can run tasks, contexts, and groups. | `Moderate` | 1576 has `RolloutContext`, `Episode`, shared tools, and interception pools. Add a narrow context manager facade for notebooks/tests. |
| `State` as the author-facing mutable rollout object | Metrics, rewards, tools, users, and harnesses read/write `State`. `State.to_output(...)` maps to existing `RolloutOutput`. | `Significant` | Sequence the State-vs-Trace decision after hidden args and read/write scratchpads. A Trace-primary design still includes a first-class mutable scratchpad or `StateView` for environment authors. |
| Linear transcript as the primary authoring model | `state.transcript: list[Turn]` is the canonical history. | `Significant` as primary; `Straightforward` as a read-only view. | Keep ergonomic linear transcript access for simple env authors regardless of the final persisted object. This can be a branch view. |
| `state_columns` mapping into output | Authors can lift selected state fields into top-level output columns. | `Moderate` | Add output projection in 1576 eval writer for downstream dashboards that depend on this. Prefer explicit scratchpad/info and metric fields for new APIs. |
| Weighted reward components | `@vf.reward(weight=...)` can combine reward components into the final reward. | `Moderate` | Required. Keep reward weights as a first-class scoring feature rather than asking each environment to hand-roll aggregation. |
| `@vf.advantage` hooks and `Env(advantage="rl")` | Environments can compute advantages in the v1 layer. | `Significant` | Fold this into the prime-rl algorithm-module PR rather than keeping 1559's exact API. Preserve user-extensible, env-configurable algorithms without requiring users to fork prime-rl, and include rollout/trace fields for algorithm-produced advantages. |
| Group initialization with different tasks/states per rollout | `Taskset.init_group(task, n)` can create different rollout tasks and prefilled states for one group. | `Significant` | Defer `init_group` and preserve a clear `expand_group(task, n)` extension point. Avoid baking mutable group state into the first unified API. |
| Group-stage metrics/update/cleanup over mutable states | Decorators can run at `stage="group"` and mutate/read grouped states. | `Significant` | Add group metrics/rewards over traces first. Add group update/cleanup for migrated environments that require group mutation before scoring. |
| Decorator priorities | 1559 supports priority-ordered decorators, including reward components that can run before downstream reward components. | `Moderate` | Required. Reward priorities need to support expensive operations that write scratchpad/state in-place once, such as judge queries, and are reused by multiple downstream reward components. |
| Full lifecycle decorator set | `@setup`, `@update`, `@cleanup`, `@teardown`, `@stop`, `@metric`, `@reward`, `@advantage`, priorities, rollout/group stages. | `Moderate` for `setup/update/cleanup`; `Significant` for full parity. | Keep priority support for decorators, especially rewards. Be selective about which lifecycle stages need full decorator machinery. |
| Decorator handler registration through named class methods | Taskset/harness methods decorated and discovered by the framework. | `Moderate` | 1576 already has decorators for scoring/stopping. Extend consistently and keep deterministic priority ordering. |
| Teacher model client in lifecycle/scoring contexts | Harness context can include a teacher client. | `Moderate` | Add a secondary client field to 1576 context only when an environment needs in-framework judging. Otherwise examples can call `vf.resolve_client(...)` explicitly. |
| Typed extras schema/defaults | 1559 has `Extras`, extras schemas, defaults, and state serialization checks. | `Moderate` | Add typed `Trace.info` schemas or taskset config-owned info models for environment author validation. Do not widen core types with arbitrary raw mappings in public signatures. |
| `Toolset` class as a public authoring abstraction | Authors define tools/resources as methods on a `Toolset` subclass with `ToolsetConfig`. | `Significant` | This is required. Use `Toolset` as the public name and model it as an MCP server that can expose multiple tools. 1576's `Tools` dataclass can become the internal/simple server spec. |
| `@vf.tool` method decorator | Supports schema inference, hidden args, bindings, `sets`, `extends`, return normalization, and visibility metadata. | `Significant` | Port for compatibility with 1559's env zoo. It requires the framework to observe and mediate individual tool calls. |
| Hidden/bound tool arguments | Tool schemas hide context-bound args such as task, state, extras, runtime handles. | `Significant` | Required. Implement through a framework-owned MCP proxy or equivalent mediation layer. Direct MCP URL brokering is insufficient for hidden framework context. |
| Tool `sets` and `extends` state updates | Tool results can update `state.extras`, artifacts, metrics, reward, completion, stop condition, and latest turn reward. | `Significant` | Required because read/write scratchpads are needed. The exact storage target follows the State-vs-Trace decision, but tools/users need a clean way to read and write rollout-local scratchpad data. |
| Dynamic setup-provided tools | Setup handlers can return dynamic tool definitions/routes for a rollout. | `Significant` | Add only after the proxy exists. Without a proxy, 1576 can only choose servers before harness launch. |
| Task-level toolset/tool visibility config | `Task.toolsets`, `Task.tools`, `VisibilityConfig`, `show`/`hide`/`enabled`. | `Moderate` | Add declarative filtering as a thin layer over `Taskset.tools(task)` for authoring ergonomics. 1576 can already implement conditional tools in code. |
| Tool resources | `@vf.resource` and `Resources` objects can expose non-tool MCP resources. | `Moderate` | Add resource exposure for MCP-resource-consuming harnesses. Tool calls remain the higher-priority path. |
| MCP server response normalization into messages/content | `ServerResponse` and registry handling can turn server responses into transcript messages and updates. | `Significant` | Proxy-driven tool mediation is the right home. This does not leak into every harness. |
| User subclass of `Toolset` | User simulation uses the same hidden binding and state update system as tools. | `Significant` | User hidden args are required just like tool hidden args. The unified user API can remain distinct, but it needs the same scratchpad/update capability when an env pattern needs it. |
| Empty-prompt user bootstrap | The user simulator can produce the first message when no model prompt exists. | `Moderate` | Add to 1576 interception flow. It is a useful pattern for user-led conversations and game environments. |
| User response can update state and stop condition | `@vf.user(..., sets=..., extends=...)` can mutate state while responding. | `Significant` | Prefer a 1576 pattern where the simulator writes outcome files or response metadata, then `finalize` reads them. Add direct rollout-object updates for migrated envs that require them. |
| System prompt as messages/config/path with merge strategies | `SystemPrompt`, `SystemPromptConfig`, and `SystemPromptStrategy` support richer prompt composition. | `Straightforward` for strings/files; `Moderate` for strategies. | Rich merge strategies are supported while harness capability checks remain explicit. |
| Harness nesting through `Context` | A harness/tool can run another harness context and reject nested scored runs. | `Significant` | 1576's graph branches/subagents are the better primitive. Implement nested harnesses as branches rather than copying `Context`. |
| State update hooks after each turn | `@update` handlers can inspect and mutate state during rollout. | `Moderate` to `Significant` depending on timing. | Add a post-turn hook to 1576 for examples that need online state updates. Prefer trace construction plus `finalize` for simpler tasks. |
| `ReplayTaskset` and `ReplayHarness` | Replays local/HF JSONL conversations into `State.transcript`. | `Moderate` | Port to 1576 as a replay harness that constructs the unified rollout record, including trace branches in a graph-backed output. This is a useful fixture and debugging tool. |
| BFCL, Tau2, NemoGym, OpenEnv, OpenReward ports | 1559 migrates several existing env families into v1. | `Significant` for full parity; some individual tasksets are `Moderate`. | Use these ports as compatibility tests for 1576. Preserve behavior without assuming exact API preservation. |
| OpenAI Agents, DSPy, LangChain, Opencode-style harness examples | 1559 includes framework/program harness ports beyond the 1576 default/Codex/RLM set. | `Moderate` to `Significant` per framework. | Port only the harnesses that represent active product commitments. Keep the rest as examples after the core settles. |
| Runtime provider injection API | 1559 exposes `RuntimeProvider` and `make_runtime_provider`. | `Moderate` | Add this extension point for external runtime provider authors. 1576's runtime classes cover the built-in paths. |
| Daytona runtime config stub | Included with other runtime config types. | `Straightforward`, but `Drop candidate` | Drop this stub until there is a working Daytona integration. Stubs add surface area without behavior. |
| Python 3.10 compatibility | 1559 remains closer to current repo compatibility. | `Significant` if 1576 has already moved to 3.11-only assumptions. | Drop Python 3.10 support. It is close enough to EOL that it does not constrain v1. |
| Existing `vf-eval` path as primary v1 runner | 1559 plugs into current eval script and output machinery. | `Significant` if made primary; `Moderate` as compatibility command. | Do not make `vf-eval` the user-facing path. Preserve `prime eval run` as the public bridge over v0 and v1, with helper commands, auto-installation, uploading, and v1-specific flags where needed. |
| `vf-init --v1` generated templates | 1559 updates the generated v1 environment structure. | `Moderate` | Required. The unified template supports v0 envs, v1 taskset-only envs, v1 taskset-plus-harness envs, and optional tool/user stubs with local-by-default subcomponents. |

## Usage Patterns Supported Only In PR 1576

| Pattern | 1576 Support | Add To 1559 | Recommendation |
| --- | --- | --- | --- |
| Trace graph output | `Trace` stores message nodes, branches, metrics, rewards, errors, info, timing, token/logprob data, and wire serialization. | `Significant` | This is one of the strongest reasons to use 1576's architecture as the base. The final State-vs-Trace public object follows the scratchpad and hidden-argument design. |
| Branch views over a message graph | `Branch` recovers root-to-leaf paths for compaction, subagents, and training spans. | `Significant` | Do not retrofit lightly. 1559's linear transcript is easier for authors but weaker for training. |
| Token IDs, logprobs, sampled masks, multimodal payloads, routed experts | Graph and train client preserve training-relevant per-token metadata. | `Significant` | This belongs in the core output model. Adding it to 1559's `State` is invasive and awkward. |
| Provider-native raw request relay | `EvalClient` forwards raw provider JSON and mutates only model/sampling fields while preserving raw responses for programs. | `Significant` | 1559 protocol classes would need a major rewrite to match this. Use 1576's relay model for provider fidelity. |
| Streaming SSE relay | Dialects parse streaming provider responses while preserving trace data. | `Significant` | Port only as part of a larger client/interception redesign. |
| Anthropic/OpenAI dialect registry with aux routes | Chat, Responses, Anthropic, and Anthropic count_tokens-style aux handling are dialects. | `Moderate` to `Significant` | 1559 can add protocols, but the raw relay/dialect model is cleaner in 1576. |
| Reasoning passback support | Chat dialect handles reasoning fields and passback. | `Moderate` | Port through 1576's dialect model. |
| Train client with renderers/vLLM tokenization | `TrainClient` renders messages/tool calls for generation and records token metadata. | `Significant` | This is central to training and lives behind optional extras. Core v1 does not require `renderers`, `transformers`, or `torch`. |
| `renderer_model_name` for LoRA/base tokenizer mismatch | Client config pins renderer tokenizer separately from serving model. | `Moderate` | Useful feature, but it depends on the train client path. |
| Session-affinity header | Client sends session ID headers for prefix reuse. | `Moderate` | Could be added to 1559 model calls, but trace/session ownership is clearer in 1576. |
| Env server process | `serve` command, ZMQ/msgpack protocol, worker pool, `run_rollout`, `run_group`, `info`. | `Significant` | This is another major reason to use 1576 as base. Retrofitting into 1559 creates a parallel runner. |
| Static and elastic worker pools | 1576 can multiplex eval work through server pools. | `Significant` | Preserve with the 1576 server architecture. |
| Interception pool with multiplexing | `Environment.interception_pool()` shares interception servers/tunnels. | `Significant` | 1559 has interception per context but not the same pooled server model. |
| v0 legacy bridge to v1 `Trace` | Legacy v0 envs run through `LegacyEnvServer` and map outputs to v1 traces. | `Significant` | Preserve this for product continuity and a single v0/v1 runner. |
| New v1 CLIs: `eval`, `validate`, `serve` | Dedicated commands for v1 eval, model-free validation, and serving. | `Significant` | Keep these as internal/helper commands where useful. The end-user eval surface remains `prime eval run` across v0 and v1. |
| Config plugin IDs and Hub install-on-demand | IDs like `name`, `org/name`, `org/name@version`; built-ins and local modules resolved by loader. | `Moderate` to `Significant` | Keep this public surface and port loader IDs without mixing redundant import paths. |
| Annotation-based config narrowing | Loader annotations narrow taskset/harness config types. | `Moderate` | Worth preserving in a unified implementation. It gives good UX with typed configs. |
| `Taskset.validate(task, runtime)` | Model-free validation path is first class and powers CLI. | `Moderate` | Add to the unified taskset-owned behavior model. |
| `Taskset.finalize(task, trace, runtime)` | Tasksets can inspect runtime artifacts after harness execution and before scoring. | `Moderate` | 1559 can approximate with cleanup/update hooks, but finalize is a cleaner explicit stage. |
| Stage-specific task timeouts | Task fields include setup, harness, finalize, and scoring timeouts. | `Moderate` | Straightforward conceptually, but must be threaded through lifecycle execution. |
| Structured retry config | Model, runtime, and rollout retries are separate with include/exclude exception filters. | `Moderate` | Preserve structured retry config instead of eval-level `max_retries`. |
| Runtime call retry helpers | Runtime operations use retry config. | `Moderate` | Useful regardless of base PR. |
| Shared tool config with `colocated`/`shared` mutual exclusion | `ToolsConfig` gives one obvious choice for colocated, shared, or own-runtime tools. | `Straightforward` | Keep the simple colocated/shared/own-runtime UX, but expose it under `Toolset` naming. |
| `Tools` dataclass with script/command/url | Minimal tool-server object that maps directly to process/remote execution. | `Straightforward` | Rename or wrap this as a simple `Toolset` server spec. Public API keeps one mental model rather than splitting between `Tools` and `Toolset`. |
| Program harness contract around `launch(...)` | Harnesses launch commands/programs and treat nonzero exit as `ProgramError`. | `Moderate` | 1559 has command/program harnesses, but 1576's base contract is cleaner for agentic programs. |
| Default bash/agent harness | 1576 includes a default harness for agentic shell-style tasks. | `Moderate` | Preserve as a core agentic harness example. |
| Codex harness | 1576 includes Codex harness support. | `Moderate` to `Significant` | Preserve when Codex evaluations are a first-class target. |
| Compact harness | 1576 includes a compact harness example. | `Moderate` | Preserve as branch/compaction coverage. It depends on trace graph to be maximally useful. |
| RLM no-root harness fixes | 1576 carries RLM harness work. | `Moderate` | Port per product need. |
| Task `workdir` | Task can carry a working directory. | `Straightforward` | Preserve for coding/task sandboxes. |
| Runtime PEP 723 `run_uv_script` helper and interpreter caching | Subprocess runtime can execute inline scripts efficiently. | `Moderate` | Useful to port into any final runtime implementation. |
| Subprocess runtime process-group cleanup and API-key filtering | 1576 hardens local process management and environment sanitization. | `Moderate` | Preserve in the unified runtime. |
| Docker host-network/resource behavior | 1576 Docker runtime includes resource and networking choices aligned with tool/harness execution. | `Moderate` | Compare with 1559 Docker behavior and merge the stricter implementation. |
| Prime runtime creation limiter, labels, timeout, atexit cleanup | 1576 has production-oriented Prime runtime safeguards. | `Moderate` | Preserve in the unified Prime runtime. These are implementation improvements, not UX conflicts. |
| Run output directories and resume dashboard | New eval runner handles output dirs, resume, and rich progress. | `Moderate` | Keep as runner functionality behind the `prime eval run` bridge. |
| Example tasksets: Code Golf, ScaleSWE, R2E Gym, SWE Lego, Terminal Bench 2, DeepWiki, Wikispeedia, GSM8K/AIME, etc. | 1576 adds many examples around agentic and training-relevant tasks. | `Moderate` to `Significant` per taskset | Port examples based on strategic coverage. They are also good integration tests for trace/runtime/client behavior. |
| Python 3.11+ dependency stance | 1576 moves v1 deps such as `renderers` into the base and requires modern Python. | `Straightforward` as policy; `Significant` under a strict release-compatibility constraint | Keep Python 3.11+. Move heavy training dependencies out of core and behind extras/import boundaries. |

## Recommendations For Updating PR 1576 To Cover PR 1559 Features

Use PR 1576 as the architectural base for the long-term v1 runtime/server/provider direction. Its provider relay, v1 server, retry model, legacy bridge, and training-oriented trace work are harder to add after the fact than 1559's authoring conveniences. The update plan ports the 1559 features that represent real environment behavior, not every 1559 API shape.

### Keep 1576 Core Concepts Intact

- Keep `Environment`, `Episode`, `Rollout`, `RolloutContext`, and the env server as the execution backbone.
- Keep provider-native relay, dialects, and graph/token metadata as core infrastructure.
- Keep training clients and renderer-backed tokenization behind optional extras/import boundaries.
- Keep the 1576 CLI/server model available for implementation and debugging, but expose `prime eval run` as the public eval entrypoint.
- Keep explicit harness capability flags; they are useful fail-fast checks.
- Sequence the final public output/scratchpad object decision after hidden args and read/write scratchpad semantics.

### Add Compatibility Where It Preserves Real Existing Behavior

- Add a standard split mechanism. The direct shapes are `Taskset.load_tasks(split: Literal["train", "eval"] = "eval")` or `TasksetConfig.split` with `load_tasks()` reading config. The first is closer to 1559; the second fits 1576's config-driven style. Train/eval split becomes a documented golden path.
- Add an eval fallback for absent eval data at the runner boundary, matching existing behavior where an empty eval split stays empty until the eval path decides whether to fall back and warn.
- Add a lightweight `Environment.from_components(taskset, harness=None, **config)` or `vf.Env(...)` alias for Python/notebook users. Keep `EnvRun` narrow and avoid duplicating the full runner API.
- Add read/write rollout scratchpads before deciding whether `State`, `Trace`, or a hybrid facade is the right authoring object.
- Add transcript and final-completion convenience views regardless of the persisted output object. Simple env authors do not need to think in graph terms unless they use graph features.
- Add output projection for selected scratchpad/info, metrics, or rewards for dashboards that depend on `state_columns`.
- Add `Task.workdir`, per-stage timeouts, and runtime hardening from 1576 as the standard; map 1559 task resources/image into those fields where possible.
- Add rich system-prompt merge strategies while preserving explicit harness capability checks.
- Add a `prime eval run` bridge that hides `eval`/`vf-eval`, prints helper commands, auto-installs environments, uploads to the platform, and keeps the v0/v1 top-level UX aligned.
- Add init templates for v0 envs, v1 taskset-only envs, v1 taskset-plus-harness envs, and optional tool/user stubs. Generated files keep subcomponents local by default.

### Port The 1559 Tooling Model Through A Proxy

The biggest functional gap is 1559's framework-owned tool execution. In 1559, the framework sees each tool call, injects hidden arguments, applies `sets`/`extends`, updates state, hides tools, and normalizes responses. In 1576, the harness/program generally receives MCP URLs and calls tools directly. A direct MCP URL broker cannot implement 1559's hidden binding semantics because the framework must sit between the agent and the real MCP server.

The right unification path is:

1. Use `Toolset` as the public name and mental model.
2. Treat one `Toolset` as one MCP server that can expose multiple tools.
3. Keep 1576's `Tools` dataclass only as an internal or renamed simple server spec.
4. Add a `Toolset` adapter that can produce command/script/url-backed MCP servers from decorated Python methods or simple server specs.
5. Support multiple toolsets on a taskset or harness without extra ceremony.
6. Add a framework-owned MCP proxy for decorated toolsets.
7. Have the proxy expose only visible tool schemas to the model/program.
8. On each call, inject bound values from `Task`, rollout scratchpad, runtime, taskset config, and whichever trace/state object is selected.
9. Forward to local Python methods or upstream MCP servers.
10. Apply `sets`/`extends` into the selected scratchpad/output object, metrics, rewards, stop state, artifacts, or message metadata.
11. Record the tool call and result in the persisted rollout record.

This ports the valuable 1559 behavior while preserving 1576's execution model. It also gives a single place to implement dynamic setup tools, task-level visibility, hidden args, response normalization, and resource exposure.

The same hidden-argument and scratchpad machinery is available to user simulators. User APIs can remain distinct from tool APIs, but users need a clean way to read task/context values and write rollout-local state when an environment pattern requires it.

### Be Selective With 1559 Lifecycle Semantics

- Add explicit post-turn or post-generation hooks for migrated environments that need online state updates.
- Prefer `Taskset.setup`, `Taskset.finalize`, `Taskset.validate`, `@metric`, `@reward`, and `@group_reward` as the main lifecycle.
- Preserve reward weights and deterministic decorator priorities. Reward priorities are required for expensive operations that write scratchpad/state in-place once, such as judge queries, so multiple downstream reward components can reuse the result.
- Keep priority support general enough for other decorators that use the same handler system, while excluding lifecycle stages that no migrated environment needs.
- Move advantage logic into prime-rl algorithm modules. Preserve env-level configurability and user-extensible algorithms without requiring users to fork or edit prime-rl. Add rollout fields for algorithm-produced advantages if that is the cleanest integration point.
- Support teacher/judge model calls through explicit `resolve_client(...)` use first. Add a first-class teacher client once several environments use it.

### Decide The Group Rollout Contract Explicitly

1576 currently covers the common case: sample multiple rollouts for the same task, then score traces together. 1559 supports a more general pattern: expand one group request into different tasks/states before generation. That is powerful but also widens the contract.

Use this sequence:

- Keep 1576 as-is and model group variants as distinct tasks in the taskset. This is the default.
- Add `Taskset.expand_group(task, n) -> list[Task]` for group-specific prompt/task variation.
- Port 1559's full `init_group(task, n) -> tasks, states` only for mutable pre-rollout state. This is the most invasive option and remains outside the first unified API.

The implementation leaves a clear extension point for adding `expand_group` without changing the public group-reward decorator shape.

### Port 1559 Environments In Tiers

- Tier 1: replay harness/taskset, simple single-turn tasksets, split-based examples, and output projection. These validate the basic authoring adapter.
- Tier 2: command/program harnesses that map naturally onto 1576's harness contract.
- Tier 3: TextArena/Wordle and other user simulators. Prefer 1576's simulator model where outcome data is written to runtime artifacts and read in `finalize`, with direct hidden state updates for patterns that require them.
- Tier 4: BFCL, Tau2, OpenEnv, OpenReward, NemoGym, and framework adapters. These drive the proxy and lifecycle decisions rather than being ported through one-off compatibility hacks.

Every existing v1 environment pattern has a clean migration into this 1576-based shape. The bar is not line-for-line API compatibility, but authors do not rewrite around missing hidden args, scratchpads, toolsets, user simulation, system-prompt merging, or harness packaging.

## Recommendations For Updating PR 1559 To Cover PR 1576 Features

Using PR 1559 as the base is viable only when immediate compatibility with the existing v1 environment zoo is the highest priority and training/server architecture can wait. Near-full 1576 coverage requires several large architectural imports.

### Required Large Imports

- Add or deeply integrate `Trace` graph capabilities that can represent branches, token IDs, logprobs, sampled masks, multimodal data, routed experts, and raw provider responses.
- Add the provider-native relay and dialect registry so programs can talk to OpenAI/Anthropic-compatible endpoints without lossy SDK normalization.
- Add the optional `TrainClient` and renderer integration for vLLM/tokenization/logprob capture.
- Add the env server protocol, worker pools, interception pools, and `serve` command.
- Add the v0 legacy bridge that maps v0 outputs into the new trace format.
- Add the v1 validate/serve helper commands and route user-facing eval through `prime eval run`.

These are not small feature ports. They change the core execution and output model of 1559.

### Smaller 1576 Features Worth Cherry-Picking Under A 1559 Base

- `RetryConfig` with separate model/runtime/rollout retry scopes.
- `Taskset.validate(task, runtime)` and a validation CLI.
- `Taskset.finalize(...)` as an explicit post-harness stage.
- Stage-specific timeouts.
- `Task.workdir`.
- Runtime hardening: process-group cleanup, API-key filtering, `run_uv_script`, interpreter caching, Prime runtime limiters, labels, timeouts, and atexit cleanup.
- Harness capability flags.
- Annotation-based config narrowing.
- A simple command/script/url `Toolset` server spec alongside decorated Python `Toolset`s.
- Selected 1576 harnesses such as default shell, Codex, Compact, and RLM.

### Risk Under A 1559 Base

The main risk is ending up with two uncoordinated output/scratchpad models: `State` for authoring/eval compatibility and `Trace` for training/server compatibility. When both are first-class without one ownership model, every tool, user simulator, harness, metric, reward, retry, and eval writer must decide which object owns truth. That makes the implementation harder to reason about than either PR alone.

Under a 1559 base, unification first designs hidden arguments and read/write scratchpads, then decides how `State` and `Trace` relate. Without that decision, full 1576 parity becomes a long-term fork inside v1.

## Best Path To A Unified Implementation

The strongest unification plan is to use PR 1576's runtime/server/provider architecture as the base and port selected 1559 features into it.

Reasons:

- 1576 has the harder-to-retrofit primitives: trace graph, provider relay, optional training client, env server, retry model, v0 bridge, and plugin architecture.
- 1559's strongest features are mostly authoring conveniences and environment behavior. They can be added as adapters, compatibility views, or proxy behavior without replacing 1576's core.
- 1559's `State` and lifecycle model are easier to emulate on top of a 1576-style runtime/server core than 1576's graph/training/server model is to emulate on top of 1559.
- 1576 is closer to a single future data plane for hosted evals, hosted training, local evals, v0 compatibility, and agentic harnesses.

The unified architecture is:

- Core rollout record: a single object model that supports graph traces and read/write scratchpads. The exact public `State` versus `Trace` shape follows the hidden-argument design.
- Simple author view: transcript/completion/scratchpad views over the rollout record.
- Core runner: `Environment` -> `Episode` -> `Rollout`.
- User-facing CLI: `prime eval run`; helper/internal commands may include validate/serve.
- Core loading: plugin IDs plus typed config narrowing.
- Task data: typed `Task` with stable IDs, split support, prompt/instruction fields, resources, workdir, and timeouts.
- Scoring: `@metric`, weighted and priority-ordered `@reward`, `@group_reward`, plus explicit `setup`/`finalize`/`validate`.
- Tools: public `Toolset` abstraction for MCP servers, including simple command/url toolsets and decorated Python toolsets through a proxy.
- Users: keep a simple user simulator path, with hidden args and scratchpad writes for patterns that require them.
- Runtime: merge the best 1559 and 1576 subprocess, Docker, and Prime runtime details; drop inactive stubs.
- Legacy: keep 1576 v0 bridge.
- Examples: keep examples under `environments/`, covering tasksets, harnesses, tools, users, and third-party agent frameworks with symmetric packaging.

## Required Decisions

These decisions require explicit product/API judgment rather than mechanical porting.

### Public Rollout Object

Decision path: design hidden arguments and scratchpads first, then finalize the public `State` versus `Trace` shape. The public object must support read/write scratchpads for tools/users and graph/training data for provider and RL flows.

Reasoning: `State` is much easier for simple eval authors and scratchpad patterns. `Trace` better represents branches, subagents, token-level training metadata, and raw provider relay behavior. The right answer depends on how much mutable rollout-local data is needed during tool/user execution.

### Loader Shape

Decision: support old loaders only at compatibility boundaries. New packages use `load_taskset` and `load_harness` with the unified environment composition.

Reasoning: the old shape is convenient, but it hides the split between taskset and harness that both PRs are trying to establish.

### Tool Binding

Decision: hidden args and multiple toolsets are core requirements. Implement them through a `Toolset` proxy/adapter on top of the 1576 execution model, not as a replacement for runtime-backed MCP servers.

Reasoning: hidden args, `sets`, `extends`, dynamic tools, and stateful user simulation are not just UI differences. They enable real environment patterns. The implementation does not require every harness to know about Python method decorators.

### Advantage Computation

Decision: do not port 1559's exact decorator-first API. Fold the feature into the prime-rl algorithm-module work, with env-level algorithm configurability and user-extensible algorithms that do not require editing prime-rl.

Reasoning: advantage computation lives in the RL algorithm layer, but environments still configure or select algorithms and the rollout object carries algorithm-produced advantages.

### Group Initialization

Decision: start without `init_group` and preserve an easy path to add `expand_group(task, n)`.

Reasoning: repeated rollouts of one task cover the common group reward pattern. Mutable pre-rollout group state is powerful, but it is also a broader contract.

### System Prompt Strategies

Decision: support rich merge strategies. Keep harness capability flags and explicit prompt resolution so strategy behavior remains inspectable.

Reasoning: prompt composition can become a hidden policy layer quickly. 1576's direct warnings and capability flags are easier to reason about.

### Runtime Stubs

Decision: drop inactive stubs until there is a working integration.

Reasoning: v1 runtime surface is small and real. Subprocess, Docker, Prime, and Modal already cover the active paths.

### Python Version

Decision: drop Python 3.10 and use a Python 3.11+ stance.

Reasoning: Python 3.10 is close to EOL. Heavy training dependencies still remain outside core requirements.

## Features To Drop Or Defer

Drop or defer these from 1559:

- Exact `State` primary API when the scratchpad design lands elsewhere.
- 1559's exact `@advantage` API, in favor of prime-rl algorithm modules with env-level configurability.
- Unused lifecycle stages, but not reward weights or decorator priorities.
- Daytona runtime stub.
- Exact old `vf-eval` primacy for v1.
- Exact generated environment package shape.
- Empty runtime stubs.
- Group `init_group`, with `expand_group` kept as the clear extension point.
- Dynamic setup tools until the tool proxy exists.

Drop or defer these from 1576 when they block merge:

- Some example tasksets that are not needed as coverage for core APIs.
- Rich dashboard polish behind CLI stability.
- Hub install-on-demand inside loaders when it complicates local development or CI.
- Optional harnesses that are not active product commitments.

Preserve these from 1576 for hosted training and evals:

- `Trace` graph.
- Provider-native relay and dialects.
- Train client/renderers integration as an optional extra.
- Env server/worker pool.
- v0 legacy bridge.
- Retry config.
- Validate CLI.
- Prime CLI bridge through `prime eval run`.

Preserve these from 1559 through an equivalent in 1576:

- Train/eval split.
- Existing environment behavior for the major migrated env families.
- Tool/user patterns that require hidden args and read/write scratchpads.
- Reward weights and decorator priorities.
- Replay harness/taskset.
- Simple Python construction path or an equally ergonomic replacement.
- Rich system-prompt merge strategies.
- Init templates for v0, v1 taskset-only, v1 taskset-plus-harness, and tool/user stubs.
- Independent harness execution with optional runtime reuse.

## Implementation Sequence

1. Choose 1576 as the base branch for unification.
2. Freeze the intended public v1 object model except for the State-vs-Trace scratchpad decision: `Environment`, `Taskset`, `Harness`, `Task`, `Toolset`, `User`, runtime/client configs, and loader IDs.
3. Design hidden args and read/write scratchpads, then decide how `State` and `Trace` relate publicly.
4. Add train/eval split support and document it as the taskset data-loading path.
5. Add transcript/completion convenience views over the selected rollout object.
6. Add weighted reward aggregation and deterministic decorator priority ordering.
7. Add output projection for existing eval consumers that need `state_columns`.
8. Port 1559's replay harness/taskset to prove replayed transcripts map cleanly.
9. Add `Toolset` adapter and MCP proxy for hidden args, visibility, and scratchpad/output updates.
10. Port one 1559 tool-heavy environment and one 1559 user-sim-heavy environment as acceptance tests for the proxy.
11. Add the Prime CLI bridge so users run `prime eval run` for both v0 and v1.
12. Add init templates for v0, v1 taskset-only, v1 taskset-plus-harness, and tool/user stubs.
13. Add independent harness execution with optional runtime reuse.
14. Evaluate group expansion after porting group reward examples.
15. Port selected framework/benchmark environments from 1559 in behavior-preserving order.
16. Backfill docs and examples under `environments/`, including tasksets, harnesses, tools, users, and third-party agent frameworks.
17. Run the combined test matrix: 1576 graph/client/server tests, 1559 tool/user/group tests adapted to the selected rollout object, and live `prime eval run` smoke tests for representative envs.

## Acceptance Tests For Near-Full Coverage

A unified PR proves the following before replacing both branches.

- A basic single-turn taskset runs through the v1 runner and produces a persisted trace-compatible rollout record.
- The same basic taskset runs through `prime eval run`, with helper commands, auto-installation, and upload behavior where appropriate.
- A taskset with train/eval split can evaluate the eval split and train on the train split.
- A replay taskset can reconstruct a transcript as the unified rollout record.
- A command/program harness can run in subprocess and Docker runtimes.
- A harness can run independently through a simple API such as `harness.run(task="Write 'hello world' backwards.", model="openai/gpt-5")`.
- A harness can reuse an existing runtime when requested.
- Prime runtime exposure works for at least one harness/tool path.
- A shared MCP tool server is started once and reused across rollouts.
- A per-rollout MCP tool server works in its own runtime.
- Multiple toolsets can be attached to one taskset or harness, and each toolset can expose multiple tools.
- A decorated `Toolset` hides bound args from the model and injects task/context/scratchpad values.
- A decorated tool can update scratchpad data, metrics, reward, and stop condition through a documented mechanism.
- Weighted reward components aggregate deterministically into the final reward.
- Priority-ordered reward decorators can run an expensive judge query once, write the result to scratchpad/state, and allow multiple downstream reward components to reuse it.
- A user simulator can respond after assistant messages.
- A user simulator can optionally bootstrap a conversation before the first assistant/model turn.
- A user simulator can use hidden args and write scratchpad data when needed.
- A group reward can score multiple traces for the same task.
- With group expansion, a group reward can score traces generated from distinct expanded tasks.
- Provider-native chat, responses, and Anthropic dialects preserve raw responses and trace messages.
- Streaming responses are traced correctly.
- Training generation records token IDs, logprobs, sampled masks, and renderer metadata when training extras are installed.
- v0 legacy environments still run and map into the unified v1 rollout output.
- Retry config correctly retries model/runtime failures without hiding final trace errors.
- `validate` can run task validation without model calls.
- Init templates can create a v0 env, v1 taskset-only env, v1 taskset-plus-harness env, and tool/user stubs with local subcomponents.
- Rich system-prompt merge strategies work across tasksets and harnesses.
- Third-party harness examples under `environments/` show a clear packaging strategy for Python agent frameworks.
- Representative migrated envs from both PRs pass local smoke tests.

## Bottom Line

PR 1559 is the stronger source of Verifiers-native authoring ergonomics and existing environment coverage. PR 1576 is the stronger source of future runtime, server, provider, and training architecture. The best unification path is to base on PR 1576's architecture, then port 1559's split loading, replay support, selected environment ports, rich system-prompt handling, init templates, and especially decorated `Toolset`/user semantics through a framework-owned MCP proxy.

Trying to base on PR 1559 and import 1576's graph/server/client stack requires replacing the heart of 1559 after the fact. Basing on 1576 and adding 1559 compatibility features is still substantial, but it keeps one runtime/provider/server architecture while leaving the State-vs-Trace public object decision to the scratchpad and hidden-argument design.
