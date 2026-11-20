# API reference

This page lists the complete public API of verifiers. A unit test (`tests/unit/test_api.py`) keeps it equal to the `__all__` of each public namespace.

## Public and internal

verifiers follows the Python conventions for public and internal interfaces in [PEP 8](https://peps.python.org/pep-0008/#public-and-internal-interfaces) and the [typing specification](https://typing.python.org/en/latest/spec/distributing.html#library-interface-public-and-private-symbols):

- **Public namespaces**: `verifiers.v1` (imported as `vf`) and the plugin packages below it, `verifiers.v1.harnesses.*`, `verifiers.v1.tasksets.*`, `verifiers.v1.envs.*` and `verifiers.v1.judges.*`. Each one declares its public names in `__all__`.
- **Internal modules**: every module or package whose name starts with an underscore, for example `verifiers.v1._trace` or `verifiers.v1.harnesses.bash._harness`. A name inside an internal module is internal, even when it has no underscore. Import it from its public namespace instead.
- **Internal names and members**: every name, method or attribute that starts with an underscore. On a public class, every member without an underscore is public.
- **Config fields**: the fields of a public config class are public. They are also the TOML and CLI surface.
- **Framework-built objects**: verifiers constructs `Agent`, `AgentRun`, `Interaction`, `HarnessSession`, `Runtime`, `Deployment`, `Trace` and `Episode` and gives them to your code. Their constructors are internal. Use their public members.
- **Subclassing**: subclass `Taskset`, `Task`, `TaskData`, `Harness`, `ACPHarness`, `Env`, `Judge`, `Toolset`, `RuntimeBackend` and the config classes. Override only the hooks they document. Do not subclass `Client` yet: its request types are internal.

### Compatibility

A release that removes, renames or changes a public name lists the change and its migration in a `Breaking` section of its release notes. Internal names can change in any release without notice.

### Check your code

The package ships a `py.typed` marker, so type checkers read its public API. To find uses of internal names, enable one of these checks:

- Pyright: `reportPrivateUsage` and `reportPrivateImportUsage`.
- Ruff: `PLC2701` (`import-private-name`) and `SLF001` (`private-member-access`).

## `verifiers.v1`

Import the namespace as `import verifiers.v1 as vf`.

### Tasksets and tasks

| Name | Description |
| --- | --- |
| `Taskset` | Base class of a taskset. Implement `load()` to build its tasks. |
| `TasksetConfig` | Config of one taskset: its `id`, task config and system prompt. |
| `SharedTasksetConfig` | Taskset settings that several tasksets of one run can share. |
| `Task` | Base class of a task: lifecycle hooks, tools, metrics and rewards over one `TaskData`. |
| `TaskConfig` | Run-time settings that `Task` behavior reads, including plugged judges and functions. |
| `TaskData` | Serializable data of one task: prompt, files, references and resource needs. |
| `WireTaskData` | Wire form of `TaskData` that keeps task-specific fields without the task class. |
| `TaskResources` | Optional resource limits of a task. |
| `TaskTimeout` | Optional per-task timeouts, in seconds. |
| `SelectConfig` | Task selection: `include`, `exclude`, `shuffle`, `skip` and `limit`. |
| `SelectCLIConfig` | `SelectConfig` with the CLI short flags `-n` and `-s`. |
| `TaskMatchConfig` | Tasks named by position or identity. |
| `TaskMatchFilesConfig` | Text files with more entries for each `TaskMatchConfig` list. |
| `DecoratedFunctionConfig` | A plugged function that replaces a decorated task method. |
| `RewardFunctionConfig` | A plugged function that replaces a `@vf.reward` method. |
| `StopFunctionConfig` | A plugged function that replaces a `@vf.stop` method. |
| `ScoringConfig` | How and where a task is scored: shared or isolated runtime, retries and network. |
| `State` | Base class of the mutable state of one rollout. |
| `StateT` | Type variable for a `State` subclass. |

### Decorators

| Name | Description |
| --- | --- |
| `reward` | Mark a `Task` method as a weighted reward. |
| `metric` | Mark a `Task` or `Harness` method as a metric that is recorded but not summed. |
| `stop` | Mark a predicate that stops the rollout when it returns true. |
| `tool` | Mark a `Toolset` method as an MCP tool for the model. |
| `intercept` | Mark a handler that inspects a typed boundary and can replace its value. |

### Harnesses

| Name | Description |
| --- | --- |
| `Harness` | Base class of a harness: the program the model runs in. |
| `HarnessConfig` | Config of one harness: its `id`, environment, tools and skills. |
| `HarnessSession` | Handle onto one harness execution in one rollout. |
| `ID` | Plugin id: the name of a package that exports a plugin. |
| `ACPHarness` | Base class of a harness backed by one ACP process per rollout. |
| `ACPConfig` | The ACP process and initial prompt of one harness. |
| `ACPTurn` | One completed ACP prompt and its extension metadata. |
| `ModelContext` | Model, endpoint config and sampling of one rollout. |
| `ModelRoute` | Where the model turns of one rollout go. The caller can change it during the rollout. |
| `RouteRejected` | Raised when a rollout cannot take a route change. |

### Environments and agents

| Name | Description |
| --- | --- |
| `Env` | Base class of an environment: a taskset, its agents and how one task becomes one episode. |
| `EnvConfig` | Config of an environment: the `[env]` block of a run. |
| `SharedEnvConfig` | Environment settings that several environments of one run can share. |
| `SingleAgentEnv` | The default environment: one `agent` seat on the taskset. |
| `SingleAgentEnvConfig` | Config of `SingleAgentEnv`. |
| `EnvTimeoutConfig` | Timeouts of the `run()` and `finalize()` hooks of an environment, in seconds. |
| `RunSlot` | One planned episode, observable while it runs. |
| `Agent` | A harness, a model and a runtime policy that can run any task. |
| `AgentConfig` | Config of one agent seat. |
| `AgentTimeoutConfig` | Timeouts of the stages of one agent run, in seconds. |
| `AgentRun` | One run that `Agent.start()` schedules, including its retries. |
| `Agents` | The agents of a config, addressed by attribute (`agents.solver`). |
| `make_agent` | Make the agent for an `AgentConfig`. |
| `default_agent_harness` | The harness id that an unpinned agent uses. |
| `Interaction` | An agent rollout that the caller drives turn by turn as its user. |
| `Segment` | The agent and tool output of one harness segment, as `Interaction.turn` returns it. |
| `RetryConfig` | Ordered retry rules at one execution boundary. |
| `RetryRule` | One retry rule. The first rule that matches an error applies. |
| `InterceptionServerConfig` | One interception server that every rollout shares. |
| `StaticInterceptionPoolConfig` | A fixed set of interception servers. |
| `ElasticInterceptionPoolConfig` | A pool of interception servers that grows on demand. This is the default. |
| `PrimeTunnelConfig` | Expose the interception server through a Prime tunnel. |
| `CustomTunnelConfig` | Reach the interception server at your own URL. |
| `BaseConfig` | Base class of every config, with strict validation. |
| `merge_defaults` | Fill a raw config block with the values of a defaults block. |
| `resolve_env_field` | Narrow the `env` field of a run config to the concrete environment config class. |
| `narrowed_env_annotation` | The `env` field annotation after the CLI narrows it. |

### Runtimes

| Name | Description |
| --- | --- |
| `Runtime` | Command and file access to one box, with fixed user and environment bindings. |
| `RuntimeProcess` | A process whose input and output stream across a runtime boundary. |
| `ProgramResult` | Exit code, stdout and stderr of one finished program. |
| `RuntimeConfig` | Any runtime config, discriminated by `type`. |
| `SubprocessConfig` | Run on the host as subprocesses. |
| `DockerConfig` | Run in a local Docker container. |
| `PrimeConfig` | Run in a Prime sandbox. |
| `CustomRuntimeConfig` | Run on a runtime backend that you import by path. |
| `BindMount` | A host file or directory to mount into a container. |
| `PrimeVolume` | A named Prime volume to mount into a sandbox. |
| `NetworkPolicyConfig` | What a box can reach. |
| `RuntimeInfo` | Any resolved runtime, discriminated by `type`. |
| `BaseRuntimeInfo` | Base class of a resolved runtime. A custom backend sets its `info` to a subclass. |
| `SubprocessRuntimeInfo` | A resolved subprocess runtime. |
| `DockerRuntimeInfo` | A resolved Docker runtime. |
| `PrimeRuntimeInfo` | A resolved Prime runtime. |
| `CustomRuntimeInfo` | A resolved custom runtime. |
| `Deployment` | The execution resources of one rollout attempt: workspace, services and verifier. |
| `DeploymentConfig` | How a task deployment runs. |
| `ComposeSpec` | Task services from a Docker Compose file. |
| `ServicesSpec` | Named Prime VMs beside the `main` workspace, on one private network. |
| `provision_runtime` | Start a runtime from a config and stop it when the context exits. |
| `set_base_sandbox_labels` | Set labels that every Prime sandbox of this process gets. |
| `RuntimeBackend` | Base class of a runtime backend. Subclass it for a custom runtime. |

### Tools

| Name | Description |
| --- | --- |
| `Toolset` | Base class of an MCP tool server. Mark its tools with `@vf.tool`. |
| `ToolsetConfig` | Config of a tool server that runs for each task. |
| `SharedToolsetConfig` | Config of a tool server that all rollouts of a run share. |
| `ToolsetTimeoutConfig` | Timeouts of a launched tool server, in seconds. |

### Clients and sampling

| Name | Description |
| --- | --- |
| `Client` | A model client. Use it as a type. Make one with `resolve_client`. |
| `ClientConfig` | Any client config, discriminated by `type`. |
| `BaseClientConfig` | An OpenAI-compatible endpoint. The API key comes from an environment variable. |
| `EvalClientConfig` | Forward each request to an endpoint. This is the default. |
| `TrainClientConfig` | Send requests to a vLLM generate endpoint with client-side tokenization, for training. |
| `ClientTimeoutConfig` | Timeouts of each phase of one HTTP request, in seconds. |
| `SamplingConfig` | Sampling parameters. Provider-specific keys pass through. |
| `resolve_client` | Make the `Client` for a client config. |
| `resolve_api_key` | The API key of a client config. |
| `resolve_headers` | The request headers of a client config. |
| `build_async_openai` | An `AsyncOpenAI` client for a client config. |

### Judges

| Name | Description |
| --- | --- |
| `Judge` | Base class of an LLM judge that scores a trace. |
| `JudgeConfig` | Config of one judge: endpoint, model, sampling, name and weight. |
| `Judges` | A list of judge configs. |
| `JudgeResponse` | The text, parsed value and usage of one judge call. |
| `JudgeView` | What a judge sees: `last_reply` or `full_trace`. |
| `DecisionJudge` | A judge that grades how well the response satisfies the task, on graded levels. |
| `DecisionJudgeConfig` | Config of `DecisionJudge`. |
| `ReferenceJudge` | A judge that checks the answer against a reference answer. |
| `ReferenceJudgeConfig` | Config of `ReferenceJudge`. |
| `RubricJudge` | A judge that grades the answer against weighted criteria. |
| `RubricJudgeConfig` | Config of `RubricJudge`. |
| `Criterion` | One weighted criterion of a rubric. |

### Messages

| Name | Description |
| --- | --- |
| `Message` | Any message, discriminated by `role`. |
| `Messages` | A list of messages. |
| `MessageContent` | Message content: a string or a list of content parts. |
| `SystemMessage` | A system message. |
| `UserMessage` | A user message. |
| `AssistantMessage` | An assistant message, with optional reasoning and tool calls. |
| `ToolMessage` | The result of one tool call. |
| `ContentPart` | Any content part, discriminated by `type`. |
| `TextContentPart` | A text content part. |
| `ImageUrlContentPart` | An image content part. |
| `ImageUrlSource` | The URL of an image content part. |
| `Tool` | A tool definition that the model can call. |
| `ToolCall` | One tool call from the model. |
| `Request` | The typed conversation that is about to cross a model or harness boundary. |
| `Response` | One model response: message, finish reason, usage and tokens. |
| `Usage` | Token counts from the provider. |
| `SamplerSupport` | The sampler support of each token, in CSR form. |
| `TurnTokens` | The training tokens of one turn. |
| `RoutedExperts` | The expert routing of each token, for mixture-of-experts training. |
| `WeightVersionSpan` | A range of completion tokens that one weight version sampled. |
| `content_text` | The text of message content, without images. |
| `image_data_url` | Encode an image as a base64 PNG data URL. |
| `decode_ndarray` | Decode an array that a trace record stores. |

### Traces and episodes

| Name | Description |
| --- | --- |
| `Trace` | The record of one agent run: messages, model calls, rewards, metrics and errors. |
| `WireTrace` | Wire form of `Trace` that keeps task-specific fields without the task class. |
| `TraceTask` | The task as the trace records it. |
| `Reward` | One named reward: score and weight. |
| `Error` | One recorded error. |
| `Episode` | The traces of all agents on one task. |
| `WireEpisode` | Wire form of `Episode`. |
| `EnvInfo` | The environment that ran an episode. |
| `GroupInfo` | The rollout group of an episode. |
| `AgentInfo` | The agent that produced a trace. |
| `RunInfo` | Any run info, discriminated by `type`. |
| `EvalRunInfo` | An evaluation run. |
| `TrainRunInfo` | A training run. |
| `WorkInfo` | Any work info, discriminated by `type`. |
| `EvalWorkInfo` | The evaluation step and policy of an episode. |
| `TrainWorkInfo` | The training step and policy of an episode. |
| `VersionInfo` | The verifiers version and commit that produced a trace. |
| `ModelCall` | One recorded model call. |
| `InterceptRecord` | One recorded intercept handler. |
| `PolicyEvent` | One framework policy decision on a model call. |
| `PolicySpan` | The policy versions that generated an episode. |
| `AgentSpan` | The agent stage, split into model time and harness time. |
| `Timing` | The time spans of the stages of one run. |
| `TimeSpan` | Start and end timestamps, with their duration in seconds. |
| `TimeSplit` | One measured duration, in seconds. |
| `Branch` | One root-to-leaf path of the message graph: one training sample. |
| `MessageNode` | One message of the message graph and the tokens that it adds. |
| `TRACE_VERSION` | The current trace schema version. |
| `ACPInfo` | The metadata that an ACP harness sends with one model request. |
| `ACP_SEMANTIC_EDGES_METADATA_KEY` | The ACP metadata key of semantic edges. |
| `ParentLink` | One semantic parent of a `MessageNode`. |
| `SemanticEdge` | A relation between two model requests that a harness declares. |
| `SemanticEdgeSet` | A list of semantic edges that a harness publishes. |
| `min_new_input_tokens` | The minimum number of new input tokens of each model call. |
| `routing_session_id` | The inference session that routes the model calls of a rollout. |

### Scoring and grading

| Name | Description |
| --- | --- |
| `grade_submission` | Grade collected artifacts in a fresh verifier runtime. |
| `run_submission` | Run untrusted files as a non-root user in a fresh, offline box. |
| `Artifact` | One path to copy from the solver runtime to the verifier runtime. |
| `ARTIFACTS_DIR` | The directory whose contents `collect` always copies. |
| `collect` | Collect the artifact paths from a runtime as a tar archive. |
| `restore` | Extract collected artifacts in a runtime at their original paths. |
| `PATCH_CAP_BYTES` | The maximum size of a captured patch, in bytes. |
| `capture_patch` | Record the cumulative diff of the agent in `trace.info["patch"]`. |
| `resolve_head` | The current commit SHA of a repository in a runtime. |
| `snapshot_untracked` | The untracked files of a repository in a runtime, to pass to `capture_patch` as `ignore`. |
| `compare_stdout_results` | Compare program output with expected output, with a numeric tolerance. |
| `extract_boxed_answer` | The content of the last `\boxed{}` in a text. |
| `parse_judge_choice` | The choice letter of a judge reply. |
| `parse_pytest_outcomes` | The outcome of each test in pytest output. |
| `read_answer_file_or_last_reply` | The content of an answer file, or the last reply when the file does not exist. |
| `verify_boxed_math_answer` | Score a boxed math answer against the expected answer. |

### Errors

| Name | Description |
| --- | --- |
| `RolloutError` | Base class of a failure that the trace records instead of a crash. |
| `EnvError` | The `run()` or `finalize()` hook of an environment failed. |
| `TaskError` | Task code failed: `setup`, `finalize`, a reward or a metric. |
| `HarnessError` | The harness failed to install or start, or its process failed. |
| `ToolsetError` | A tool server could not be built or served. |
| `SandboxError` | A runtime operation failed. |
| `ProviderError` | A model provider call failed. |
| `InterceptionError` | Communication with the interception server failed. |
| `TunnelError` | The tunnel to the interception server failed. |

### Plugin loading

| Name | Description |
| --- | --- |
| `import_taskset` | Import the package of a taskset id. |
| `import_harness` | Import the package of a harness id. |
| `import_judge` | Import the module of a judge id. |
| `import_environment` | Import the package of an environment id. |
| `taskset_class` | The `Taskset` class of a taskset id. |
| `harness_class` | The `Harness` class of a harness id. |
| `judge_class` | The `Judge` class of a judge id. |
| `environment_class` | The `Env` class of a taskset id or an environment id. |
| `task_type` | The `Task` class of a taskset id. |
| `taskset_config_type` | The config class of a taskset id. |
| `harness_config_type` | The config class of a harness id. |
| `judge_config_type` | The config class of a judge id. |
| `env_config_type` | The config class of an environment. |
| `load_taskset` | Make the taskset for a taskset config. |
| `load_harness` | Make the harness for a harness config. |
| `load_judge` | Make the judge for a judge config. |
| `load_environment` | Make the environment for an environment config. |
| `resolve_env_config` | Validate raw environment config data as its concrete config class. |
| `narrow_plugin_field` | Validate a config field that holds a plugin config as the concrete config class of its `id`. |
| `default_harness_id` | The harness id that a taskset uses by default. |

### Trainer streaming

| Name | Description |
| --- | --- |
| `DeltaStreamer` | Send the changes of live traces as deltas. |
| `EpisodeAssembly` | Apply received deltas to rebuild the traces. |
| `plan_rollouts` | Assign accepted episodes to task occurrences by content hash. |

## `verifiers.v1.harnesses`

| Name | Description |
| --- | --- |
| `BashHarness`, `BashHarnessConfig`, `BrowserUseHarness`, `BrowserUseHarnessConfig`, `ClaudeCodeHarness`, `ClaudeCodeHarnessConfig`, `CodexHarness`, `CodexHarnessConfig`, `DeepSeekHarness`, `DeepSeekHarnessConfig`, `HermesAgentHarness`, `HermesAgentHarnessConfig`, `KimiCodeHarness`, `KimiCodeHarnessConfig`, `MiniSWEAgentHarness`, `MiniSWEAgentHarnessConfig`, `NullHarness`, `NullHarnessConfig`, `OpenClawHarness`, `OpenClawHarnessConfig`, `OpenCodeHarness`, `OpenCodeHarnessConfig`, `PiHarness`, `PiHarnessConfig`, `PrimeAgentHarness`, `PrimeAgentHarnessConfig`, `RLMHarness`, `RLMHarnessConfig`, `StirrupHarness`, `StirrupHarnessConfig`, `Terminus2Harness`, `Terminus2HarnessConfig` | Every built-in harness and its config, from the packages below. |

Each built-in harness is a package that `--harness.id` names. The package exports the harness class and its config class.

## `verifiers.v1.harnesses.bash`

| Name | Description |
| --- | --- |
| `BashHarness`, `BashHarnessConfig` | A chat loop with a bash tool. |
| `CompactionConfig` | When the chat loop compacts the context. |

## `verifiers.v1.harnesses.browser_use`

| Name | Description |
| --- | --- |
| `BrowserUseHarness`, `BrowserUseHarnessConfig` | The browser-use agent. |

## `verifiers.v1.harnesses.claude_code`

| Name | Description |
| --- | --- |
| `ClaudeCodeHarness`, `ClaudeCodeHarnessConfig` | Claude Code over ACP. |

## `verifiers.v1.harnesses.codex`

| Name | Description |
| --- | --- |
| `CodexHarness`, `CodexHarnessConfig` | Codex over ACP. |

## `verifiers.v1.harnesses.deepseek`

| Name | Description |
| --- | --- |
| `DeepSeekHarness`, `DeepSeekHarnessConfig` | The DeepSeek agent over ACP. |

## `verifiers.v1.harnesses.hermes_agent`

| Name | Description |
| --- | --- |
| `HermesAgentHarness`, `HermesAgentHarnessConfig` | Hermes Agent over ACP. |

## `verifiers.v1.harnesses.kimi_code`

| Name | Description |
| --- | --- |
| `KimiCodeHarness`, `KimiCodeHarnessConfig` | Kimi Code over ACP. |

## `verifiers.v1.harnesses.mini_swe_agent`

| Name | Description |
| --- | --- |
| `MiniSWEAgentHarness`, `MiniSWEAgentHarnessConfig` | mini-swe-agent. |

## `verifiers.v1.harnesses.null`

| Name | Description |
| --- | --- |
| `NullHarness`, `NullHarnessConfig` | A chat loop with only the task tools. |
| `CompactionConfig` | When the chat loop compacts the context. |

## `verifiers.v1.harnesses.openclaw`

| Name | Description |
| --- | --- |
| `OpenClawHarness`, `OpenClawHarnessConfig` | OpenClaw over ACP. |

## `verifiers.v1.harnesses.opencode`

| Name | Description |
| --- | --- |
| `OpenCodeHarness`, `OpenCodeHarnessConfig` | OpenCode over ACP. |

## `verifiers.v1.harnesses.pi`

| Name | Description |
| --- | --- |
| `PiHarness`, `PiHarnessConfig` | The Pi coding agent over ACP. |

## `verifiers.v1.harnesses.prime_agent`

| Name | Description |
| --- | --- |
| `PrimeAgentHarness`, `PrimeAgentHarnessConfig` | The Prime agent over ACP. |

## `verifiers.v1.harnesses.rlm`

| Name | Description |
| --- | --- |
| `RLMHarness`, `RLMHarnessConfig` | The recursive language model agent over ACP. |
| `CompactionConfig` | When the RLM agent compacts the context. |

## `verifiers.v1.harnesses.stirrup`

| Name | Description |
| --- | --- |
| `StirrupHarness`, `StirrupHarnessConfig` | The Stirrup agent over ACP. |

## `verifiers.v1.harnesses.terminus_2`

| Name | Description |
| --- | --- |
| `Terminus2Harness`, `Terminus2HarnessConfig` | The Terminus 2 agent. |

## `verifiers.v1.tasksets`

| Name | Description |
| --- | --- |
| `HarborConfig`, `HarborTaskset`, `NeMoGymConfig`, `NeMoGymTaskset`, `OpenEnvConfig`, `OpenEnvData`, `OpenEnvEnv`, `OpenEnvEnvConfig`, `OpenEnvTask`, `OpenEnvTaskset` | The built-in tasksets from the packages below. |

Each built-in taskset is a package that `--env.taskset.id` names.

## `verifiers.v1.tasksets.harbor`

| Name | Description |
| --- | --- |
| `HarborTaskset`, `HarborConfig` | Tasks from a Harbor dataset. |
| `HarborTask`, `HarborTaskConfig`, `HarborData` | One Harbor task, its config and its data. |
| `VerifierConfig` | The verifier of a Harbor task. |
| `HarborArtifact` | One artifact that a Harbor task copies to its verifier. |
| `CollectHook` | One command that collects verifier evidence in a service. |
| `Author` | One author of a Harbor task. |
| `dataset_dir` | Download a Harbor dataset and return its local directory. |
| `parse_task` | Read one Harbor task directory into `HarborData`. |
| `make_tar` | Archive a directory as a gzipped tar. |

## `verifiers.v1.tasksets.nemo_gym`

| Name | Description |
| --- | --- |
| `NeMoGymTaskset`, `NeMoGymConfig` | Tasks from a NeMo Gym resource server. |
| `NeMoGymTask`, `NeMoGymTaskConfig`, `NeMoGymData` | One NeMo Gym task, its config and its data. |
| `NeMoGymEnv` | The environment of NeMo Gym tasks. |

## `verifiers.v1.tasksets.openenv`

| Name | Description |
| --- | --- |
| `OpenEnvTaskset`, `OpenEnvConfig` | Tasks from an OpenEnv environment. |
| `OpenEnvTask`, `OpenEnvData` | One OpenEnv task and its data. |
| `OpenEnvEnv`, `OpenEnvEnvConfig` | The environment that plays an OpenEnv episode. |

## `verifiers.v1.tasksets.textarena`

| Name | Description |
| --- | --- |
| `TextArenaTaskset`, `TextArenaConfig` | Tasks from a TextArena game. |
| `TextArenaTask`, `TextArenaData` | One TextArena task and its data. |
| `TextArenaEnv`, `TextArenaEnvConfig` | The environment that plays a TextArena game. |

## `verifiers.v1.envs`

This package has no names of its own. Each built-in environment is a package that `--env.id` names.

## `verifiers.v1.envs.agentic_judge`

| Name | Description |
| --- | --- |
| `AgenticJudgeEnv`, `AgenticJudgeEnvConfig` | A solver agent, then a judge agent that grades its result. |
| `JudgeTaskConfig` | Config of the judge task. |
| `ScoreConfig` | How the judge verdicts become rewards. |
| `Criterion` | One criterion that the judge agent grades. |
| `TextFile` | One text file that the judge agent receives. |

## `verifiers.v1.envs.best_of_n`

| Name | Description |
| --- | --- |
| `BestOfNEnv`, `BestOfNEnvConfig` | `n` independent attempts on one task, with the best attempt marked. |

## `verifiers.v1.envs.best_of_n_judge`

| Name | Description |
| --- | --- |
| `BestOfNJudgeEnv`, `BestOfNJudgeEnvConfig` | `n` attempts, then a judge agent that ranks them. |
| `RankTask`, `RankTaskConfig` | The ranking task of the judge agent. |

## `verifiers.v1.envs.single_agent`

| Name | Description |
| --- | --- |
| `SingleAgentEnv`, `SingleAgentEnvConfig` | One agent on each task. This is the default environment. |

## `verifiers.v1.envs.user_sim`

| Name | Description |
| --- | --- |
| `UserSimEnv`, `UserSimEnvConfig` | An assistant agent that talks with a simulated user agent. |

## `verifiers.v1.judges`

| Name | Description |
| --- | --- |
| `DecisionJudge`, `DecisionJudgeConfig`, `ReferenceJudge`, `ReferenceJudgeConfig`, `RubricJudge`, `RubricJudgeConfig`, `Criterion` | The built-in judges from the modules below. |

Each built-in judge is a module that a judge `id` names.

## `verifiers.v1.judges.decision`

| Name | Description |
| --- | --- |
| `DecisionJudge`, `DecisionJudgeConfig` | A judge that grades how well the response satisfies the task, on graded levels. |

## `verifiers.v1.judges.reference`

| Name | Description |
| --- | --- |
| `ReferenceJudge`, `ReferenceJudgeConfig` | A judge that checks the answer against a reference answer. |

## `verifiers.v1.judges.rubric`

| Name | Description |
| --- | --- |
| `RubricJudge`, `RubricJudgeConfig` | A judge that grades the answer against weighted criteria. |
| `Criterion` | One weighted criterion of a rubric. |
| `CriterionVerdict`, `RubricVerdicts` | The verdict of one criterion, and all verdicts of one grading. |
| `criterion_metrics` | The metrics of the criterion verdicts. |
| `drop_reason` | The reason to drop a trace, from its criterion verdicts. |
| `weighted_score` | The weighted score of the criterion verdicts. |
