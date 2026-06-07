# prime-rl v1 Integration Sketch

This document sketches the smallest `prime-rl` changes needed to consume
`verifiers.v1` environments cleanly.

## Goal

`prime-rl` should treat v0 and v1 as two environment protocols behind one
trainer-facing adapter:

- v0 produces `trajectory` rollout outputs through the existing
  `verifiers.Environment` and ZMQ server path.
- v1 produces strict `State` outputs with `transcript`, token-level advantages,
  and typed task/state records through `verifiers.v1.Env`.

The split belongs at the environment adapter boundary. Training code should not
scatter `if trajectory else transcript` branches across tokenization,
interleaving, logging, or advantage handling.

## Current prime-rl Touchpoints

`src/prime_rl/orchestrator/envs.py` currently assumes the v0 contract:

- `vf.load_environment(...)` returns `vf.Environment`.
- `REQUIRED_STATE_COLUMNS = ["trajectory"]`.
- `run_rollout(...)` passes `vf.RolloutInput`, `client`, `model`,
  `sampling_args`, `state_columns`, and `env_client`.
- group scoring is detected through `env.rubric`.
- `run_group(...)` asks the v0 env to run and score the group in one call.

`src/prime_rl/orchestrator/trajectories.py` also assumes the v0 rollout shape:

- rollout output has `output["trajectory"]`.
- each step is a `vf.TrajectoryStep`.
- token backfill reconstructs missing step tokens from `prompt` and
  `completion`.
- interleaving consumes per-step token masks, logprobs, routed experts, and
  multimodal sidecars.

## Minimal Adapter Shape

Introduce one protocol-specific adapter object in `prime-rl`:

```python
class EnvAdapter(Protocol):
    name: str
    sampling_args: dict

    @property
    def requires_group_scoring(self) -> bool: ...

    def get_dataset(self, seed: int | None = None) -> Any: ...

    async def run_rollout(
        self,
        *,
        client: vf.ClientConfig,
        model: str,
        example: dict,
        cache_salt: str | None,
        teacher: vf1.ModelConfig | None = None,
    ) -> RolloutView: ...

    async def score_group(
        self,
        *,
        views: list[RolloutView],
    ) -> list[RolloutView]: ...
```

`RolloutView` is a trainer-local view, not a Verifiers public type. Its job is
to give tokenization/interleaving one shape:

```python
class RolloutView(Protocol):
    example_id: str | int | None
    error: object | None
    stop_condition: str | None

    def iter_turns(self) -> Iterable[TurnView]: ...
    def has_env_token_advantages(self) -> bool: ...
```

The v0 implementation wraps `output["trajectory"]`. The v1 implementation wraps
`state["transcript"]` or the live `State` before serialization.

## v1 Environment Adapter

The v1 adapter imports `verifiers.v1 as vf1` and keeps all v1-specific logic in
one file:

```python
class V1EnvAdapter:
    def __init__(self, config: EnvConfig):
        self.env: vf1.Env = vf1.load_environment(config.stripped_id, **config.args)
        self.sampling_args = config.sampling.to_sampling_args()

    @property
    def requires_group_scoring(self) -> bool:
        return self.env.requires_group_scoring

    async def run_rollout(
        self,
        *,
        client: vf.ClientConfig,
        model: str,
        example: dict,
        cache_salt: str | None,
        teacher: vf1.ModelConfig | None = None,
    ) -> RolloutView:
        task = self.env.taskset.task_from_row(example)
        model_config = vf1.ModelConfig(
            name=model,
            client=vf1.ClientConfig.model_validate(client.model_dump()),
            sampling=self._sampling_args_with_salt(cache_salt),
        )
        state = await self.env.run_rollout(
            task,
            model=model_config,
            teacher=teacher,
        )
        return V1RolloutView(task=task, state=state)

    async def score_group(self, *, views: list[RolloutView]) -> list[RolloutView]:
        v1_views = cast(list[V1RolloutView], views)
        self.env.score_group(
            tasks=[view.task for view in v1_views],
            states=[view.state for view in v1_views],
        )
        return v1_views
```

The exact `ClientConfig` conversion should use the current concrete type names
in `prime-rl` and `verifiers.v1`, but the rule is simple: convert config data at
the boundary, never put live clients into `State`.

## Dataset And Task Rows

v1 `Taskset` owns row-to-task construction. `prime-rl` should keep train/eval
sampling as row dictionaries and ask the adapter to realize each row:

```python
task = env.taskset.task_from_row(example)
```

That lets dynamic task construction remain taskset-owned. If an environment
needs task-local setup from a server, that setup should happen inside the v1
rollout lifecycle and write serializable data to `state.extras`, not into the
trainer row.

## Transcript Tokenization

Move the current trajectory functions behind turn views:

```python
class TurnView(Protocol):
    prompt: vf1.Messages
    completion: vf1.Messages
    tokens: vf1.TurnTokens | None
    reward: float | None
    prompt_advantages: list[float] | None
    completion_advantages: list[float] | None
```

Then the existing renderer/tokenizer logic can be shared:

- v0 adapter maps `TrajectoryStep.prompt` and `TrajectoryStep.completion`.
- v1 adapter maps `Turn.prompt` and `Turn.completion`.
- token backfill stays a trainer concern when the env did not return tokens.

This keeps `transcript` canonical for v1 and avoids emitting a derived
`trajectory` field from v1 just to satisfy `prime-rl`.

## Advantages

v1 environments may provide token-level advantages directly:

- `Turn.tokens.prompt_advantages`
- `Turn.tokens.completion_advantages`

`prime-rl` should treat those as authoritative. Trainer-side advantage
computation should run only for rollout groups that do not already contain
environment-provided token advantages.

The minimal precedence rule is:

1. token advantages on turns win;
2. otherwise use v1 state-level rewards/advantages if the selected trainer
   algorithm supports scalar rollout advantages;
3. otherwise compute advantages in `prime-rl`.

The v1 adapter should expose this as one method on `RolloutView`, so the trainer
does not inspect Verifiers internals:

```python
if all(view.has_env_token_advantages() for view in group):
    samples = interleave_rollout_view(view, use_env_advantages=True)
else:
    compute_trainer_advantages(group)
```

Mixed groups should fail fast. A group where some rollouts have env token
advantages and others do not is ambiguous for normalization.

## Teacher Flow

Model and teacher should be symmetric config data:

```python
student = vf1.ModelConfig(
    name=model_name,
    client=client_config,
    sampling=student_sampling,
)
teacher = vf1.ModelConfig(
    name=teacher_model_name,
    client=teacher_client_config,
    sampling=teacher_sampling,
)
```

`prime-rl` should pass the teacher config into the v1 adapter when a run
requests teacher behavior. The adapter passes it to `env.run_rollout(...)`.

No live teacher client should be stored in `State`. The v1 harness resolves the
client from `ModelConfig` inside the live context.

## Group Scoring

v1 does not need `run_group`. `prime-rl` can keep its current scheduling model:

1. launch `N` rollouts for the same example;
2. collect `N` `V1RolloutView`s;
3. call `adapter.score_group(views=group)`;
4. interleave/tokenize the scored views.

This matches v1's `run_rollout` plus `score_group` contract and keeps group
advantage overrides in the environment.

The main tradeoff is runtime lifetime. Today v1 closes rollout runtimes before
`score_group`. If a group reward needs live sandboxes, v1 will need an internal
group lifecycle owner before `prime-rl` can support that case.

## Env Server Process

The first v1 path should run in-process inside the `prime-rl` env worker rather
than forcing v1 through the v0 ZMQ `RolloutInput` API. That keeps the v1 state
contract intact and avoids inventing a trajectory compatibility layer.

If v1 needs a remote env server later, add a v1-specific RPC surface that sends:

- serialized task rows or tasks,
- serialized `ModelConfig` for student and optional teacher,
- serialized `State` outputs with `transcript`.

Do not route v1 through v0 `vf.RolloutInput`.

## Minimal Diff Plan

1. Add `V0EnvAdapter` around the current `Env` implementation.
2. Add `V1EnvAdapter` that imports `verifiers.v1 as vf1` and returns
   `RolloutView`s.
3. Replace `REQUIRED_STATE_COLUMNS = ["trajectory"]` with adapter-owned output
   requirements.
4. Move `backfill_rollout_tokens` and `interleave_rollout` onto
   `RolloutView`/`TurnView` inputs.
5. Add teacher config plumbing next to the existing student model/client
   config.
6. Add advantage precedence handling before trainer-side advantage computation.
7. Add tests with one v0 env and three v1 envs: simple rollout, group scoring,
   and env-provided token advantages.

## Open Design Tensions

- Runtime-backed group rewards need a v1 grouped lifetime owner. `prime-rl`
  should not solve that by keeping private runtime handles.
- v1 dynamic task setup should remain taskset/harness-owned. A future task
  buffer can expose richer task production, but the first adapter should not add
  a task-server concept.
- Trainer-local views reduce branching, but they are still another small type
  surface in `prime-rl`. The alternative is direct `trajectory`/`transcript`
  branching in tokenizer code, which will be harder to keep correct.
