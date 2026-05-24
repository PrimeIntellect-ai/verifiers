import re
import sys
from types import ModuleType

import pytest
from pydantic import BaseModel

import verifiers.v1 as vf


REF_MODULE = "v1_taskset_binding_refs"
ref_module = ModuleType(REF_MODULE)
sys.modules[REF_MODULE] = ref_module


def ref(name: str) -> str:
    return f"{REF_MODULE}:{name}"


def dynamic_ref(value: object) -> str:
    name = getattr(value, "__name__", type(value).__name__)
    ref_name = f"{name}_{id(value)}"
    setattr(ref_module, ref_name, value)
    return ref(ref_name)


def config_value(value: object) -> object:
    if callable(value):
        return dynamic_ref(value)
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, dict):
        return {str(key): config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [config_value(item) for item in value]
    if isinstance(value, tuple):
        return [config_value(item) for item in value]
    return value


def config_data(config: object | None) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        return config.model_dump(exclude_none=True)
    if isinstance(config, dict):
        return {str(key): item for key, item in config.items()}
    raise TypeError("test config must be a mapping or config object")


def make_taskset(config: object | None = None, **values: object) -> vf.Taskset:
    data = {**config_data(config), **values}
    return vf.Taskset(config=vf.TasksetConfig.model_validate(config_value(data)))


def source_rows() -> list[dict[str, object]]:
    return [
        {
            "prompt": [{"role": "user", "content": "reply ok"}],
            "answer": "ok",
        }
    ]


def prefixed_source_rows() -> list[dict[str, object]]:
    return [
        {
            "prompt": [{"role": "user", "content": "reply ok"}],
            "answer": "ok",
            "prefix": "bound:",
        }
    ]


def two_prefixed_source_rows() -> list[dict[str, object]]:
    return [
        {
            "prompt": [{"role": "user", "content": "reply ok"}],
            "answer": "ok",
            "prefix": "first:",
        },
        {
            "prompt": [{"role": "user", "content": "reply ok"}],
            "answer": "ok",
            "prefix": "second:",
        },
    ]


class Prefixer:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, value: str) -> str:
        return f"{self.prefix}{value}"


class TagExtractor:
    def __init__(self, tag: str):
        self.pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)

    def __call__(self, completion: list[dict[str, object]]) -> str:
        message = vf.get_messages(completion, role="assistant")[-1]
        match = self.pattern.search(str(message.content or ""))
        return "" if match is None else match.group(1).strip()


prefixer_factory_calls = 0


def load_factory_prefixer() -> Prefixer:
    global prefixer_factory_calls
    prefixer_factory_calls += 1
    return Prefixer("factory:")


def load_config_prefixer() -> Prefixer:
    return Prefixer("config:")


def load_defaulted_prefixer(prefix: str = "defaulted:") -> Prefixer:
    return Prefixer(prefix)


def load_bound_prefixer(prefix: str) -> Prefixer:
    return Prefixer(prefix)


def load_answer_extractor() -> TagExtractor:
    return TagExtractor("answer")


@vf.reward
async def prefix_reward(state, prefixer) -> float:
    state["prefixed"] = prefixer("ok")
    return 1.0


@vf.reward
async def framework_state_reward(state) -> float:
    state["framework_state_seen"] = True
    return 1.0


@vf.setup
async def setup_with_override(state, token) -> None:
    state["token"] = token


@vf.reward
async def missing_binding_reward(state, extractor) -> float:
    _ = state, extractor
    return 0.0


@vf.reward
async def extracted_answer_reward(task, state, extract_answer) -> float:
    response = extract_answer(state.get("completion") or [])
    return float(response == task["answer"])


async def score_taskset(taskset: vf.Taskset) -> vf.State:
    env = vf.Env(taskset=taskset, harness=vf.Harness(config=vf.HarnessConfig()))
    task = next(iter(taskset))
    state = await env.harness.setup_state(task, vf.State.for_task(task))
    await env.harness.runtime.score_rollout(task, state)
    return state


@pytest.mark.asyncio
async def test_taskset_object_binding_rejects_live_instance() -> None:
    taskset = make_taskset(
        source=source_rows,
        rewards=[prefix_reward],
        objects={"prefixer": Prefixer("inst:")},
        bindings={"prefix_reward.prefixer": "objects.prefixer"},
    )

    with pytest.raises(TypeError, match="factory function or class"):
        await score_taskset(taskset)


@pytest.mark.asyncio
async def test_taskset_object_factory_is_lazy_and_resolved_once() -> None:
    global prefixer_factory_calls
    prefixer_factory_calls = 0

    taskset = make_taskset(
        source=source_rows,
        rewards=[prefix_reward],
        objects={"prefixer": dynamic_ref(load_factory_prefixer)},
        bindings={"prefix_reward.prefixer": "objects.prefixer"},
    )
    env = vf.Env(taskset=taskset, harness=vf.Harness(config=vf.HarnessConfig()))
    task = next(iter(taskset))
    state = await env.harness.setup_state(task, vf.State.for_task(task))

    await env.harness.runtime.score_rollout(task, state)
    await env.harness.runtime.score_rollout(task, state)

    assert prefixer_factory_calls == 1
    assert state["prefixed"] == "factory:ok"


@pytest.mark.asyncio
async def test_taskset_object_factory_accepts_defaulted_arguments() -> None:
    taskset = make_taskset(
        source=source_rows,
        rewards=[prefix_reward],
        objects={"prefixer": dynamic_ref(load_defaulted_prefixer)},
        bindings={"prefix_reward.prefixer": "objects.prefixer"},
    )

    state = await score_taskset(taskset)

    assert state["prefixed"] == "defaulted:ok"


@pytest.mark.asyncio
async def test_taskset_object_factory_accepts_bound_arguments() -> None:
    taskset = make_taskset(
        source=prefixed_source_rows,
        rewards=[prefix_reward],
        objects={"prefixer": dynamic_ref(load_bound_prefixer)},
        bindings={
            "prefixer.prefix": "task.prefix",
            "prefix_reward.prefixer": "objects.prefixer",
        },
    )

    state = await score_taskset(taskset)

    assert state["prefixed"] == "bound:ok"


@pytest.mark.asyncio
async def test_taskset_object_factory_bindings_are_rollout_scoped() -> None:
    taskset = make_taskset(
        source=two_prefixed_source_rows,
        rewards=[prefix_reward],
        objects={"prefixer": dynamic_ref(load_bound_prefixer)},
        bindings={
            "prefixer.prefix": "task.prefix",
            "prefix_reward.prefixer": "objects.prefixer",
        },
    )
    env = vf.Env(taskset=taskset, harness=vf.Harness(config=vf.HarnessConfig()))
    first, second = list(taskset)
    first_state = await env.harness.setup_state(first, vf.State.for_task(first))
    second_state = await env.harness.setup_state(second, vf.State.for_task(second))

    await env.harness.runtime.score_rollout(first, first_state)
    await env.harness.runtime.score_rollout(second, second_state)

    assert first_state["prefixed"] == "first:ok"
    assert second_state["prefixed"] == "second:ok"


@pytest.mark.asyncio
async def test_taskset_object_factory_rejects_unbound_arguments() -> None:
    taskset = make_taskset(
        source=prefixed_source_rows,
        rewards=[prefix_reward],
        objects={"prefixer": dynamic_ref(load_bound_prefixer)},
        bindings={"prefix_reward.prefixer": "objects.prefixer"},
    )

    with pytest.raises(TypeError, match="unbound factory arguments"):
        await score_taskset(taskset)


@pytest.mark.asyncio
async def test_framework_args_win_over_taskset_bindings() -> None:
    taskset = make_taskset(
        source=source_rows,
        rewards=[framework_state_reward],
        bindings={"framework_state_reward.state": "objects.missing"},
    )

    state = await score_taskset(taskset)

    assert state["framework_state_seen"] is True


@pytest.mark.asyncio
async def test_caller_kwargs_win_over_taskset_bindings_for_handlers() -> None:
    taskset = make_taskset(
        source=source_rows,
        setups=[setup_with_override],
        objects={"token": lambda: "bound"},
        bindings={"setup_with_override.token": "objects.token"},
    )
    env = vf.Env(taskset=taskset, harness=vf.Harness(config=vf.HarnessConfig()))
    task = next(iter(taskset))
    state = vf.State.for_task(task)

    await env.harness.runtime.run_rollout_handlers(
        [setup_with_override], task=task, state=state, token="explicit"
    )

    assert state["token"] == "explicit"


@pytest.mark.asyncio
async def test_missing_taskset_binding_error_names_signal_and_arg() -> None:
    taskset = make_taskset(source=source_rows, rewards=[missing_binding_reward])

    with pytest.raises(
        TypeError,
        match="reward signal 'missing_binding_reward'.*extractor",
    ):
        await score_taskset(taskset)


@pytest.mark.asyncio
async def test_taskset_config_map_round_trips_objects_and_bindings() -> None:
    config = vf.TasksetConfig(
        objects={"prefixer": dynamic_ref(load_config_prefixer)},
        bindings={"prefix_reward.prefixer": "objects.prefixer"},
    )
    taskset = make_taskset(
        source=source_rows,
        rewards=[prefix_reward],
        config=config,
    )

    state = await score_taskset(taskset)

    assert state["prefixed"] == "config:ok"


@pytest.mark.asyncio
async def test_taskset_bindings_support_shared_extractor_pattern() -> None:
    taskset = make_taskset(
        source=source_rows,
        rewards=[extracted_answer_reward],
        objects={"extract_answer": dynamic_ref(load_answer_extractor)},
        bindings={"extracted_answer_reward.extract_answer": "objects.extract_answer"},
    )
    env = vf.Env(taskset=taskset, harness=vf.Harness(config=vf.HarnessConfig()))
    task = next(iter(taskset))
    state = await env.harness.setup_state(task, vf.State.for_task(task))
    state["completion"] = [{"role": "assistant", "content": "<answer>ok</answer>"}]

    await env.harness.runtime.score_rollout(task, state)

    assert state["reward"] == 1.0
