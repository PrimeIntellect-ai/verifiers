import re

import pytest

import verifiers.v1 as vf


def source_rows() -> list[dict[str, object]]:
    return [
        {
            "prompt": [{"role": "user", "content": "reply ok"}],
            "answer": "ok",
        }
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
    env = vf.Env(taskset=taskset, harness=vf.Harness())
    task = next(iter(taskset))
    state = await env.harness.setup_state(task, vf.State.for_task(task))
    await env.harness.runtime.score_rollout(task, state)
    return state


@pytest.mark.asyncio
async def test_taskset_object_binding_resolves_instance() -> None:
    taskset = vf.Taskset(
        source=source_rows,
        rewards=[prefix_reward],
        objects={"prefixer": Prefixer("inst:")},
        bindings={"prefix_reward.prefixer": "objects.prefixer"},
    )

    state = await score_taskset(taskset)

    assert state["prefixed"] == "inst:ok"
    assert state["reward"] == 1.0


@pytest.mark.asyncio
async def test_taskset_object_factory_is_lazy_and_resolved_once() -> None:
    calls = 0

    def make_prefixer() -> Prefixer:
        nonlocal calls
        calls += 1
        return Prefixer("factory:")

    taskset = vf.Taskset(
        source=source_rows,
        rewards=[prefix_reward],
        objects={"prefixer": make_prefixer},
        bindings={"prefix_reward.prefixer": "objects.prefixer"},
    )
    env = vf.Env(taskset=taskset, harness=vf.Harness())
    task = next(iter(taskset))
    state = await env.harness.setup_state(task, vf.State.for_task(task))

    await env.harness.runtime.score_rollout(task, state)
    await env.harness.runtime.score_rollout(task, state)

    assert calls == 1
    assert state["prefixed"] == "factory:ok"


@pytest.mark.asyncio
async def test_framework_args_win_over_taskset_bindings() -> None:
    taskset = vf.Taskset(
        source=source_rows,
        rewards=[framework_state_reward],
        bindings={"framework_state_reward.state": "objects.missing"},
    )

    state = await score_taskset(taskset)

    assert state["framework_state_seen"] is True


@pytest.mark.asyncio
async def test_caller_kwargs_win_over_taskset_bindings_for_handlers() -> None:
    taskset = vf.Taskset(
        source=source_rows,
        setups=[setup_with_override],
        objects={"token": "bound"},
        bindings={"setup_with_override.token": "objects.token"},
    )
    env = vf.Env(taskset=taskset, harness=vf.Harness())
    task = next(iter(taskset))
    state = vf.State.for_task(task)

    await env.harness.runtime.run_rollout_handlers(
        [setup_with_override], task=task, state=state, token="explicit"
    )

    assert state["token"] == "explicit"


@pytest.mark.asyncio
async def test_missing_taskset_binding_error_names_signal_and_arg() -> None:
    taskset = vf.Taskset(source=source_rows, rewards=[missing_binding_reward])

    with pytest.raises(
        TypeError,
        match="reward signal 'missing_binding_reward'.*extractor",
    ):
        await score_taskset(taskset)


@pytest.mark.asyncio
async def test_taskset_config_map_round_trips_objects_and_bindings() -> None:
    config = vf.TasksetConfig(
        objects={"prefixer": Prefixer("config:")},
        bindings={"prefix_reward.prefixer": "objects.prefixer"},
    )
    taskset = vf.Taskset(
        source=source_rows,
        rewards=[prefix_reward],
        config=config,
    )

    state = await score_taskset(taskset)

    assert state["prefixed"] == "config:ok"


@pytest.mark.asyncio
async def test_taskset_bindings_support_shared_extractor_pattern() -> None:
    taskset = vf.Taskset(
        source=source_rows,
        rewards=[extracted_answer_reward],
        objects={"extract_answer": lambda: TagExtractor("answer")},
        bindings={"extracted_answer_reward.extract_answer": "objects.extract_answer"},
    )
    env = vf.Env(taskset=taskset, harness=vf.Harness())
    task = next(iter(taskset))
    state = await env.harness.setup_state(task, vf.State.for_task(task))
    state["completion"] = [{"role": "assistant", "content": "<answer>ok</answer>"}]

    await env.harness.runtime.score_rollout(task, state)

    assert state["reward"] == 1.0
