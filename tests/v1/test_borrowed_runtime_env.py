import asyncio
from collections.abc import Awaitable, Callable

import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.runtimes import (
    BaseRuntimeInfo,
    DockerConfig,
    ProgramResult,
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
)
from verifiers.v1.session import RolloutLimits
from verifiers.v1.trace import Trace


class _Runtime(Runtime):
    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__("shared")
        self.config = config
        self.info = BaseRuntimeInfo(borrowed=True)
        self.seen: list[dict[str, str]] = []

    async def start(self) -> None:
        pass

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        del argv
        self.seen.append(self.process_env(env))
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def _read(self, path: str) -> bytes:
        raise FileNotFoundError(path)

    async def write(self, path: str, data: bytes) -> None:
        del path, data


class _Data(vf.TaskData):
    tenant: str


class _Task(vf.Task[_Data]):
    def __init__(
        self,
        data: _Data,
        after_capture: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(data)
        self.after_capture = after_capture

    def runtime_env(self) -> dict[str, str]:
        return {"TENANT": self.data.tenant, "ORDER": "task"}

    async def setup(self, trace: Trace, runtime: Runtime) -> None:
        del trace
        await runtime.run(["capture"], {"ORDER": "command"})
        await self.after_capture()


class _Harness(Harness[HarnessConfig]):
    async def setup(self, runtime: Runtime) -> None:
        del runtime
        raise RuntimeError("stop after task setup")

    async def launch(self, *args, **kwargs) -> ProgramResult:
        del args, kwargs
        return ProgramResult(exit_code=0, stdout="", stderr="")


def _rollout(
    task: _Task,
    runtime: _Runtime,
    runtime_config: RuntimeConfig | None = None,
) -> Rollout:
    harness = _Harness(HarnessConfig(id="null"))
    return Rollout(
        task=task,
        agent_config=vf.AgentConfig(harness=harness.config, runtime=runtime.config),
        harness=harness,
        ctx=ModelContext(model="test", client=EvalClientConfig()),
        runtime_config=runtime_config or runtime.config,
        runtime=runtime,
        timeouts=RolloutTimeouts(setup=1, finalize=1, scoring=1),
        limits=RolloutLimits(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        SubprocessConfig(),
        DockerConfig(allow=[]),
        DockerConfig(allow=["example.com"]),
    ],
)
async def test_matching_policy_borrowers_keep_independent_environments(
    config: RuntimeConfig,
) -> None:
    runtime = _Runtime(config)
    runtime.env = {"OWNER": "private"}
    both_entered = asyncio.Event()
    entered = 0

    async def overlap() -> None:
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await both_entered.wait()

    rollouts = [
        _rollout(_Task(_Data(idx=i, prompt="test", tenant=tenant), overlap), runtime)
        for i, tenant in enumerate(("alpha", "beta"))
    ]
    await asyncio.wait_for(
        asyncio.gather(*(rollout.open() for rollout in rollouts)), timeout=1
    )
    await asyncio.gather(*(rollout.abort() for rollout in rollouts))

    assert runtime.env == {"OWNER": "private"}
    assert {env["TENANT"] for env in runtime.seen} == {"alpha", "beta"}
    assert all(env["ORDER"] == "command" for env in runtime.seen)
    assert all("OWNER" not in env for env in runtime.seen)


@pytest.mark.asyncio
async def test_different_policy_borrowers_remain_serialized() -> None:
    runtime = _Runtime(DockerConfig(allow=[]))
    runtime.env = {"OWNER": "private"}
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def hold_first() -> None:
        first_entered.set()
        await release_first.wait()

    async def mark_second() -> None:
        second_entered.set()

    first = _rollout(
        _Task(_Data(idx=0, prompt="test", tenant="alpha"), hold_first), runtime
    )
    second = _rollout(
        _Task(_Data(idx=1, prompt="test", tenant="beta"), mark_second),
        runtime,
        DockerConfig(allow=["example.com"]),
    )
    first_open = asyncio.create_task(first.open())
    await first_entered.wait()
    second_open = asyncio.create_task(second.open())
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(second_entered.wait()), timeout=0.05)

    release_first.set()
    await first_open
    await first.abort()
    await asyncio.wait_for(second_entered.wait(), timeout=1)
    await second_open
    await second.abort()

    assert runtime.env == {"OWNER": "private"}


@pytest.mark.asyncio
async def test_runtime_views_share_borrow_policy_state() -> None:
    runtime = _Runtime(DockerConfig(allow=[]))
    view = runtime.with_env({"VIEW": "one"})
    first_policy = await view.acquire_borrow(view.config)
    second_entered = asyncio.Event()

    async def acquire_different_policy() -> None:
        policy = await runtime.acquire_borrow(DockerConfig(allow=["example.com"]))
        second_entered.set()
        await runtime.release_borrow(policy)

    second = asyncio.create_task(acquire_different_policy())
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(second_entered.wait()), timeout=0.05)

    await view.release_borrow(first_policy)
    await asyncio.wait_for(second_entered.wait(), timeout=1)
    await second
