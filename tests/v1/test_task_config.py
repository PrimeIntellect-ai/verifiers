"""The task config: constructor injection, `Taskset.tasks()` stamping, and `server_config`.

Pins the behaviors the wiring relies on: `config` is a regular (excluded) field — injected
at construction like `Taskset(config)` / `Judge(config)`, defaulted to the declared type's
defaults, stamped by `tasks()` from `TasksetConfig.task`, never serialized (which is why
`replay` re-stamps after rebuilding tasks from the wire), and inherited by `from_trace`.
"""

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import TaskError
from verifiers.v1.trace import Trace


class PlainTaskConfig(vf.TaskConfig):
    greeting: str = "hi"


class PlainTask(vf.Task[vf.State, PlainTaskConfig]):
    pass


class PlainConfig(vf.TasksetConfig):
    task: PlainTaskConfig = PlainTaskConfig()


class PlainTaskset(vf.Taskset[PlainTask, PlainConfig]):
    def load(self) -> list[PlainTask]:
        return [PlainTask(idx=i, prompt="p") for i in range(2)]


def test_config_is_injected_or_defaults() -> None:
    assert (
        PlainTask(idx=0, prompt="p").config.greeting == "hi"
    )  # declared-type defaults
    task = PlainTask(idx=0, prompt="p", config=PlainTaskConfig(greeting="yo"))
    assert task.config.greeting == "yo"


def test_undefaultable_config_fails_at_construction() -> None:
    class NeedsValue(vf.TaskConfig):
        required: str

    class NeedyTask(vf.Task[vf.State, NeedsValue]):
        pass

    with pytest.raises(TaskError, match="pass `config=...`"):
        NeedyTask(idx=0, prompt="p")
    needy = NeedyTask(idx=0, prompt="p", config=NeedsValue(required="x"))
    assert needy.config.required == "x"


def test_tasks_stamps_the_config_task_subtree() -> None:
    taskset = PlainTaskset(PlainConfig(id="plain", task=PlainTaskConfig(greeting="yo")))
    tasks = taskset.tasks()
    assert all(task.config is taskset.config.task for task in tasks)
    assert tasks[0].config.greeting == "yo"


def test_tasks_rejects_a_mismatched_task_config() -> None:
    class OtherConfig(vf.TasksetConfig):
        pass  # `task` stays the base TaskConfig — not the declared PlainTaskConfig

    with pytest.raises(TaskError, match="narrow the config field"):
        PlainTaskset(OtherConfig(id="plain")).tasks()


def test_config_never_rides_the_wire() -> None:
    task = PlainTask(idx=0, prompt="p", config=PlainTaskConfig(greeting="yo"))
    dump = task.model_dump()
    assert "config" not in dump
    rebuilt = PlainTask.model_validate(dump)  # replay's rebuild path
    assert rebuilt.config.greeting == "hi"  # back to defaults...
    restamped = rebuilt.model_copy(
        update={"config": task.config}
    )  # ...until re-stamped
    assert restamped.config.greeting == "yo"


def test_from_trace_inherits_or_overrides_the_config() -> None:
    parent = PlainTask(idx=0, prompt="p", config=PlainTaskConfig(greeting="yo"))
    trace = Trace(task=parent, state=vf.State())
    assert PlainTask.from_trace(trace).config.greeting == "yo"  # inherited
    override = PlainTaskConfig(greeting="hey")
    assert PlainTask.from_trace(trace, config=override).config is override


# ---- server_config pairing -----------------------------------------------------


class PairToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "pair"


class CustomToolsetConfig(vf.ToolsetConfig):
    corpus: str = "default"


class CustomToolset(vf.Toolset[CustomToolsetConfig]):
    TOOL_PREFIX = "custom"


def test_server_config_exact_type_match_wins() -> None:
    class Config(vf.TaskConfig):
        tools: vf.ToolsetConfig = vf.ToolsetConfig(shared=True)
        custom: CustomToolsetConfig = CustomToolsetConfig()  # subclass: not exact

    class PairedTask(vf.Task[vf.State, Config]):
        tools = (PairToolset,)

    task = PairedTask(idx=0, prompt="p")
    assert task.server_config(PairToolset) is task.config.tools
    (server,) = task.tool_servers()
    assert isinstance(server, PairToolset) and server.config.shared


def test_server_config_unique_isinstance_match() -> None:
    class Config(vf.TaskConfig):
        custom: CustomToolsetConfig = CustomToolsetConfig(corpus="wiki")

    class PairedTask(vf.Task[vf.State, Config]):
        tools = (PairToolset,)

    # PairToolset declares the base ToolsetConfig; the subclass instance is the
    # unique isinstance match.
    task = PairedTask(idx=0, prompt="p")
    assert task.server_config(PairToolset) is task.config.custom


def test_server_config_defaults_when_nothing_matches() -> None:
    class PairedTask(vf.Task):
        tools = (PairToolset,)

    assert (
        PairedTask(idx=0, prompt="p").server_config(PairToolset) == vf.ToolsetConfig()
    )


def test_server_config_ambiguity_raises() -> None:
    class Config(vf.TaskConfig):
        first: vf.ToolsetConfig = vf.ToolsetConfig()
        second: vf.ToolsetConfig = vf.ToolsetConfig()

    class PairedTask(vf.Task[vf.State, Config]):
        tools = (PairToolset,)

    with pytest.raises(TaskError, match="ambiguous"):
        PairedTask(idx=0, prompt="p").server_config(PairToolset)
