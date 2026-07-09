"""Task construction: data + config in, `Taskset.tasks()` assembly, and `server_config`.

Pins the behaviors the split relies on: a `Task` is a plain class built from its two
inputs (`MyTask(data, config=...)` — config defaults to the declared type's defaults),
`tasks()` wraps every loaded `TaskData` row in the declared Task with `TasksetConfig.task`,
only the data rides the wire (`trace.task`), and `from_trace` rebuilds the declared data
type (configs come from whoever spawns the task).
"""

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import TaskError
from verifiers.v1.trace import Trace


class PlainData(vf.TaskData):
    answer: str = ""


class PlainTaskConfig(vf.TaskConfig):
    greeting: str = "hi"


class PlainTask(vf.Task[PlainData, vf.State, PlainTaskConfig]):
    pass


class PlainConfig(vf.TasksetConfig):
    task: PlainTaskConfig = PlainTaskConfig()


class PlainTaskset(vf.Taskset[PlainTask, PlainConfig]):
    def load(self) -> list[PlainData]:
        return [PlainData(idx=i, prompt="p") for i in range(2)]


def test_task_is_built_from_data_and_config() -> None:
    data = PlainData(idx=0, prompt="p", answer="42")
    task = PlainTask(data)
    assert task.data is data
    assert task.config.greeting == "hi"  # declared-type defaults
    task = PlainTask(data, config=PlainTaskConfig(greeting="yo"))
    assert task.config.greeting == "yo"


def test_undefaultable_config_fails_at_construction() -> None:
    class NeedsValue(vf.TaskConfig):
        required: str

    class NeedyTask(vf.Task[PlainData, vf.State, NeedsValue]):
        pass

    data = PlainData(idx=0, prompt="p")
    with pytest.raises(TaskError, match="pass `config=...`"):
        NeedyTask(data)
    assert NeedyTask(data, config=NeedsValue(required="x")).config.required == "x"


def test_tasks_constructs_the_declared_type() -> None:
    taskset = PlainTaskset(PlainConfig(id="plain", task=PlainTaskConfig(greeting="yo")))
    tasks = taskset.tasks()
    assert all(type(task) is PlainTask for task in tasks)
    assert all(task.config is taskset.config.task for task in tasks)
    assert [task.data.idx for task in tasks] == [0, 1]


def test_tasks_rejects_mismatched_rows_and_config() -> None:
    class OtherData(vf.TaskData):
        pass

    class WrongRows(PlainTaskset):
        def load(self) -> list[vf.TaskData]:
            return [OtherData(idx=0, prompt="p")]

    with pytest.raises(TaskError, match="one task type"):
        WrongRows(PlainConfig(id="plain")).tasks()

    class OtherConfig(vf.TasksetConfig):
        pass  # `task` stays the base TaskConfig — not the declared PlainTaskConfig

    with pytest.raises(TaskError, match="narrow the config field"):
        PlainTaskset(OtherConfig(id="plain")).tasks()


def test_only_the_data_rides_the_wire() -> None:
    task = PlainTaskset(PlainConfig(id="plain")).tasks()[0]
    trace = Trace(task=task.data, state=vf.State())
    dump = trace.model_dump()
    assert dump["task"] == task.data.model_dump()  # pure data — no behavior, no config
    rebuilt = PlainData.model_validate(dump["task"])  # replay's rebuild path
    assert rebuilt == task.data


def test_from_trace_rebuilds_the_declared_data() -> None:
    data = PlainData(idx=0, prompt="p", answer="42")
    trace = Trace(task=data, state=vf.State())
    derived = PlainTask.from_trace(trace)
    assert derived.data == data
    assert derived.config.greeting == "hi"  # data-only trace: config from the defaults
    override = PlainTaskConfig(greeting="hey")
    assert PlainTask.from_trace(trace, config=override).config is override


# ---- server_config pairing -----------------------------------------------------


class PairToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "pair"


class CustomToolsetConfig(vf.ToolsetConfig):
    corpus: str = "default"


class CustomToolset(vf.Toolset[CustomToolsetConfig]):
    TOOL_PREFIX = "custom"


DATA = PlainData(idx=0, prompt="p")


def test_server_config_exact_type_match_wins() -> None:
    class Config(vf.TaskConfig):
        tools: vf.ToolsetConfig = vf.ToolsetConfig(colocated=True)
        custom: CustomToolsetConfig = CustomToolsetConfig()  # subclass: not exact

    class PairedTask(vf.Task[PlainData, vf.State, Config]):
        tools = (PairToolset,)

    task = PairedTask(DATA)
    assert task.server_config(PairToolset) is task.config.tools
    (server,) = task.tool_servers()
    assert isinstance(server, PairToolset) and server.config.colocated


def test_server_config_unique_isinstance_match() -> None:
    class Config(vf.TaskConfig):
        custom: CustomToolsetConfig = CustomToolsetConfig(corpus="wiki")

    class PairedTask(vf.Task[PlainData, vf.State, Config]):
        tools = (PairToolset,)

    # PairToolset declares the base ToolsetConfig; the subclass instance is the
    # unique isinstance match.
    task = PairedTask(DATA)
    assert task.server_config(PairToolset) is task.config.custom


def test_server_config_defaults_when_nothing_matches() -> None:
    class PairedTask(vf.Task):
        tools = (PairToolset,)

    assert PairedTask(DATA).server_config(PairToolset) == vf.ToolsetConfig()


def test_server_config_ambiguity_raises() -> None:
    class Config(vf.TaskConfig):
        first: vf.ToolsetConfig = vf.ToolsetConfig()
        second: vf.ToolsetConfig = vf.ToolsetConfig()

    class PairedTask(vf.Task[PlainData, vf.State, Config]):
        tools = (PairToolset,)

    with pytest.raises(TaskError, match="ambiguous"):
        PairedTask(DATA).server_config(PairToolset)


# ---- taskset-scoped (shared) tools ----------------------------------------------


class SharedCorpusToolset(vf.Toolset[vf.SharedToolsetConfig]):
    TOOL_PREFIX = "corpus"


def test_taskset_scoped_tools_build_from_the_taskset_config() -> None:
    class Config(vf.TasksetConfig):
        tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig(url="http://corpus")
        task: PlainTaskConfig = PlainTaskConfig()

    class SharedTaskset(vf.Taskset[PlainTask, Config]):
        tools = (SharedCorpusToolset,)

        def load(self) -> list[PlainData]:
            return [PlainData(idx=0, prompt="p")]

    taskset = SharedTaskset(Config(id="x"))
    (server,) = taskset.tool_servers()
    assert isinstance(server, SharedCorpusToolset)
    assert server.config is taskset.config.tools  # exact-type match, taskset level
    # scope is the registration site: the tasks themselves declare no tools
    assert taskset.tasks()[0].tool_servers() == []
