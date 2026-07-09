"""The attached taskset config: `Taskset.tasks()` -> `Task.config`, and `server_config`.

Pins the behaviors the wiring relies on: the attachment is a pydantic private attr, so it
must survive `model_copy` (the judge bake), `model_copy(deep=True)` (replay's per-rescore
copies) and `copy.deepcopy`, and be dropped by a wire round-trip (`model_validate(model_dump())`)
— which is why `replay` re-attaches after rebuilding tasks.
"""

import copy

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import TasksetError


class PlainTask(vf.Task):
    pass


class PlainConfig(vf.TasksetConfig):
    greeting: str = "hi"


class PlainTaskset(vf.Taskset[PlainTask, PlainConfig]):
    def load(self) -> list[PlainTask]:
        return [PlainTask(idx=i, prompt="p") for i in range(2)]


def test_tasks_attaches_the_taskset_config() -> None:
    taskset = PlainTaskset(PlainConfig(id="plain"))
    tasks = taskset.tasks()
    assert all(task.config is taskset.config for task in tasks)
    assert tasks[0].config.greeting == "hi"


def test_unattached_config_raises_with_the_remedy() -> None:
    task = PlainTask(idx=0, prompt="p")
    with pytest.raises(TasksetError, match="attach_config"):
        _ = task.config


def test_attachment_survives_copies_but_not_the_wire() -> None:
    config = PlainConfig(id="plain")
    task = PlainTask(idx=0, prompt="p").attach_config(config)
    assert task.model_copy(update={"name": "n"}).config is config  # judge bake
    assert task.model_copy(deep=True).config == config  # replay rescore copies
    assert copy.deepcopy(task).config == config
    rebuilt = PlainTask.model_validate(task.model_dump())  # the wire drops it...
    with pytest.raises(TasksetError):
        _ = rebuilt.config
    assert rebuilt.attach_config(config).config is config  # ...and replay re-attaches


# ---- server_config pairing -----------------------------------------------------


class PairToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "pair"


class CustomToolsetConfig(vf.ToolsetConfig):
    corpus: str = "default"


class CustomToolset(vf.Toolset[CustomToolsetConfig]):
    TOOL_PREFIX = "custom"


class PairedTask(vf.Task):
    tools = (PairToolset,)


def test_server_config_exact_type_match_wins() -> None:
    class Config(vf.TasksetConfig):
        tools: vf.ToolsetConfig = vf.ToolsetConfig(shared=True)
        custom: CustomToolsetConfig = CustomToolsetConfig()  # subclass: not exact

    task = PairedTask(idx=0, prompt="p").attach_config(Config(id="x"))
    assert task.server_config(PairToolset) is task.config.tools
    (server,) = task.tool_servers()
    assert isinstance(server, PairToolset) and server.config.shared


def test_server_config_unique_isinstance_match() -> None:
    class Config(vf.TasksetConfig):
        custom: CustomToolsetConfig = CustomToolsetConfig(corpus="wiki")

    # PairToolset declares the base ToolsetConfig; the subclass instance is the
    # unique isinstance match.
    task = PairedTask(idx=0, prompt="p").attach_config(Config(id="x"))
    assert task.server_config(PairToolset) is task.config.custom


def test_server_config_defaults_when_nothing_matches() -> None:
    task = PairedTask(idx=0, prompt="p").attach_config(vf.TasksetConfig(id="x"))
    assert task.server_config(PairToolset) == vf.ToolsetConfig()
    # ... and for a standalone task with no config attached at all.
    assert PairedTask(idx=0, prompt="p").server_config(PairToolset) == vf.ToolsetConfig()


def test_server_config_ambiguity_raises() -> None:
    class Config(vf.TasksetConfig):
        first: vf.ToolsetConfig = vf.ToolsetConfig()
        second: vf.ToolsetConfig = vf.ToolsetConfig()

    task = PairedTask(idx=0, prompt="p").attach_config(Config(id="x"))
    with pytest.raises(TasksetError, match="ambiguous"):
        task.server_config(PairToolset)
