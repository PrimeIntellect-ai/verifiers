"""A tool method named like a ServerBase attribute fails at class definition."""

import types

import pytest

import verifiers.v1 as vf


def test_tool_named_run_fails_at_class_definition() -> None:
    """The issue-#2937 repro: `run` replaces the classmethod that starts the server."""
    with pytest.raises(TypeError, match="run"):

        class GameToolset(vf.Toolset[vf.ToolsetConfig]):
            TOOL_PREFIX = None

            @vf.tool
            def run(self, command: str) -> str:
                """Run a command."""
                return command


@pytest.mark.parametrize(
    "name", ["run", "setup", "setup_task", "register", "state", "config"]
)
def test_tools_may_not_shadow_server_attributes(name: str) -> None:
    @vf.tool
    def handler(self) -> str:
        """A tool with a reserved name."""
        return "x"

    with pytest.raises(TypeError, match=name):
        types.new_class(
            "CustomToolset",
            (vf.Toolset[vf.ToolsetConfig],),
            {},
            lambda namespace: namespace.update({name: handler}),
        )


def test_renamed_method_keeps_the_exposed_tool_name() -> None:
    """The workaround from the issue keeps working: rename the method, keep the name."""

    class GameToolset(vf.Toolset[vf.ToolsetConfig]):
        TOOL_PREFIX = None

        @vf.tool(name="run")
        def execute(self, command: str) -> str:
            """Run a command."""
            return command

        @vf.tool
        def echo(self, text: str) -> str:
            """Echo the text."""
            return text

    toolset = GameToolset(vf.ToolsetConfig())
    assert toolset.execute.tool_name == "run"
    assert toolset.echo("hi") == "hi"
