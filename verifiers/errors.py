class Error(Exception):
    """Base class for all errors."""

    def __init__(self, cause: Exception):
        self.cause = cause

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.cause!r})"


class ModelError(Error):
    """Used to catch errors while interacting with the model."""

    pass


class EmptyModelResponseError(ModelError):
    """Used to catch empty or invalid model responses (e.g. response.choices is None)."""

    pass


class OverlongPromptError(Error):
    """Used to catch overlong prompt errors (e.g. prompt + requested number of tokens exceeds model context length)"""

    pass


class ToolError(Error):
    """Parent class for all tool errors."""

    pass


class ToolParseError(ToolError):
    """Used to catch errors while parsing tool calls."""

    pass


class ToolCallError(ToolError):
    """Used to catch errors while calling tools."""

    pass


class InfraError(Error):
    """Used to catch errors while interacting with infrastructure."""

    pass


class SandboxError(InfraError):
    """Used to catch errors while interacting with sandboxes."""

    pass
