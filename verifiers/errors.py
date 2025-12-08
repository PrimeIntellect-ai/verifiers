class Error(Exception):
    """Base class for all errors."""

    def __init__(self, cause: Exception):
        self.cause = cause

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cause={repr(self.cause)})"


class ToolError(Error):
    """Parent class for all tool errors."""

    pass


class ToolParseError(ToolError):
    """Used to catch errors while parsing tool calls."""

    pass


class ToolCallError(ToolError):
    """Used to catch errors while calling tools."""

    pass
