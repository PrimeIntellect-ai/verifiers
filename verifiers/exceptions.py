class Error(Exception): ...


class ToolError(Error):
    message: str
    cause: Exception

    def __init__(self, message: str, cause: Exception):
        self.message = message
        self.cause = cause
