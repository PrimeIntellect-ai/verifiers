class Error(Exception):
    """Base class for all errors."""


class ModelError(Error):
    """Used to catch errors while interacting with the model."""

    pass


class InvalidModelResponseError(ModelError):
    """Used to catch empty or invalid model responses."""

    pass


class EmptyModelResponseError(InvalidModelResponseError):
    """Used to catch empty model responses."""

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


class TunnelError(InfraError):
    """Raised when a tunnel process dies or becomes unreachable."""

    pass


class SandboxError(InfraError):
    """Used to catch errors while interacting with sandboxes."""

    pass


class BrowserSandboxError(SandboxError):
    """Used to catch errors while interacting with browser sandboxes."""

    pass


class KernelProtocolError(Error):
    """Raised by the multi-agent kernel on protocol violations
    (wrong agent, duplicate submission, finished episode, etc.).

    Subclass of vf.Error so rollout-layer vf.Error boundaries can
    distinguish protocol violations from other framework errors.
    """

    pass


class ContentParseError(KernelProtocolError):
    """Raised when model output violates the channel-markup contract
    (nested/unclosed/multiple ``<think>`` or configured private tag).

    Distinct from kernel-state protocol violations: apply_action
    quarantines these on the utterance (empty public_channel +
    ``parse_error`` flag) rather than aborting the rollout, so a
    single agent's formatting slip does not DoS the whole episode.
    """

    pass
