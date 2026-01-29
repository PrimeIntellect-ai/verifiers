"""Protocol defining the interface for browser mode implementations."""

from typing import Protocol, Any, TYPE_CHECKING
import verifiers as vf

if TYPE_CHECKING:
    from ..browser_env import BrowserEnv


class BrowserMode(Protocol):
    """Protocol that all browser modes must implement."""

    def register_tools(self, env: "BrowserEnv") -> None:
        """Register mode-specific tools with the environment."""
        ...

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """Create session and initialize state for this mode."""
        ...

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inject mode-specific arguments into tool calls."""
        ...

    async def cleanup_session(self, state: vf.State) -> None:
        """Clean up session after rollout."""
        ...

    async def teardown(self) -> None:
        """Clean up resources on environment teardown."""
        ...

    def filter_screenshots_in_messages(self, messages: list) -> list:
        """Filter screenshots in messages. Optional - defaults to returning messages unchanged."""
        ...
