"""
Browser smoke test environment for Prime Env Hub.

This environment provides a simple browser navigation test using
the BrowserEnv integration from verifiers. It navigates to the
Prime Intellect homepage and verifies the agent can read page content.

Usage:
    prime env install browser-test
    prime eval run browser-test -m gpt-4o-mini
"""

from verifiers.envs.integrations.browser_env import (
    load_environment as load_browser_environment,
)


def load_environment(
    mode: str = "dom",
    max_turns: int = 10,
    judge_model: str = "gpt-4o-mini",
    browserbase_api_key: str | None = None,
    browserbase_project_id: str | None = None,
    stagehand_model: str = "openai/gpt-4o-mini",
    server_url: str = "http://localhost:3000",
    **kwargs,
):
    """
    Load the browser smoke test environment.

    This is a simple smoke test that navigates to the Prime Intellect
    homepage and verifies the agent can read page content.

    Args:
        mode: Browser control mode ("dom" or "cua")
            - "dom": Natural language operations via Stagehand SDK
            - "cua": Vision-based primitives via CUA server
        max_turns: Maximum conversation turns
        judge_model: Model for judging task completion
        browserbase_api_key: Browserbase API key (or set BROWSERBASE_API_KEY env var)
        browserbase_project_id: Browserbase project ID (or set BROWSERBASE_PROJECT_ID env var)
        stagehand_model: Model for Stagehand operations (DOM mode only)
        server_url: CUA server URL (CUA mode only)
        **kwargs: Additional arguments passed to BrowserEnv

    Returns:
        Configured BrowserEnv instance

    Example:
        >>> env = load_environment()  # smoke test with DOM mode
        >>> env = load_environment(mode="cua", server_url="http://localhost:3000")
    """
    return load_browser_environment(
        mode=mode,
        benchmark="smoke_test",
        max_turns=max_turns,
        judge_model=judge_model,
        browserbase_api_key=browserbase_api_key,
        browserbase_project_id=browserbase_project_id,
        stagehand_model=stagehand_model,
        server_url=server_url,
        **kwargs,
    )
