"""
Local Browser Example Environment.

This environment demonstrates browser automation of localhost applications
using the Local CUA mode. It runs a Next.js application alongside a CUA
server in a sandbox, allowing agents to interact with the app using
vision-based primitives.

Key features:
- No internet access - browser can only interact with the localhost app
- No 'goto' tool - browser starts at the app's URL
- Vision-based primitives: click, type, scroll, etc.
- Perfect for testing web application UIs

Usage:
    # Using the example Next.js app (default)
    prime eval run local-browser-example -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY

    # Using a custom app path
    prime eval run local-browser-example -m openai/gpt-4.1-mini -a '{"app_path": "/path/to/your/nextjs/app"}'
"""

from pathlib import Path
from typing import Literal

import verifiers as vf
from verifiers.envs.integrations.browser_env.modes.local_cua_mode import LocalCUAMode
from datasets import Dataset


LOCAL_CUA_SYSTEM_PROMPT = """You are a browser automation agent testing a localhost web application.

You can control the browser using these tools:
- click(x, y, button): Click at coordinates
- double_click(x, y): Double-click at coordinates
- type_text(text): Type text into focused element
- keypress(keys): Press keyboard keys
- scroll(x, y, scroll_x, scroll_y): Scroll at position
- back(): Go back in history
- forward(): Go forward in history
- wait(time_ms): Wait for specified milliseconds
- screenshot(): Capture current page state

IMPORTANT: There is no 'goto' tool. You are testing a localhost application and cannot navigate to external URLs. The browser starts at the application's home page.

After each action, you will receive a screenshot showing the current page state.
Analyze the screenshot to determine your next action.

Complete the given task by interacting with the application's UI elements."""


def create_example_dataset() -> Dataset:
    """
    Create a simple dataset for testing the local browser example.

    This dataset tests basic interactions with the example Next.js app:
    - Finding UI elements
    - Clicking buttons
    - Typing text
    - Navigating between pages
    """
    return Dataset.from_dict(
        {
            "question": [
                "Click the 'Increment' button three times and tell me what the counter shows.",
                "Type 'Hello World' into the text input field and click Submit. What does the submitted value show?",
                "Click on 'Option B' in the Selection Test section. What option is now selected?",
            ],
            "answer": [
                "3",
                "Hello World",
                "Option B",
            ],
            "task_id": [
                "local-counter-test",
                "local-input-test",
                "local-selection-test",
            ],
        }
    )


JUDGE_PROMPT = """You are evaluating a browser automation agent's interaction with a localhost web application.

Task:
```
{question}
```

Expected Answer:
```
{answer}
```

Agent's Response:
```
{response}
```

Did the agent successfully complete the task and provide the correct answer? The answer may be embedded in a longer response or phrased differently, but should convey the same information.

Respond "yes" if the agent completed the task correctly, "no" if not."""


async def judge_answer(
    judge,
    prompt: str | list,
    completion: str | list,
    answer: str,
    state: vf.State,
) -> float:
    """LLM judge reward for evaluating task completion."""
    judge_response = await judge(prompt, completion, answer, state)
    is_correct = "yes" in judge_response.lower()
    return 1.0 if is_correct else 0.0


class LocalBrowserEnv(vf.StatefulToolEnv):
    """
    Environment for testing localhost applications with browser automation.

    This environment uses LocalCUAMode to run both a CUA server and a target
    web application in a sandbox container. The browser starts at the app's
    URL and agents can interact with it using vision-based primitives.
    """

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        max_turns: int = 15,
        system_prompt: str = LOCAL_CUA_SYSTEM_PROMPT,
        # Application configuration
        app_path: str | Path | None = None,
        app_port: int = 3000,
        app_start_command: str = "npm run start",
        app_build_command: str | None = "npm install && npm run build",
        # CUA server configuration
        cua_server_port: int = 3001,
        # Viewport configuration
        viewport_width: int = 1024,
        viewport_height: int = 768,
        # Screenshot configuration
        save_screenshots: bool = False,
        keep_recent_screenshots: int | None = 2,
        # Sandbox configuration
        cpu_cores: int = 2,
        memory_gb: int = 4,
        use_prebuilt_image: bool = True,
        prebuilt_image: str = "team-cmlr3u2er002zhr01tj8f48ts/localbrowserapp:v1.0.1",
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            max_turns=max_turns,
            stop_errors=[vf.SandboxError],
            **kwargs,
        )

        self._local_cua_mode = LocalCUAMode(
            app_path=app_path,
            app_port=app_port,
            app_start_command=app_start_command,
            app_build_command=app_build_command,
            cua_server_port=cua_server_port,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            save_screenshots=save_screenshots,
            keep_recent_screenshots=keep_recent_screenshots,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            use_prebuilt_image=use_prebuilt_image,
            prebuilt_image=prebuilt_image,
        )

        self._rubric = rubric
        self._local_cua_mode.register_tools(self)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Initialize the local browser sandbox."""
        state = await self._local_cua_mode.setup_state(state, **kwargs)
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        """Inject session and sandbox IDs into tool calls."""
        return self._local_cua_mode.update_tool_args(
            tool_name, tool_args, messages, state, **kwargs
        )

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        """Get prompt messages, filtering screenshots."""
        messages = await super().get_prompt_messages(state)
        return self._local_cua_mode.filter_screenshots_in_messages(list(messages))

    @vf.cleanup
    async def cleanup_session(self, state: vf.State) -> None:
        """Clean up the browser session and sandbox."""
        await self._local_cua_mode.cleanup_session(state)

    @vf.teardown
    async def teardown(self) -> None:
        """Clean up all resources."""
        if hasattr(self, "_local_cua_mode") and self._local_cua_mode is not None:
            await self._local_cua_mode.teardown()


def load_environment(
    max_turns: int = 15,
    judge_model: str = "gpt-4o-mini",
    system_prompt: str = LOCAL_CUA_SYSTEM_PROMPT,
    # Application configuration
    app_path: str | Path | None = None,
    app_port: int = 3000,
    app_start_command: str = "npm run start",
    app_build_command: str | None = "npm install && npm run build",
    # CUA server configuration
    cua_server_port: int = 3001,
    # Viewport configuration
    viewport_width: int = 1024,
    viewport_height: int = 768,
    # Screenshot configuration
    save_screenshots: bool = False,
    keep_recent_screenshots: int | None = 2,
    # Sandbox configuration
    cpu_cores: int = 2,
    memory_gb: int = 4,
    use_prebuilt_image: bool = True,
    prebuilt_image: str = "team-cmlr3u2er002zhr01tj8f48ts/localbrowserapp:v1.0.1",
    **kwargs,
) -> vf.Environment:
    """
    Load a local browser example environment.

    This environment runs a Next.js application (either the example app or
    a custom one) alongside a CUA server in a sandbox. Agents can interact
    with the app using vision-based browser primitives.

    Args:
        max_turns: Maximum conversation turns (default: 15)
        judge_model: Model for judging task completion
        system_prompt: System prompt for the agent
        app_path: Path to your Next.js app (default: uses example app)
        app_port: Port for the application (default: 3000)
        app_start_command: Command to start the app (default: "npm run start")
        app_build_command: Command to build the app (default: "npm install && npm run build")
        cua_server_port: Port for CUA server (default: 3001)
        viewport_width: Browser viewport width (default: 1024)
        viewport_height: Browser viewport height (default: 768)
        save_screenshots: Save screenshots to disk (default: False)
        keep_recent_screenshots: Number of screenshots to keep in context (default: 2)
        cpu_cores: CPU cores for sandbox (default: 2)
        memory_gb: Memory in GB for sandbox (default: 4)
        use_prebuilt_image: Use pre-built Docker image (default: False)
        prebuilt_image: Pre-built image name (default: deepdream19/cua-local-server:latest)
        **kwargs: Additional arguments passed to LocalBrowserEnv

    Returns:
        Configured LocalBrowserEnv instance

    Example:
        # Using the example Next.js app
        >>> env = load_environment()

        # Using a custom Next.js app
        >>> env = load_environment(app_path="/path/to/my/nextjs/app")

        # With custom build command
        >>> env = load_environment(
        ...     app_path="/path/to/app",
        ...     app_build_command="yarn && yarn build"
        ... )
    """
    # Create inline dataset
    dataset = create_example_dataset()

    # Create judge rubric for evaluation
    rubric = vf.JudgeRubric(
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
    )
    rubric.add_reward_func(judge_answer, weight=1.0)

    return LocalBrowserEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        system_prompt=system_prompt,
        app_path=app_path,
        app_port=app_port,
        app_start_command=app_start_command,
        app_build_command=app_build_command,
        cua_server_port=cua_server_port,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        save_screenshots=save_screenshots,
        keep_recent_screenshots=keep_recent_screenshots,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        use_prebuilt_image=use_prebuilt_image,
        prebuilt_image=prebuilt_image,
        **kwargs,
    )
