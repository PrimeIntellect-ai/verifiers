"""
Full Browse Environment.

Browser automation environment with a toolset aligned to the Wide Browse
subagent traces. Uses the same CUA server and sandbox infrastructure as
the local browser example, but exposes a richer set of tools:

- computer: unified tool with action batching (click, type, key, scroll, wait, etc.)
- get_page_text: extract full page text
- read_page: accessibility tree with element refs and coordinates
- find: search for elements by natural-language query
- form_input: fill form fields by ref
- tabs_context: get current tab state

Usage:
    # With the built-in example app
    prime eval run full-browse -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY

    # With a custom app
    prime eval run full-browse -m openai/gpt-4.1-mini -a '{"app_path": "/path/to/app"}'
"""

from pathlib import Path

import verifiers as vf
from verifiers.envs.integrations.browser_env.modes.full_browse_mode import (
    FullBrowseMode,
)
from datasets import Dataset


FULL_BROWSE_SYSTEM_PROMPT = """You are a browser automation agent that can interact with web pages using a rich set of tools.

## Tools

### computer
Execute low-level browser actions. Pass a list of actions to execute sequentially.
Each action is a dict with an "action" key. Supported actions:
- left_click: {"action": "left_click", "coordinate": [x, y]}
- right_click: {"action": "right_click", "coordinate": [x, y]}
- double_click: {"action": "double_click", "coordinate": [x, y]}
- triple_click: {"action": "triple_click", "coordinate": [x, y]}
- type: {"action": "type", "text": "hello"}
- key: {"action": "key", "key": "Enter"}
- scroll: {"action": "scroll", "coordinate": [x, y], "direction": "up"|"down"}
- wait: {"action": "wait", "duration": 2}
- screenshot: {"action": "screenshot"}

Multiple actions can be chained in one call. A screenshot is always returned after the last action.

### get_page_text
Extract the full text content of the current page. Useful for reading without relying on screenshots.

### read_page
Get the page's element tree with refs and coordinates. Use filter="interactive" to see only buttons, links, inputs etc. Each element gets a ref (e.g. ref_42) usable with form_input. On large pages, use depth=2 or depth=3 to limit output. Use ref_id="ref_42" to focus on a subtree.

### find
Search for elements matching a query. Returns refs and coordinates for matching elements.

### form_input
Set a form field value using its ref from read_page or find: form_input(ref="ref_42", value="hello")

### tabs_context
Get current browser tab state.

## Strategy
1. Start by observing the page (screenshot or read_page).
2. Use read_page with filter="interactive" to find clickable elements when screenshots are unclear. On large pages, add depth=2 to limit output.
3. Use computer for click/type/scroll actions, combining related actions in one call.
4. Use get_page_text to extract information from text-heavy pages.
5. Use find to locate specific elements by description.
6. Use form_input for filling forms precisely (avoids click-then-type timing issues).

After each action you will receive updated page state. Analyze it to determine your next action.
Complete the given task by interacting with the application."""


def create_example_dataset() -> Dataset:
    """Create a simple dataset for testing the full browse environment."""
    return Dataset.from_dict(
        {
            "question": [
                "Click the 'Increment' button three times and tell me what the counter shows.",
                "Type 'Hello World' into the text input field and click Submit. What does the submitted value show?",
                "Use read_page to find the Selection Test section, then click 'Option B'. What option is now selected?",
            ],
            "answer": [
                "3",
                "Hello World",
                "Option B",
            ],
            "task_id": [
                "fb-counter-test",
                "fb-input-test",
                "fb-selection-test",
            ],
        }
    )


JUDGE_PROMPT = """You are evaluating a browser automation agent's interaction with a web application.

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


class FullBrowseEnv(vf.StatefulToolEnv):
    """
    Environment with Full Browse toolset for browser automation.

    Uses FullBrowseMode to provide a unified ``computer`` tool with action
    batching plus higher-level inspection tools (get_page_text, read_page,
    find, form_input, tabs_context).
    """

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        max_turns: int = 25,
        system_prompt: str = FULL_BROWSE_SYSTEM_PROMPT,
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

        self._full_browse_mode = FullBrowseMode(
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
        self._full_browse_mode.register_tools(self)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Initialize the browser sandbox."""
        state = await self._full_browse_mode.setup_state(state, **kwargs)
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
        return self._full_browse_mode.update_tool_args(
            tool_name, tool_args, messages, state, **kwargs
        )

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        """Get prompt messages, filtering old screenshots."""
        messages = await super().get_prompt_messages(state)
        return self._full_browse_mode.filter_screenshots_in_messages(list(messages))

    @vf.cleanup
    async def cleanup_session(self, state: vf.State) -> None:
        """Clean up the browser session and sandbox."""
        await self._full_browse_mode.cleanup_session(state)

    @vf.teardown
    async def teardown(self) -> None:
        """Clean up all resources."""
        if hasattr(self, "_full_browse_mode") and self._full_browse_mode is not None:
            await self._full_browse_mode.teardown()


def load_environment(
    max_turns: int = 25,
    judge_model: str = "gpt-4o-mini",
    system_prompt: str = FULL_BROWSE_SYSTEM_PROMPT,
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
    Load a Full Browse environment.

    This environment provides a toolset aligned to the Wide Browse subagent
    traces for browser automation. It uses the same CUA server and sandbox
    infrastructure as the local browser example, but exposes richer tools.

    Args:
        max_turns: Maximum conversation turns (default: 25)
        judge_model: Model for judging task completion
        system_prompt: System prompt for the agent
        app_path: Path to your app (default: built-in example app)
        app_port: Port for the application (default: 3000)
        app_start_command: Command to start the app
        app_build_command: Command to build the app
        cua_server_port: Port for CUA server (default: 3001)
        viewport_width: Browser viewport width (default: 1024)
        viewport_height: Browser viewport height (default: 768)
        save_screenshots: Save screenshots to disk
        keep_recent_screenshots: Number of screenshots to keep in context
        cpu_cores: CPU cores for sandbox
        memory_gb: Memory in GB for sandbox
        use_prebuilt_image: Use pre-built Docker image
        prebuilt_image: Pre-built image name
        **kwargs: Additional arguments

    Returns:
        Configured FullBrowseEnv instance
    """
    dataset = create_example_dataset()

    rubric = vf.JudgeRubric(
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
    )
    rubric.add_reward_func(judge_answer, weight=1.0)

    return FullBrowseEnv(
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
