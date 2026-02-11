"""
Browser CUA Mode Example Environment.

This example demonstrates using BrowserEnv with CUA (Computer Use Agent) mode
for vision-based browser control.

CUA mode uses screenshots and vision models to interact with the browser,
providing low-level primitives like click, scroll, and type_text.

By default, CUA mode uses a pre-built Docker image (deepdream19/cua-server:latest)
for fastest startup (~5-10s). No manual server setup is required.

Usage:
    # Default (pre-built image - fastest, recommended)
    prime eval run browser-cua-example -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY

    # Binary upload mode (for custom server versions)
    prime eval run browser-cua-example -m openai/gpt-4.1-mini -a '{"use_prebuilt_image": false}'

    # Manual mode (for local development)
    cd assets/templates/browserbase/cua && ./start.sh
    prime eval run browser-cua-example -m openai/gpt-4.1-mini -a '{"use_sandbox": false}'
"""

from typing import Literal

import verifiers as vf
from verifiers.envs.integrations.browser_env import BrowserEnv
from datasets import Dataset

CUA_SYSTEM_PROMPT = """You are a browser automation agent. You can control a web browser using the provided tools.

Available tools:
- click(x, y, button): Click at coordinates
- double_click(x, y): Double-click at coordinates
- type_text(text): Type text into focused element
- keypress(keys): Press keyboard keys
- scroll(x, y, scroll_x, scroll_y): Scroll at position
- goto(url): Navigate to URL
- back(): Go back in history
- forward(): Go forward in history
- wait(time_ms): Wait for specified milliseconds
- screenshot(): Capture current page state

After each action, you will receive a screenshot showing the current page state.
Analyze the screenshot to determine your next action.

Complete the given task efficiently using the minimum number of actions necessary."""


def create_example_dataset() -> Dataset:
    """
    Create a simple inline dataset for the CUA mode hello world example.

    This dataset tests basic browser navigation and content extraction
    using vision-based interactions.
    """
    return Dataset.from_dict(
        {
            "question": [
                "What does the headline say on the primeintellect.ai homepage?"
            ],
            "answer": ["The Open Superintelligence Stack"],
            "start_url": ["https://primeintellect.ai"],
            "task_id": ["cua-example-0"],
        }
    )


# Judge prompt for evaluating answer correctness
JUDGE_PROMPT = """You are evaluating a browser automation agent's answer to a question.

Question:
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

Does the agent's response contain the correct answer? The answer may be embedded in a longer response or phrased differently, but should convey the same information as the expected answer.

Respond "yes" if the agent's response contains the correct answer, "no" if it does not."""


async def judge_answer(
    judge,
    prompt: str | list,
    completion: str | list,
    answer: str,
    state: vf.State,
) -> float:
    """
    LLM judge reward that compares the agent's final answer to the reference answer.

    Args:
        judge: Callable injected by JudgeRubric that calls the judge LLM
        prompt: The original prompt/question given to the agent
        completion: The agent's full response/trajectory
        answer: The expected/reference answer from the dataset
        state: The current environment state

    Returns:
        float: 1.0 if the judge determines the answer is correct, 0.0 otherwise
    """
    judge_response = await judge(prompt, completion, answer, state)
    is_correct = "yes" in judge_response.lower()
    return 1.0 if is_correct else 0.0


def load_environment(
    max_turns: int = 15,
    judge_model: str = "gpt-4o-mini",
    system_prompt: str = CUA_SYSTEM_PROMPT,
    # CUA mode configuration
    use_sandbox: bool = True,
    server_url: str = "http://localhost:3000",
    # Shared configuration
    browserbase_api_key: str | None = None,
    browserbase_project_id: str | None = None,
    env: Literal["LOCAL", "BROWSERBASE"] = "BROWSERBASE",
    viewport_width: int = 1024,
    viewport_height: int = 768,
    save_screenshots: bool = False,
    keep_recent_screenshots: int | None = 2,
    proxies: bool = False,
    # Sandbox configuration (when use_sandbox=True)
    docker_image: str = "node:18-slim",
    cpu_cores: int = 2,
    memory_gb: int = 4,
    use_binary: bool = True,
    # Pre-built image configuration (default - fastest startup)
    use_prebuilt_image: bool = True,
    prebuilt_image: str = "deepdream19/cua-server:latest",
    **kwargs,
) -> vf.Environment:
    """
    Load a CUA mode browser example environment.

    This is a self-contained "hello world" example demonstrating how to use
    BrowserEnv with CUA mode for vision-based browser control.

    Execution modes (from fastest to most flexible):
    1. Pre-built image (default): Uses deepdream19/cua-server:latest
       Fastest startup (~5-10s). No binary upload needed.
    2. Binary upload (use_prebuilt_image=False): Builds/uploads binary
       Slower startup (~30-60s). Use for custom server versions.
    3. Manual server (use_sandbox=False): Connect to local server
       For development: cd cua-server && ./start.sh

    Available tools in CUA mode:
    - click(x, y, button): Click at coordinates
    - double_click(x, y): Double-click at coordinates
    - type_text(text): Type text into focused element
    - keypress(keys): Press keyboard keys
    - scroll(x, y, scroll_x, scroll_y): Scroll at position
    - goto(url): Navigate to URL
    - back(): Go back in history
    - forward(): Go forward in history
    - wait(time_ms): Wait for specified milliseconds
    - screenshot(): Capture current page state

    Args:
        max_turns: Maximum conversation turns (default: 15)
        judge_model: Model for judging task completion
        use_sandbox: Auto-deploy CUA server to sandbox (default: True)
        server_url: CUA server URL for manual mode (default: http://localhost:3000)
        browserbase_api_key: Browserbase API key (or set BROWSERBASE_API_KEY env var)
        browserbase_project_id: Browserbase project ID (or set BROWSERBASE_PROJECT_ID env var)
        env: Browser environment - "LOCAL" or "BROWSERBASE" (default: BROWSERBASE)
        viewport_width: Browser viewport width (default: 1024)
        viewport_height: Browser viewport height (default: 768)
        save_screenshots: Save screenshots during execution (default: False)
        keep_recent_screenshots: Number of recent screenshots to keep in context
        proxies: Enable Browserbase proxies
        docker_image: Docker image for sandbox when use_prebuilt_image=False (default: node:18-slim)
        cpu_cores: CPU cores for sandbox (default: 2)
        memory_gb: Memory in GB for sandbox (default: 4)
        use_binary: Use pre-built SEA binary when use_prebuilt_image=False (default: True)
        use_prebuilt_image: Use pre-built Docker image for fastest startup (default: True)
        prebuilt_image: Docker image to use (default: deepdream19/cua-server:latest)
        **kwargs: Additional arguments passed to BrowserEnv

    Returns:
        Configured BrowserEnv instance in CUA mode

    Example:
        # Pre-built image (default - fastest, recommended)
        >>> env = load_environment()

        # Binary upload mode (for custom server)
        >>> env = load_environment(use_prebuilt_image=False)

        # Manual mode (for local development)
        >>> env = load_environment(use_sandbox=False, server_url="http://localhost:3000")
    """
    # Create inline dataset
    dataset = create_example_dataset()

    # Create judge rubric for evaluation
    rubric = vf.JudgeRubric(
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
    )
    rubric.add_reward_func(judge_answer, weight=1.0)

    # Create BrowserEnv with CUA mode
    return BrowserEnv(
        mode="cua",
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        system_prompt=system_prompt,
        # CUA mode configuration
        use_sandbox=use_sandbox,
        server_url=server_url,
        env=env,
        browserbase_api_key=browserbase_api_key,
        browserbase_project_id=browserbase_project_id,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        save_screenshots=save_screenshots,
        keep_recent_screenshots=keep_recent_screenshots,
        proxies=proxies,
        # Sandbox configuration
        docker_image=docker_image,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        use_binary=use_binary,
        # Pre-built image configuration
        use_prebuilt_image=use_prebuilt_image,
        prebuilt_image=prebuilt_image,
        **kwargs,
    )
