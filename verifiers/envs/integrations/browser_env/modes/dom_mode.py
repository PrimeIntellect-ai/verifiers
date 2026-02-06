"""DOM-based browser mode using Stagehand SDK."""

import asyncio
import json
import os
from typing import Any
from dotenv import load_dotenv
import verifiers as vf
from stagehand import AsyncStagehand
from stagehand.session import AsyncSession


load_dotenv()


class DOMMode:
    """
    DOM-based browser mode using Stagehand SDK.
    Provides natural language tools: navigate, observe, act, extract
    """

    def __init__(
        self,
        browserbase_api_key: str | None = None,
        project_id: str | None = None,
        model_api_key: str | None = None,
        stagehand_model: str = "openai/gpt-4o-mini",
        proxy_model_to_stagehand: bool = False,
        proxies: bool = False,
        advanced_stealth: bool = False,
    ):
        self.api_key = browserbase_api_key or os.getenv("BROWSERBASE_API_KEY")
        self.project_id = project_id or os.getenv("BROWSERBASE_PROJECT_ID")
        self.model_api_key = model_api_key or os.getenv("MODEL_API_KEY")
        self.stagehand_model = stagehand_model
        self.proxy_model_to_stagehand = proxy_model_to_stagehand
        self.proxies = proxies
        self.advanced_stealth = advanced_stealth
        self.stagehand_client: AsyncStagehand | None = None
        self.logger = None  # Will be set when register_tools is called
        self._client_lock = asyncio.Lock()

    def register_tools(self, env) -> None:
        """Register DOM mode tools with the environment."""
        self.logger = env.logger
        env.add_tool(self.navigate, args_to_skip=["session"])
        env.add_tool(self.observe, args_to_skip=["session", "llm_config"])
        env.add_tool(self.act, args_to_skip=["session", "llm_config"])
        env.add_tool(self.extract, args_to_skip=["session", "llm_config"])

    def _get_api_key(self, state: vf.State) -> str | None:
        """Get API key for Stagehand operations.

        If proxy_model_to_stagehand is False, use the configured model_api_key
        (for Stagehand's default model). If True, use the verifiers client's key.
        """
        if not self.proxy_model_to_stagehand:
            # Use configured key for Stagehand's default model (e.g., OpenAI)
            return self.model_api_key

        # Proxy mode: use verifiers client's API key
        client = state.get("client")
        if client and hasattr(client, "api_key") and client.api_key:
            return client.api_key
        return self.model_api_key

    async def _create_session(self, state: vf.State) -> AsyncSession:
        """Create a new Stagehand session."""
        api_key = self._get_api_key(state)
        if not api_key:
            raise ValueError(
                "No API key available. Set MODEL_API_KEY env var or ensure "
                "verifiers client has an api_key."
            )

        async with self._client_lock:
            if self.stagehand_client is None:
                self.stagehand_client = AsyncStagehand(
                    browserbase_api_key=self.api_key,
                    browserbase_project_id=self.project_id,
                    model_api_key=api_key,
                )

        # Build browserbase session params
        browserbase_params = {}
        if self.proxies:
            browserbase_params["proxies"] = self.proxies
        if self.advanced_stealth:
            browserbase_params["browserSettings"] = {
                "advancedStealth": self.advanced_stealth
            }
        browserbase_params = browserbase_params or None

        session = await self.stagehand_client.sessions.create(
            model_name=self.stagehand_model,
            browserbase_session_create_params=browserbase_params,
        )
        return session

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        """Create per-rollout Stagehand session."""
        session = await self._create_session(state)
        state["stagehand_session"] = session
        state["stagehand_session_id"] = session.id
        return state

    def _get_llm_config(self, state: vf.State) -> dict[str, Any] | None:
        """Extract model configuration from verifiers state to route LLM calls."""
        client = state.get("client")
        model = state.get("model")

        if client is None or model is None:
            return None

        llm_config: dict[str, Any] = {"modelName": model}

        if hasattr(client, "base_url") and client.base_url:
            base_url = str(client.base_url)
            if base_url and not base_url.startswith("https://api.openai.com"):
                llm_config["baseURL"] = base_url

        if hasattr(client, "api_key") and client.api_key:
            llm_config["apiKey"] = client.api_key

        return llm_config

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inject session and model config into Stagehand tool calls."""
        updated_args = dict(tool_args)
        stagehand_tools = {"navigate", "observe", "act", "extract"}
        if tool_name in stagehand_tools:
            updated_args["session"] = state["stagehand_session"]

        llm_tools = {"observe", "act", "extract"}
        if tool_name in llm_tools and self.proxy_model_to_stagehand:
            llm_config = self._get_llm_config(state)
            updated_args["llm_config"] = llm_config

        return updated_args

    async def cleanup_session(self, state: vf.State) -> None:
        """Clean up Stagehand session after rollout."""
        session = state.get("stagehand_session")
        if session is not None:
            try:
                await session.end()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error ending session: {e}")

        state.pop("stagehand_session", None)
        state.pop("stagehand_session_id", None)

    async def teardown(self) -> None:
        """Clean up Stagehand client on environment teardown."""
        if self.stagehand_client is not None:
            try:
                await self.stagehand_client.close()
            except Exception:
                pass
            self.stagehand_client = None

    def filter_screenshots_in_messages(self, messages: list) -> list:
        """DOM mode doesn't use screenshots, return messages unchanged."""
        return messages

    # ==================== Tool Methods ====================

    async def navigate(self, url: str, session: Any) -> str:
        """Tool to navigate the browser session directly to a url.

        Args:
            url: The url to navigate to.
        """
        try:
            await session.navigate(url=url)
            return f"Navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {str(e)}"

    async def observe(
        self, instruction: str, session: Any, llm_config: Any = None
    ) -> str:
        """Tool to find possible actions on the page matching the instruction.

        Args:
            instruction: The instruction to find possible actions for.
        """
        try:
            if llm_config:
                response = await session.observe(
                    instruction=instruction, options={"model": llm_config}
                )
            else:
                response = await session.observe(instruction=instruction)
            actions = [
                {
                    "description": a.description,
                    "selector": a.selector,
                    "method": a.method,
                }
                for a in response.data.result
            ]
            if not actions:
                return "No matching elements found"
            return json.dumps(actions, indent=2)
        except Exception as e:
            return f"Error observing page: {str(e)}"

    async def act(self, instruction: str, session: Any, llm_config: Any = None) -> str:
        """Tool to request an action be performed on the current page.
        These can be any natural language, as well as be descriptions found by previous observe() calls.
        Vauge instructions will not get you good results so it is recommended to be as specific as possible so the agent performing the action knows exactly what to do.

        Examples:
            Things like 'click the article about yesterdays news' or 'open the contact us page'

        Args:
            instruction: The instruction to perform an action for.
        """
        try:
            if llm_config:
                response = await session.act(
                    input=instruction, options={"model": llm_config}
                )
            else:
                response = await session.act(input=instruction)
            result = response.data.result
            status = "Success" if result.success else "Failed"
            return f"{status}: {result.message}"
        except Exception as e:
            return f"Error executing action: {str(e)}"

    async def extract(
        self,
        instruction: str,
        schema_json: str,
        session: Any,
        llm_config: Any = None,
    ) -> str:
        """Tool to extract structured data from the current page.

        Args:
            instruction: The instruction to extract data for.
            schema_json: The schema to use for extraction.
        """
        try:
            schema = json.loads(schema_json)
            if llm_config:
                response = await session.extract(
                    instruction=instruction,
                    schema=schema,
                    options={"model": llm_config},
                )
            else:
                response = await session.extract(
                    instruction=instruction,
                    schema=schema,
                )
            return json.dumps(response.data.result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing schema JSON: {str(e)}"
        except Exception as e:
            return f"Error extracting data: {str(e)}"
