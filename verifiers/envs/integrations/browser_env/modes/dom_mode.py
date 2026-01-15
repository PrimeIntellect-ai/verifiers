"""DOM-based browser mode using Stagehand SDK."""

import os
import json
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
        project_id: str | None = None,
        model_api_key: str | None = None,
        stagehand_model: str = "openai/gpt-4o-mini",
        proxy_model_to_stagehand: bool = False,
    ):
        self.project_id = project_id or os.getenv("BROWSERBASE_PROJECT_ID")
        self.model_api_key = model_api_key or os.getenv("MODEL_API_KEY")
        self.stagehand_model = stagehand_model
        self.proxy_model_to_stagehand = proxy_model_to_stagehand
        self.stagehand_client: AsyncStagehand | None = None
        self.logger = None  # Will be set when register_tools is called

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

        if self.stagehand_client is None:
            self.stagehand_client = AsyncStagehand(
                browserbase_api_key=os.getenv("BROWSERBASE_API_KEY"),
                browserbase_project_id=self.project_id,
                model_api_key=api_key,
            )

        session = await self.stagehand_client.sessions.create(
            model_name=self.stagehand_model,
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

    # ==================== Tool Methods ====================

    async def navigate(self, url: str, session: Any) -> str:
        """Navigate to a URL."""
        try:
            await session.navigate(url=url)
            return f"Navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {str(e)}"

    async def observe(
        self, instruction: str, session: Any, llm_config: Any = None
    ) -> str:
        """Find possible actions on the page matching the instruction."""
        try:
            options = {}
            if llm_config:
                options["model"] = llm_config

            response = await session.observe(
                instruction=instruction, options=options if options else None
            )
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
        """Execute an action described in natural language."""
        try:
            options = {}
            if llm_config:
                options["model"] = llm_config

            response = await session.act(
                input=instruction, options=options if options else None
            )
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
        """Extract structured data from the page."""
        try:
            schema = json.loads(schema_json)
            options = {}
            if llm_config:
                options["model"] = llm_config

            response = await session.extract(
                instruction=instruction,
                schema=schema,
                options=options if options else None,
            )
            return json.dumps(response.data.result, indent=2)
        except json.JSONDecodeError as e:
            return f"Error parsing schema JSON: {str(e)}"
        except Exception as e:
            return f"Error extracting data: {str(e)}"
