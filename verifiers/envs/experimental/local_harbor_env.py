import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from datasets import Dataset
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.trial.trial import Trial
from prime_tunnel import Tunnel

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    ChatCompletionToolParam,
    Messages,
    MessageType,
    ModelResponse,
    RolloutInput,
    SamplingArgs,
    State,
)
from verifiers.utils.interception_utils import (
    InterceptionServer,
    create_empty_completion,
    deliver_response,
    get_streaming_model_response,
)

logger = logging.getLogger(__name__)


class LocalHarborEnv(vf.Environment):
    """
    This is an alternative to the CliAgentEnv + HarborEnv pattern.
    Instead of creating a Prime sandbox, installing Harbor CLI, and running agents
    through a custom script, this environment uses Harbor SDK directly to run agents
    and intercepts API calls via a tunnel to collect trajectory.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_name: str = "terminus-2",
        model_name: str = "anthropic/claude-sonnet-4",
        environment_type: str = "prime",
        interception_port: int = 8765,
        timeout_seconds: float = 3600.0,
        timeout_multiplier: float = 1.0,
        delete_sandbox: bool = True,
        disable_verifier: bool = False,
        trials_dir: Path | None = None,
        max_turns: int | None = None,
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path)
        self.task_names = tasks
        self.agent_name = agent_name
        self.model_name = model_name
        self.environment_type = environment_type
        self.interception_port = interception_port
        self.timeout_seconds = timeout_seconds
        self.timeout_multiplier = timeout_multiplier
        self.delete_sandbox = delete_sandbox
        self.disable_verifier = disable_verifier
        self.trials_dir = trials_dir or Path("trials")
        self.max_turns = max_turns

        # Tunnel and interception server
        self._tunnel: Tunnel | None = None
        self._tunnel_lock = asyncio.Lock()
        self._interception_server = InterceptionServer(port=interception_port)

        # Load dataset
        dataset = self._load_harbor_dataset()
        rubric = Rubric(funcs=[self._harbor_reward], weights=[1.0])

        super().__init__(
            dataset=dataset,
            rubric=rubric,
            message_type="chat",
            **kwargs,
        )

    async def get_tunnel_url(self) -> str:
        """Get tunnel URL, starting the tunnel if needed."""
        async with self._tunnel_lock:
            if self._tunnel is None:
                if logger.isEnabledFor(logging.DEBUG):
                    self._tunnel = Tunnel(
                        local_port=self.interception_port,
                        log_level="debug",
                    )
                else:
                    self._tunnel = Tunnel(local_port=self.interception_port)
                url = await self._tunnel.start()
                logger.debug(f"Prime Tunnel started: {url}")
                return url
            else:
                assert self._tunnel.url is not None, "Tunnel started but URL is None"
                return self._tunnel.url

    def _load_harbor_dataset(self) -> Dataset:
        """Load Harbor tasks from dataset directory into a Dataset with prompts."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        tasks = []
        for task_dir in sorted(self.dataset_path.iterdir()):
            if not task_dir.is_dir():
                continue

            if self.task_names and task_dir.name not in self.task_names:
                continue

            task_toml = task_dir / "task.toml"
            instruction_md = task_dir / "instruction.md"

            if not task_toml.exists() or not instruction_md.exists():
                logger.warning(
                    f"Skipping {task_dir.name}: missing task.toml or instruction.md"
                )
                continue

            with open(task_toml, "rb") as f:
                config = tomllib.load(f)

            instruction = instruction_md.read_text().strip()
            messages = [{"role": "user", "content": instruction}]

            task_entry = {
                "example_id": len(tasks),
                "task": task_dir.name,
                "prompt": messages,
                "info": {
                    "task_dir": str(task_dir),
                    "docker_image": config.get("environment", {}).get("docker_image"),
                    "config": config,
                },
            }

            tasks.append(task_entry)

        if not tasks:
            raise ValueError(f"No valid Harbor tasks found in {self.dataset_path}")

        logger.info(f"Loaded {len(tasks)} Harbor tasks from {self.dataset_path}")
        return Dataset.from_list(tasks)

    async def _harbor_reward(self, state: State, **kwargs) -> float:
        """Extract reward from state (set during rollout)."""
        return state.get("reward", 0.0)

    async def setup_state(self, state: State) -> State:
        """Setup state for a rollout: start tunnel and register for interception."""
        rollout_id = f"rollout_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        # Start interception server and tunnel
        await self._interception_server.start()
        tunnel_url = await self.get_tunnel_url()
        state["interception_base_url"] = f"{tunnel_url}/rollout/{rollout_id}/v1"

        # Register rollout for interception
        request_queue = self._interception_server.register_rollout(rollout_id)
        state["request_id_queue"] = request_queue
        state["agent_completed"] = False

        return state

    def _build_trial_config(self, state: State) -> TrialConfig:
        """Build Harbor TrialConfig with api_base pointing to our tunnel."""
        task_info = state.get("info") or {}
        task_dir = task_info.get("task_dir")
        if not task_dir:
            raise ValueError("task_dir not found in state info")

        # Build agent kwargs with api_base for interception
        agent_kwargs: dict[str, Any] = {
            "api_base": state["interception_base_url"],
        }
        if self.max_turns is not None:
            agent_kwargs["max_turns"] = self.max_turns

        # Map environment type string to enum
        env_type = EnvironmentType(self.environment_type)

        return TrialConfig(
            task=TaskConfig(path=Path(task_dir)),
            trials_dir=self.trials_dir,
            timeout_multiplier=self.timeout_multiplier,
            agent=AgentConfig(
                name=self.agent_name,
                model_name=self.model_name,
                kwargs=agent_kwargs,
            ),
            environment=EnvironmentConfig(
                type=env_type,
                delete=self.delete_sandbox,
            ),
            verifier=VerifierConfig(
                disable=self.disable_verifier,
            ),
        )

    async def get_prompt_messages(self, state: State) -> Messages:
        """Wait for agent to make an API request or complete."""
        request_id_queue = state["request_id_queue"]
        poll_interval = 2.0

        while True:
            try:
                request_id = await asyncio.wait_for(
                    request_id_queue.get(),
                    timeout=poll_interval,
                )
                state["current_request_id"] = request_id
                intercept = self._interception_server.intercepts[request_id]
                return intercept["messages"]

            except asyncio.TimeoutError:
                if state.get("agent_completed", False):
                    return []
                if time.time() - state["timing"]["start_time"] > self.timeout_seconds:
                    return []

    async def get_model_response(
        self,
        state: State,
        prompt: Messages,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        oai_tools: list[ChatCompletionToolParam] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType | None = None,
    ) -> ModelResponse:
        """Get model response and unblock the waiting HTTP handler."""
        # Handle agent completion case (empty prompt)
        if not prompt:
            return create_empty_completion(model or state["model"])

        request_id = state.get("current_request_id")
        intercept = (
            self._interception_server.intercepts.get(request_id) if request_id else None
        )

        if intercept:
            model = state.get("model") or model
            oai_tools = intercept.get("tools") or oai_tools

        response: ModelResponse | None = None
        error: BaseException | None = None

        try:
            if intercept and intercept.get("stream"):
                response = await get_streaming_model_response(
                    state=state,
                    prompt=prompt,
                    intercept=intercept,
                    client=client,
                    model=model,
                    oai_tools=oai_tools,
                    sampling_args=sampling_args,
                )
            else:
                response = await super().get_model_response(
                    state=state,
                    prompt=prompt,
                    client=client,
                    model=model,
                    oai_tools=oai_tools,
                    sampling_args=sampling_args,
                    message_type=message_type,
                )
        except BaseException as e:
            error = e
            raise
        finally:
            if intercept:
                deliver_response(intercept, response, error)
                state["current_request_id"] = None

        return response

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: ModelResponse,
    ):
        """Add model response to trajectory."""
        if not prompt_messages:
            return
        if len(state["trajectory"]) == 0:
            state["prompt"] = prompt_messages

        # Add to trajectory
        if isinstance(response, ChatCompletion):
            choice = response.choices[0]
            step = {
                "messages": prompt_messages,
                "response": choice.message.model_dump() if choice.message else {},
            }
            state["trajectory"].append(step)

    async def _run_trial(self, state: State):
        """Run Harbor Trial and extract reward from result."""
        trial_config = self._build_trial_config(state)
        trial = Trial(trial_config)

        try:
            result = await trial.run()
            state["trial_result"] = result

            # Extract reward from verifier_result
            if result.verifier_result and result.verifier_result.rewards:
                rewards = result.verifier_result.rewards
                # Use "reward" key if present, otherwise take first value
                if "reward" in rewards:
                    state["reward"] = float(rewards["reward"])
                elif rewards:
                    state["reward"] = float(next(iter(rewards.values())))
                else:
                    state["reward"] = 0.0
            else:
                state["reward"] = 0.0

            # Store exception info if present
            if result.exception_info:
                state["error"] = Exception(
                    f"{result.exception_info.exception_type}: "
                    f"{result.exception_info.exception_message}"
                )

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            state["reward"] = 0.0
            state["error"] = e
        finally:
            state["agent_completed"] = True

    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Run a rollout using Harbor Trial with API interception.

        This method:
        1. Initializes state from input
        2. Starts the Trial in the background
        3. Handles intercepted API requests concurrently
        4. Returns state with trajectory and reward
        """
        state = await self.init_state(input, client, model, sampling_args)
        state = await self.setup_state(state)

        try:
            # Start the trial in the background
            trial_task = asyncio.create_task(self._run_trial(state))

            # Process intercepted requests until trial completes
            while not state.get("agent_completed", False):
                prompt_messages = await self.get_prompt_messages(state)

                if not prompt_messages:
                    # Agent completed or timeout
                    break

                response = await self.get_model_response(
                    state=state,
                    prompt=prompt_messages,
                    client=client,
                    model=model,
                    sampling_args=sampling_args,
                )

                await self.add_model_response(state, prompt_messages, response)

            # Wait for trial to fully complete
            await trial_task

        finally:
            await self._cleanup_rollout(state)

        state["is_completed"] = True
        return state

    async def _cleanup_rollout(self, state: State):
        """Clean up rollout resources."""
        rollout_id = state.get("rollout_id")
        if rollout_id:
            self._interception_server.unregister_rollout(rollout_id)

    @vf.teardown
    async def teardown_resources(self):
        """Stop Prime Tunnel and HTTP interception server."""
        async with self._tunnel_lock:
            if self._tunnel is not None:
                try:
                    await self._tunnel.stop()
                    logger.debug("Prime Tunnel stopped")
                except Exception as e:
                    logger.warning(f"Error stopping Prime Tunnel: {e}")
                finally:
                    self._tunnel = None
        await self._interception_server.stop()

    @vf.stop
    async def agent_completed(self, state: State) -> bool:
        """Check if agent has completed."""
        return state.get("agent_completed", False)

    @vf.stop
    async def timeout_reached(self, state: State) -> bool:
        """Check rollout timeout."""
        elapsed = time.time() - state["timing"]["start_time"]
        return elapsed > self.timeout_seconds

    @vf.stop
    async def has_error(self, state: State) -> bool:
        """Check if there was an error."""
        return state.get("error") is not None
