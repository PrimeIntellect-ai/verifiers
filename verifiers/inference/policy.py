# ABOUTME: Base abstraction for model inference policies
# ABOUTME: Decouples inference from training infrastructure for deployment flexibility
from abc import ABC, abstractmethod

from openai import AsyncOpenAI, OpenAI

from verifiers.types import Messages, ModelResponse, SamplingArgs


class InferencePolicy(ABC):
    """
    Abstract base class for model inference policies.

    Provides a unified interface for generating responses across different
    backends (API, local models, vLLM servers). Designed for deployment
    scenarios where training infrastructure is not required.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: Messages,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Generate a model response for the given prompt.

        Args:
            prompt: Input messages (chat format) or string (completion format)
            sampling_args: Sampling parameters (temperature, top_p, etc.)
            **kwargs: Additional backend-specific parameters

        Returns:
            ModelResponse (ChatCompletion or Completion object)
        """
        pass

    @classmethod
    def from_client(
        cls, client: AsyncOpenAI | OpenAI, model: str
    ) -> "APIPolicy":
        """
        Create a policy from an existing OpenAI-compatible client.

        Args:
            client: AsyncOpenAI or OpenAI client instance
            model: Model name/identifier

        Returns:
            APIPolicy wrapping the client
        """
        # Import here to avoid circular dependency at module level
        # APIPolicy is defined below in this same file
        return APIPolicy(client=client, model=model)


class APIPolicy(InferencePolicy):
    """
    Inference policy for OpenAI-compatible API endpoints.

    Wraps AsyncOpenAI clients for use with any OpenAI-compatible API
    (OpenAI, Anthropic, vLLM server, etc.).
    """

    def __init__(self, client: AsyncOpenAI | OpenAI, model: str):
        """
        Initialize API policy.

        Args:
            client: AsyncOpenAI or OpenAI client
            model: Model name for generation requests
        """
        self.client = (
            client if isinstance(client, AsyncOpenAI) else AsyncOpenAI(
                api_key=client.api_key, base_url=client.base_url
            )
        )
        self.model = model

    async def generate(
        self,
        prompt: Messages,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate response using the wrapped API client."""
        sampling_args = sampling_args or {}

        # Determine message type
        is_chat = isinstance(prompt, list)

        if is_chat:
            # Chat completions format
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=prompt,  # type: ignore
                **sampling_args,
            )
        else:
            # Completions format
            response = await self.client.completions.create(
                model=self.model,
                prompt=prompt,  # type: ignore
                **sampling_args,
            )

        return response
