# ABOUTME: vLLM-based inference policy for high-throughput serving
# ABOUTME: Wraps VLLMClient for production deployment scenarios
from typing import TYPE_CHECKING

from verifiers.inference.policy import InferencePolicy
from verifiers.inference.vllm_client import VLLMClient
from verifiers.types import Messages, ModelResponse, SamplingArgs

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


class VLLMPolicy(InferencePolicy):
    """
    Inference policy using vLLM for high-throughput serving.

    Optimized for production deployment with:
    - Continuous batching for efficient GPU utilization
    - PagedAttention for memory efficiency
    - Optional weight syncing for online learning scenarios
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        model: str | None = None,
        connection_timeout: float = 0.0,
    ):
        """
        Initialize vLLM policy.

        Args:
            host: vLLM server host
            port: vLLM server port
            model: Model name (for reference tracking)
            connection_timeout: Timeout for initial server connection
        """
        self.client = VLLMClient(
            host=host,
            port=port,
            connection_timeout=connection_timeout,
        )
        self.model = model or "vllm-model"
        self._supports_weight_sync = False

    async def generate(
        self,
        prompt: Messages,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate response using vLLM server."""
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

    def enable_weight_sync(self, group_port: int = 51216) -> None:
        """
        Enable weight synchronization for online learning.

        Args:
            group_port: Port for weight update communication

        Note:
            This is only needed for training scenarios where model
            weights are updated during inference.
        """
        self.client.group_port = group_port
        self.client.init_communicator()
        self._supports_weight_sync = True

    def sync_weights(self, model: "PreTrainedModel") -> None:
        """
        Synchronize model weights to vLLM server.

        Args:
            model: Source model with updated weights

        Raises:
            RuntimeError: If weight sync not enabled
        """
        if not self._supports_weight_sync:
            raise RuntimeError(
                "Weight sync not enabled. Call enable_weight_sync() first."
            )

        # Update all parameters
        for name, param in model.named_parameters():
            self.client.update_named_param(name, param.data)

        # Reset cache after weight update
        self.client.reset_prefix_cache()

    @classmethod
    def from_client(cls, client: VLLMClient, model: str | None = None) -> "VLLMPolicy":
        """
        Create policy from existing VLLMClient.

        Args:
            client: Configured VLLMClient instance
            model: Model name override

        Returns:
            VLLMPolicy wrapping the client
        """
        policy = cls.__new__(cls)
        policy.client = client
        policy.model = model or "vllm-model"
        policy._supports_weight_sync = False
        return policy
