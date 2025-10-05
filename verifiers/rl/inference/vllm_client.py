import logging
import time

import requests
from openai import AsyncOpenAI
from requests import ConnectionError
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout

logger = logging.getLogger(__name__)


class VLLMClient(AsyncOpenAI):
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions and hot-swap LoRA adapters using vLLM's native APIs.
    Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and load LoRA adapters on the fly:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> client.load_lora_adapter("experiment", "/path/to/adapter")
        ```
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        connection_timeout: float = 0.0,
    ):
        super().__init__(base_url=f"http://{host}:{port}/v1", api_key="local")
        self.session = requests.Session()
        # Configure connection pooling to handle rapid requests better
        adapter = HTTPAdapter(
            pool_connections=10, pool_maxsize=10, max_retries=3, pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.host = host
        self.server_port = port  # Renamed from server_port to port to match super init
        self.server_url = f"http://{self.host}:{self.server_port}"
        self.check_server(connection_timeout)  # check server and fail after timeout

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"{self.server_url}/health"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
                        "seconds. Make sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(
                f"Server is not up yet. Retrying in {retry_interval} seconds..."
            )
            time.sleep(retry_interval)

    def load_lora_adapter(
        self, lora_name: str, adapter_path: str, timeout: float = 300.0
    ) -> None:
        """
        Load a LoRA adapter on the vLLM server using its native hot-swapping API.

        Args:
            lora_name (`str`):
                Identifier used by vLLM to register the adapter.
            adapter_path (`str`):
                Absolute path to the saved LoRA adapter directory.
            timeout (`float`, *optional*, defaults to `300.0`):
                Timeout for the HTTP request.
        """

        url = f"{self.server_url}/v1/load_lora_adapter"
        payload = {"lora_name": lora_name, "lora_path": adapter_path}
        try:
            response = self.session.post(url, json=payload, timeout=timeout)
        except Timeout as exc:
            logger.error(
                "Timeout waiting for vLLM to load adapter '%s' after %.1f seconds",
                lora_name,
                timeout,
            )
            raise Exception(
                f"Request timeout while loading adapter '{lora_name}'"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - log and propagate
            logger.error("Error loading LoRA adapter '%s': %s", lora_name, exc)
            raise

        if response.status_code != 200:
            raise Exception(
                f"Failed to load LoRA adapter '{lora_name}': {response.status_code} {response.text}"
            )

        logger.info(
            "Loaded LoRA adapter '%s' from %s", lora_name, adapter_path
        )

    def unload_lora_adapter(self, lora_name: str) -> None:
        """Unload a LoRA adapter from the vLLM server if it is present."""

        url = f"{self.server_url}/v1/unload_lora_adapter"
        try:
            response = self.session.post(url, json={"lora_name": lora_name}, timeout=60.0)
        except Exception as exc:  # noqa: BLE001 - log and propagate
            logger.error("Error unloading LoRA adapter '%s': %s", lora_name, exc)
            raise

        if response.status_code != 200:
            raise Exception(
                f"Failed to unload LoRA adapter '{lora_name}': {response.status_code} {response.text}"
            )

        logger.info("Unloaded LoRA adapter '%s'", lora_name)

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"{self.server_url}/reset_prefix_cache"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
