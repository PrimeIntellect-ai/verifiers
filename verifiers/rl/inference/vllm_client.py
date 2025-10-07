import atexit
import logging
import time

import requests
import torch  # type: ignore
from openai import AsyncOpenAI
from requests import ConnectionError
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout
from trl.import_utils import is_requests_available, is_vllm_available  # type: ignore
from vllm.distributed.device_communicators.pynccl import (  # type: ignore
    PyNcclCommunicator,
)
from vllm.distributed.utils import StatelessProcessGroup  # type: ignore

logger = logging.getLogger(__name__)


class VLLMClient(AsyncOpenAI):
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
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

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.update_model_params(model)
        ```
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError(
                "requests is not installed. Please install it with `pip install requests`."
            )
        if not is_vllm_available():
            raise ImportError(
                "vLLM is not installed. Please install it with `pip install vllm`."
            )

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

        self.group_port = group_port
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

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """

        # Get the world size from the server
        url = f"{self.server_url}/get_world_size"
        try:
            response = requests.get(url)
        except Exception as e:
            logger.error(f"Failed to get world size: {e}")
            raise

        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
            logger.info(f"vLLM world size: {vllm_world_size}")
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process
        logger.info(f"Client rank: {self.rank}, total world size: {world_size}")

        # Initialize weight update group
        url = f"{self.server_url}/init_communicator"
        # Send the actual host address for the StatelessProcessGroup connection
        try:
            response = self.session.post(
                url,
                json={
                    "host": self.host,
                    "port": self.group_port,
                    "world_size": world_size,
                },
            )
        except Exception as e:
            logger.error(f"Failed to init communicator: {e}")
            raise

        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Brief delay to allow server initialization. While not strictly required (client socket will retry on
        # connection failure), this prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(
            host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
        )
        # Use device 0 like the old code - this seems to work better for multi-GPU setups
        device = 0
        logger.info(
            f"Initializing PyNcclCommunicator on device {device}, rank {self.rank}, world_size {world_size}"
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device=device)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.server_url}/update_named_param"

        # Add timeout to prevent hanging on HTTP request
        try:
            response = self.session.post(
                url, json={"name": name, "dtype": dtype, "shape": shape}, timeout=300.0
            )
        except Timeout:
            logger.error(f"Timeout waiting for server response for {name} after 300s")
            raise Exception(f"Request timeout for {name} after 300s")
        except Exception as e:
            logger.error(f"Error sending request for {name}: {e}")
            raise

        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"{self.server_url}/reset_prefix_cache"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def get_num_background_tasks(self):
        """
        Gets the number of background tasks.
        """
        url = f"{self.server_url}/get_num_background_tasks"
        response = self.session.post(url)
        return response.json()["num_background_tasks"]

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"http://{self.host}:{self.server_port}/close_communicator"

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(
                    f"Request failed: {response.status_code}, {response.text}"
                )
