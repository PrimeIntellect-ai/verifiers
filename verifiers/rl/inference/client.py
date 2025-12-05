import atexit
import logging
import time

import requests
import torch  # type: ignore
from openai import AsyncOpenAI
from requests import ConnectionError
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, Timeout
from vllm.distributed.device_communicators.pynccl import (  # type: ignore
    PyNcclCommunicator,
)
from vllm.distributed.utils import StatelessProcessGroup  # type: ignore

logger = logging.getLogger(__name__)


class VLLMClient(AsyncOpenAI):
    """
    Client class to interact with a vLLM server for inference and weight updates.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        super().__init__(base_url=f"http://{host}:{port}/v1", api_key="local")
        self.session = requests.Session()
        # configure connection pooling to handle rapid requests better
        adapter = HTTPAdapter(
            pool_connections=10, pool_maxsize=10, max_retries=3, pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.host = host
        self.server_port = port
        self.server_url = f"http://{self.host}:{self.server_port}"

        self.group_port = group_port
        self.check_server(connection_timeout)

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        url = f"{self.server_url}/health"
        start_time = time.time()

        while True:
            try:
                response = requests.get(url)
            except RequestException as exc:
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
                        "seconds. Make sure the server is running by running `vf-vllm`."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return None

            logger.info(
                f"Server is not up yet. Retrying in {retry_interval} seconds..."
            )
            time.sleep(retry_interval)

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """

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

        # initialize weight update group
        url = f"{self.server_url}/init_communicator"
        # send host address for the StatelessProcessGroup connection
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

        # Brief delay to allow server initialization, prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        pg = StatelessProcessGroup.create(
            host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
        )
        device = 0
        logger.info(
            f"Initializing PyNcclCommunicator on device {device}, rank {self.rank}, world_size {world_size}"
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device=device)
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.server_url}/update_named_param"

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

        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

    def reset_prefix_cache(self):
        url = f"{self.server_url}/reset_prefix_cache"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def get_num_background_tasks(self):
        url = f"{self.server_url}/get_num_background_tasks"
        response = self.session.post(url)
        return response.json()["num_background_tasks"]

    def close_communicator(self):
        url = f"http://{self.host}:{self.server_port}/close_communicator"

        try:
            response = self.session.post(url)
        except ConnectionError:
            pass
        else:
            if response.status_code != 200:
                raise Exception(
                    f"Request failed: {response.status_code}, {response.text}"
                )
