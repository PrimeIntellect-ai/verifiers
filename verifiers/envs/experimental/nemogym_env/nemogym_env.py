import json
import uuid
import httpx
import asyncio
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from datasets import Dataset

import verifiers as vf
from verifiers.types import State, Completion
from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest, APIClient
from src.utils import (
    ensure_repo,
    load_dataset_rows,
    get_oai_tools,
    completions_to_nemo_response,
    convert_nemo_messages_to_oai,
    make_tool_caller,
    check_server
)


class NemogymEnv(vf.StatefulToolEnv):
    def __init__(
        self, 
        **kwargs
    ):
        self.run_id = uuid.uuid4()
        self.resource_server = resource_server
        self.repo_path = repo_path