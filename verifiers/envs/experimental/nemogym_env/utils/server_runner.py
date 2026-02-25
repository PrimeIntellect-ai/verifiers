# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified entry point for running any resource server and accessing its tools/examples.

Usage:
    from server_runner import ServerRunner

    # List all available servers
    ServerRunner.list_servers()

    # Get tools for a server
    tools = ServerRunner.get_tools("example_single_tool_call")

    # Get example prompts
    examples = ServerRunner.get_examples("example_single_tool_call")

    # Get system prompt
    system_prompt = ServerRunner.get_system_prompt("example_single_tool_call")

    # Run a server
    ServerRunner.run("example_single_tool_call", host="localhost", port=8000)
"""

import importlib
import inspect
import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import uvicorn
from fastapi import FastAPI

from nemo_gym.base_resources_server import SimpleResourcesServer, BaseResourcesServerConfig


RESOURCES_SERVERS_DIR = Path(__file__).parent.parent / "gym" / "resources_servers"


class ServerRunner:
    """Unified interface for running resource servers and accessing their tools."""

    @staticmethod
    def list_servers() -> List[str]:
        """List all available resource server names."""
        servers = []
        for path in glob(str(RESOURCES_SERVERS_DIR / "*" / "app.py")):
            server_name = Path(path).parent.name
            servers.append(server_name)
        return sorted(servers)

    @staticmethod
    def get_server_class(server_name: str) -> Type[SimpleResourcesServer]:
        """
        Dynamically import and return the server class for a given server name.

        Args:
            server_name: Name of the server (e.g., "example_single_tool_call")

        Returns:
            The server class that extends SimpleResourcesServer
        """
        module_path = f"resources_servers.{server_name}.app"
        module = importlib.import_module(module_path)

        # Find the class that extends SimpleResourcesServer
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, SimpleResourcesServer)
                and obj is not SimpleResourcesServer
                and obj.__module__ == module.__name__):
                return obj

        raise ValueError(f"No SimpleResourcesServer subclass found in {module_path}")

    @staticmethod
    def get_config_class(server_name: str) -> Type[BaseResourcesServerConfig]:
        """Get the config class for a server."""
        module_path = f"resources_servers.{server_name}.app"
        module = importlib.import_module(module_path)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BaseResourcesServerConfig)
                and obj is not BaseResourcesServerConfig
                and obj.__module__ == module.__name__):
                return obj

        # Fall back to base config
        return BaseResourcesServerConfig

    @staticmethod
    def get_tools(server_name: str) -> List[Dict[str, Any]]:
        """
        Get the tools schema for a server from its example data.

        Args:
            server_name: Name of the server

        Returns:
            List of tool schemas (OpenAI function calling format)
        """
        example_path = RESOURCES_SERVERS_DIR / server_name / "data" / "example.jsonl"

        if not example_path.exists():
            raise FileNotFoundError(f"No example data found at {example_path}")

        with open(example_path) as f:
            first_example = json.loads(f.readline())

        return first_example.get("responses_create_params", {}).get("tools", [])

    @staticmethod
    def get_examples(server_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get example prompts for a server.

        Args:
            server_name: Name of the server
            limit: Maximum number of examples to return (default: all)

        Returns:
            List of example dictionaries with responses_create_params
        """
        example_path = RESOURCES_SERVERS_DIR / server_name / "data" / "example.jsonl"

        if not example_path.exists():
            raise FileNotFoundError(f"No example data found at {example_path}")

        examples = []
        with open(example_path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                examples.append(json.loads(line))

        return examples

    @staticmethod
    def get_single_example(server_name: str, index: int = 0) -> Dict[str, Any]:
        """
        Get a single example prompt.

        Args:
            server_name: Name of the server
            index: Which example to get (0-indexed)

        Returns:
            Single example dictionary
        """
        examples = ServerRunner.get_examples(server_name, limit=index + 1)
        if index >= len(examples):
            raise IndexError(f"Only {len(examples)} examples available, requested index {index}")
        return examples[index]

    @staticmethod
    def get_system_prompt(server_name: str) -> Optional[str]:
        """
        Get the system prompt from the first example.

        Args:
            server_name: Name of the server

        Returns:
            System prompt string, or None if not found
        """
        example = ServerRunner.get_single_example(server_name)
        inputs = example.get("responses_create_params", {}).get("input", [])

        for msg in inputs:
            if msg.get("role") in ("system", "developer"):
                return msg.get("content")

        return None

    @staticmethod
    def get_user_prompt(server_name: str, index: int = 0) -> Optional[str]:
        """
        Get the user prompt from an example.

        Args:
            server_name: Name of the server
            index: Which example to get

        Returns:
            User prompt string, or None if not found
        """
        example = ServerRunner.get_single_example(server_name, index)
        inputs = example.get("responses_create_params", {}).get("input", [])

        for msg in inputs:
            if msg.get("role") == "user":
                return msg.get("content")

        return None

    @staticmethod
    def create_app(server_name: str, host: str = "localhost", port: int = 8000) -> FastAPI:
        """
        Create a FastAPI app for a server without running it.

        Args:
            server_name: Name of the server
            host: Host to bind to
            port: Port to bind to

        Returns:
            FastAPI application instance
        """
        server_cls = ServerRunner.get_server_class(server_name)
        config_cls = ServerRunner.get_config_class(server_name)

        config = config_cls(
            host=host,
            port=port,
            entrypoint="app.py",
            name=server_name,
        )

        # Create a minimal server instance (without server_client for standalone use)
        server = server_cls.model_construct(config=config, server_client=None)

        return server.setup_webserver()

    @staticmethod
    def run(
        server_name: str,
        host: str = "localhost",
        port: int = 8000,
        reload: bool = False
    ) -> None:
        """
        Run a resource server.

        Args:
            server_name: Name of the server (e.g., "example_single_tool_call")
            host: Host to bind to
            port: Port to bind to
            reload: Enable auto-reload for development
        """
        app = ServerRunner.create_app(server_name, host, port)

        print(f"Starting {server_name} server at http://{host}:{port}")
        print(f"Tools available: {[t['name'] for t in ServerRunner.get_tools(server_name)]}")

        uvicorn.run(app, host=host, port=port, reload=reload)

    @staticmethod
    def get_server_info(server_name: str) -> Dict[str, Any]:
        """
        Get comprehensive info about a server.

        Args:
            server_name: Name of the server

        Returns:
            Dictionary with server info including tools, examples, prompts
        """
        tools = ServerRunner.get_tools(server_name)
        examples = ServerRunner.get_examples(server_name)
        system_prompt = ServerRunner.get_system_prompt(server_name)

        return {
            "name": server_name,
            "system_prompt": system_prompt,
            "tools": tools,
            "tool_names": [t["name"] for t in tools],
            "num_examples": len(examples),
            "example_user_prompts": [
                msg.get("content")
                for ex in examples
                for msg in ex.get("responses_create_params", {}).get("input", [])
                if msg.get("role") == "user"
            ],
        }


# Convenience functions
def list_servers() -> List[str]:
    """List all available servers."""
    return ServerRunner.list_servers()


def get_tools(server_name: str) -> List[Dict[str, Any]]:
    """Get tools for a server."""
    return ServerRunner.get_tools(server_name)


def get_examples(server_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get examples for a server."""
    return ServerRunner.get_examples(server_name, limit)


def get_system_prompt(server_name: str) -> Optional[str]:
    """Get system prompt for a server."""
    return ServerRunner.get_system_prompt(server_name)


def run_server(server_name: str, host: str = "localhost", port: int = 8000) -> None:
    """Run a server."""
    ServerRunner.run(server_name, host, port)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python server_runner.py <command> [args]")
        print("\nCommands:")
        print("  list                    - List all available servers")
        print("  info <server_name>      - Show server info (tools, prompts, etc.)")
        print("  tools <server_name>     - Show tools for a server")
        print("  prompt <server_name>    - Show system prompt for a server")
        print("  run <server_name> [port] - Run a server")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        print("Available servers:")
        for name in ServerRunner.list_servers():
            print(f"  - {name}")

    elif command == "info":
        if len(sys.argv) < 3:
            print("Usage: python server_runner.py info <server_name>")
            sys.exit(1)
        info = ServerRunner.get_server_info(sys.argv[2])
        print(json.dumps(info, indent=2))

    elif command == "tools":
        if len(sys.argv) < 3:
            print("Usage: python server_runner.py tools <server_name>")
            sys.exit(1)
        tools = ServerRunner.get_tools(sys.argv[2])
        print(json.dumps(tools, indent=2))

    elif command == "prompt":
        if len(sys.argv) < 3:
            print("Usage: python server_runner.py prompt <server_name>")
            sys.exit(1)
        prompt = ServerRunner.get_system_prompt(sys.argv[2])
        print(prompt or "(no system prompt found)")

    elif command == "run":
        if len(sys.argv) < 3:
            print("Usage: python server_runner.py run <server_name> [port]")
            sys.exit(1)
        server_name = sys.argv[2]
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
        ServerRunner.run(server_name, port=port)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
