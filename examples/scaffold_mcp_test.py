"""
Test MCPScaffold with actual MCP servers.

This tests the MCP integration to ensure tools can be loaded from
external MCP servers and used in the scaffold.
"""

import argparse
import asyncio
import sys

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.scaffolds import MCPScaffold, Scaffold


async def test_mcp_scaffold(client: AsyncOpenAI, model: str):
    """Test MCPScaffold with mcp-server-fetch."""

    print("=" * 60)
    print("MCP SCAFFOLD TEST")
    print("=" * 60)
    print(f"Model: {model}")
    print()

    # MCP server configuration for the fetch server
    mcp_servers = [
        {
            "name": "fetch",
            "command": "uvx",
            "args": ["mcp-server-fetch"],
            "description": "Fetch web content",
        }
    ]

    print("Creating MCPScaffold with mcp-server-fetch...")

    try:
        scaffold = MCPScaffold(
            client=client,
            model=model,
            mcp_servers=mcp_servers,
            max_tool_turns=3,
            sampling_args={"temperature": 0, "max_tokens": 500},
        )

        print("Setting up MCP connections...")
        await scaffold.setup()

        print(f"Available tools: {list(scaffold.tool_map.keys())}")
        print()

        # Test 1: Basic request without needing tools
        print("-" * 40)
        print("Test 1: Basic request (no tools needed)")
        messages = [{"role": "user", "content": "What is 2+2? Just give the number."}]
        result = await scaffold.generate(messages, state={})
        print(f"Response: {result.messages[-1].get('content', '')[:100]}")
        print(f"Tool calls: {result.tool_calls_made}")
        print()

        # Test 2: Request that could use fetch tool
        print("-" * 40)
        print("Test 2: Request that could use fetch tool")
        messages = [
            {
                "role": "user",
                "content": "Use the fetch tool to get the content from https://httpbin.org/json and tell me what you find.",
            }
        ]
        result = await scaffold.generate(messages, state={})
        print(f"Response: {result.messages[-1].get('content', '')[:200]}...")
        print(f"Tool calls: {result.tool_calls_made}")
        print()

        # Cleanup
        print("Cleaning up MCP connections...")
        await scaffold.teardown()

        print("=" * 60)
        print("MCP SCAFFOLD TEST COMPLETE")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(description="MCP Scaffold Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model to use")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    args = parser.parse_args()

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    success = await test_mcp_scaffold(client, args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
