---
title: Echo Environment Server
emoji: 🔊
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Echo Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Echo environment is through the `EchoEnv` class. The client is **async by default**:

```python
import asyncio
from echo_env import EchoAction, EchoEnv

async def main():
    # Create environment from Docker image
    client = await EchoEnv.from_docker_image("echo-env:latest")

    async with client:
        # Reset
        result = await client.reset()
        print(f"Reset: {result.observation.echoed_message}")

        # Send multiple messages
        messages = ["Hello, World!", "Testing echo", "Final message"]

        for msg in messages:
            result = await client.step(EchoAction(message=msg))
            print(f"Sent: '{msg}'")
            print(f"  → Echoed: '{result.observation.echoed_message}'")
            print(f"  → Length: {result.observation.message_length}")
            print(f"  → Reward: {result.reward}")

asyncio.run(main())
```

For **synchronous usage**, use the `.sync()` wrapper:

```python
from echo_env import EchoAction, EchoEnv

with EchoEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello!"))
    print(result.observation.echoed_message)
```

The `EchoEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when the context manager exits

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t echo-env:latest -f envs/echo_env/server/Dockerfile .
```

## Environment Details

### Action
**EchoAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**EchoObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have an Echo environment server running, you can connect directly:

```python
from echo_env import EchoAction, EchoEnv

# Async usage
async with EchoEnv(base_url="http://localhost:8000") as client:
    result = await client.reset()
    result = await client.step(EchoAction(message="Hello!"))

# Sync usage
with EchoEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello!"))
```

Note: When connecting to an existing server, closing the client will NOT stop the server.

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 envs/echo_env/server/test_echo_env.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running the Full Example

Run the complete example that demonstrates the full workflow:

```bash
python3 examples/local_echo_env.py
```

This example shows:
- Creating an environment from a Docker image
- Resetting and stepping through the environment
- Automatic cleanup with `close()`

## Project Structure

```
echo_env/
├── __init__.py            # Module exports
├── README.md              # This file
├── client.py              # EchoEnv client implementation
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── echo_environment.py  # Core environment logic
    ├── app.py             # FastAPI application
    ├── test_echo_env.py   # Direct environment tests
    └── Dockerfile         # Container image definition
```
