"""Live Worlds connections. Credentials never belong in task data or traces."""

import uuid
from dataclasses import dataclass, field

import httpx


@dataclass
class WorldConnection:
    url: str
    world_id: str
    account: str
    token: str = field(repr=False)

    async def call(self, operation: str, **arguments) -> dict:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.url.rstrip('/')}/worlds/{self.world_id}/tools/{operation}",
                headers={"Authorization": f"Bearer {self.token}"},
                json=arguments,
            )
            response.raise_for_status()
            return response.json()

    async def mutate(self, operation: str, **arguments) -> dict:
        return await self.call(operation, idempotency_key=uuid.uuid4().hex, **arguments)


# No dependency installation or platform credential is needed in the sandbox.
# Pass '-' explicitly to read JSON from stdin; omitted JSON defaults to {}.
WORLD_PROGRAM = """import json, os, sys, urllib.request, urllib.error, uuid
if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
    print("Usage: world.py OPERATION [JSON|-] [--mutate]\\n"
          "Omitted JSON means {}. Use - to read JSON from stdin.\\n"
          "Discover operations: world.py describe_tools '{}'")
    sys.exit(0)
operation = sys.argv[1]
args = [arg for arg in sys.argv[2:] if arg != "--mutate"]
arguments = json.load(sys.stdin) if args == ["-"] else json.loads(args[0]) if args else {}
if "--mutate" in sys.argv[2:]:
    arguments.setdefault("idempotency_key", uuid.uuid4().hex)
url = os.environ["WORLDS_URL"].rstrip("/")
world = os.environ["WORLDS_ID"]
request = urllib.request.Request(
    f"{url}/worlds/{world}/tools/{operation}",
    data=json.dumps(arguments).encode(),
    headers={"Authorization": "Bearer " + os.environ["WORLDS_TOKEN"],
             "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        print(response.read().decode())
except urllib.error.HTTPError as error:
    print(json.dumps({"status": error.code, "error": error.read().decode()}))
    sys.exit(1)
"""
