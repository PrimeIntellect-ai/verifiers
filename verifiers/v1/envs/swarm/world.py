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
# The CLI accepts JSON on stdin so shell quoting need not contain credentials.
WORLD_PROGRAM = """import json, os, sys, urllib.request, uuid
operation = sys.argv[1]
arguments = json.loads(sys.argv[2]) if len(sys.argv) > 2 else json.load(sys.stdin)
if len(sys.argv) > 3 and sys.argv[3] == "--mutate":
    arguments.setdefault("idempotency_key", uuid.uuid4().hex)
url = os.environ["WORLDS_URL"].rstrip("/")
world = os.environ["WORLDS_ID"]
request = urllib.request.Request(
    f"{url}/worlds/{world}/tools/{operation}",
    data=json.dumps(arguments).encode(),
    headers={"Authorization": "Bearer " + os.environ["WORLDS_TOKEN"],
             "Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=60) as response:
    print(response.read().decode())
"""
