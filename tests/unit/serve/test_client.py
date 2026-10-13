"""`EnvClient.close()` drains the fire-and-forget cancels a cancelled run leaves behind."""

import subprocess
import sys

# `_request` parks a cancelled run's fire-and-forget cancel in `_cancel_tasks` with a
# `discard` done-callback. One loop turn later the sends have finished but the callbacks
# are still queued behind us, so `close()` meets a set of finished tasks.
CLOSE_WITH_FINISHED_CANCELS = """
import asyncio, uuid
from verifiers.v1.serve.client import EnvClient

async def main():
    client = EnvClient("tcp://127.0.0.1:1")
    for _ in range(3):
        task = asyncio.get_running_loop().create_task(client._send_cancel(uuid.uuid4().hex))
        client._cancel_tasks.add(task)
        task.add_done_callback(client._cancel_tasks.discard)
    await asyncio.sleep(0)
    assert all(task.done() for task in client._cancel_tasks) and client._cancel_tasks
    await client.close()

asyncio.run(main())
"""


def test_close_drains_finished_cancels():
    """`close()` returns once every fire-and-forget cancel has run, also when all of them
    finished before it looked: the state Ctrl-C leaves behind in served mode. A child
    process bounds the check, since a `close()` that never yields to the loop never lets
    an in-loop timeout fire either."""
    subprocess.run(
        [sys.executable, "-c", CLOSE_WITH_FINISHED_CANCELS], check=True, timeout=30
    )
