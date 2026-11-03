# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["harbor-rewardkit==0.2.3"]
# ///
"""Keep RewardKit's file access and scoring in the runtime; request LLM calls over stdio."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Reserve stdout for requests; criteria and their subprocesses log to stderr.
channel = os.fdopen(os.dup(sys.stdout.fileno()), "w")
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

import litellm
import rewardkit

pending: dict[int, asyncio.Future] = {}
reader_task: asyncio.Task | None = None
next_id = 0


async def read_responses() -> None:
    reader = asyncio.StreamReader(limit=16 * 1024 * 1024)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )
    try:
        while line := await reader.readline():
            response = json.loads(line)
            pending.pop(response["id"]).set_result(response["text"])
        raise EOFError("Verifiers closed the judge connection")
    finally:
        transport.close()
        for future in pending.values():
            if not future.done():
                future.set_exception(EOFError("Verifiers closed the judge connection"))


async def complete(**kwargs):
    global reader_task, next_id
    if reader_task is None:
        reader_task = asyncio.create_task(read_responses())
    request_id = next_id
    next_id += 1
    future = pending[request_id] = asyncio.get_running_loop().create_future()
    # Only inference arguments cross this boundary; credentials stay on the host.
    request = {
        key: kwargs[key]
        for key in (
            "model",
            "messages",
            "response_format",
            "timeout",
            "reasoning_effort",
        )
        if key in kwargs
    }
    print(json.dumps({"id": request_id, "request": request}), file=channel, flush=True)
    text = await future
    # Verifiers records real usage per call. Returning none avoids counting it again
    # when the parent imports RewardKit's component results.
    return litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": text}}], usage=None
    )


if __name__ == "__main__":
    # RewardKit has no client injection hook. Override only its LLM transport in
    # this dedicated process, retaining its prompts, retries, samples and scoring.
    litellm.acompletion = complete
    rewardkit.run("/tests", workspace=Path.cwd(), output=sys.argv[1])
