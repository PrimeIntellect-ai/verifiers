# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Credential-free process worker launched inside the bound workspace view.

The RPC framing is bundled by the launcher. This process inherits B's user,
cwd and environment; it never receives provider credentials.
"""

import asyncio
import base64
import contextlib
import os
import signal
import uuid


async def main():
    processes = {}
    locks = {}
    closed = False

    def stop(process, sig):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)

    async def reap():
        async def drain(identifier, stream):
            async with locks[identifier][stream]:
                while await getattr(processes[identifier], stream).read(65536):
                    pass

        for process in processes.values():
            stop(process, signal.SIGKILL)
        # Waiting without draining can hang on a child's full output pipe.
        await asyncio.gather(
            *(
                drain(identifier, stream)
                for identifier in processes
                for stream in ("stdout", "stderr")
            ),
            *(process.wait() for process in processes.values()),
        )

    async def handle(message):
        nonlocal closed
        operation = message["operation"]
        if operation == "close":
            closed = True
            await reap()
            return None
        if closed:
            raise RuntimeError("workspace worker is closed")
        if operation == "exec":
            argv, env = message["argv"], message["env"]
            if (
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(arg, str) for arg in argv)
            ):
                raise ValueError("workspace argv must contain strings")
            if not isinstance(env, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()
            ):
                raise ValueError("workspace environment must contain strings")
            process = await asyncio.create_subprocess_exec(
                *argv,
                env={**os.environ, **env},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            if closed:
                stop(process, signal.SIGKILL)
                await process.wait()
                raise RuntimeError("workspace worker closed during exec")
            identifier = uuid.uuid4().hex
            processes[identifier] = process
            locks[identifier] = {
                name: asyncio.Lock() for name in ("stdin", "stdout", "stderr")
            }
            return identifier
        identifier = message["process_id"]
        process = processes[identifier]
        if operation == "write":
            data = base64.b64decode(message["data"], validate=True)
            async with locks[identifier]["stdin"]:
                process.stdin.write(data)
                await process.stdin.drain()
            return None
        if operation == "read":
            stream = message["stream"]
            if stream not in ("stdout", "stderr"):
                raise ValueError("unknown workspace stream")
            async with locks[identifier][stream]:
                data = await getattr(process, stream).read(65536)
            return base64.b64encode(data).decode() if data else None
        if operation == "poll":
            return process.returncode
        if operation == "wait":
            return await process.wait()
        if operation in ("kill", "terminate"):
            stop(process, signal.SIGKILL if operation == "kill" else signal.SIGTERM)
            return None
        raise ValueError("unknown workspace operation")

    try:
        await serve(handle)  # noqa: F821 — bundled stdlib RPC protocol
    finally:
        closed = True
        await reap()


if __name__ == "__main__":
    asyncio.run(main())
