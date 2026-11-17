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
    completed = {}
    closed = False

    def stop(pid, sig):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, sig)

    def retire(identifier):
        process = processes.get(identifier)
        if (
            process is not None
            and process.returncode is not None
            and process.stdout.at_eof()
            and process.stderr.at_eof()
        ):
            process.stdin.close()
            # Late reads and lifecycle calls still work without retaining pipes,
            # readers, or transports for every completed command.
            completed[identifier] = process.returncode
            del processes[identifier], locks[identifier]

    async def drain(stream):
        while await stream.read(65536):
            pass

    async def reap():
        active = [
            (identifier, process, locks[identifier])
            for identifier, process in processes.items()
        ]

        async def drain_locked(process, lock, stream):
            async with lock:
                await drain(getattr(process, stream))

        for _, process, _ in active:
            stop(process.pid, signal.SIGKILL)
        # Waiting without draining can hang on a child's full output pipe.
        await asyncio.gather(
            *(
                drain_locked(process, stream_locks[stream], stream)
                for _, process, stream_locks in active
                for stream in ("stdout", "stderr")
            ),
            *(process.wait() for _, process, _ in active),
        )
        for identifier, _, _ in active:
            retire(identifier)

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
                stop(process.pid, signal.SIGKILL)
                process.stdin.close()
                await asyncio.gather(
                    drain(process.stdout), drain(process.stderr), process.wait()
                )
                raise RuntimeError("workspace worker closed during exec")
            identifier = uuid.uuid4().hex
            processes[identifier] = process
            locks[identifier] = {
                name: asyncio.Lock() for name in ("stdin", "stdout", "stderr")
            }
            return identifier
        identifier = message["process_id"]
        if identifier in completed:
            status = completed[identifier]
            if operation in ("poll", "wait"):
                return status
            if operation == "read" and message["stream"] in ("stdout", "stderr"):
                return None
            if operation in ("kill", "terminate"):
                return None
            if operation == "write":
                raise BrokenPipeError("workspace process has exited")
            raise ValueError("unknown workspace operation")
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
            retire(identifier)
            return base64.b64encode(data).decode() if data else None
        if operation == "poll":
            retire(identifier)
            return process.returncode
        if operation == "wait":
            status = await process.wait()
            retire(identifier)
            return status
        if operation in ("kill", "terminate"):
            stop(process.pid, signal.SIGKILL if operation == "kill" else signal.SIGTERM)
            return None
        raise ValueError("unknown workspace operation")

    try:
        await serve(handle)  # noqa: F821 — bundled stdlib RPC protocol
    finally:
        closed = True
        await reap()


if __name__ == "__main__":
    asyncio.run(main())
