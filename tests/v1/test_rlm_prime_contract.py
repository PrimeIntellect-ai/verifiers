"""Deterministic RLM training-contract E2E in a Prime VM."""

import asyncio
import json
import os
import time
from typing import Any

import pytest

from verifiers.v1.harnesses.rlm import RLMHarnessConfig

CODEWORD = "violet-cascade-731"
TOOL_STAMP = "resume-ok-9d2"
FAKE_API_KEY_VAR = "RLM_CONTRACT_E2E_API_KEY"


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _completion(body: dict[str, Any], sequence: int) -> tuple[dict[str, Any], str]:
    messages = body["messages"]
    users = [
        _message_text(message) for message in messages if message["role"] == "user"
    ]
    last_role = messages[-1]["role"]

    if any(text == "child-one" for text in users):
        message, label = {"role": "assistant", "content": "CHILD-ONE"}, "child-one"
    elif any(text == "child-two" for text in users):
        message, label = {"role": "assistant", "content": "CHILD-TWO"}, "child-two"
    elif last_role == "tool":
        message, label = (
            {"role": "assistant", "content": f"{CODEWORD} [{TOOL_STAMP}]"},
            "root-final",
        )
    elif any("Call the `recall` tool" in text for text in users):
        code = (
            "import asyncio\n"
            "import os\n"
            "import subprocess\n"
            "assert os.environ.get('TASK_VISIBLE') == 'yes'\n"
            "assert os.environ.get('EXPLICIT_TASK') == 'also-yes'\n"
            "private = ('RLM_API_KEY', 'SERPER_API_KEY')\n"
            "assert all(name not in os.environ for name in private)\n"
            "child_env = subprocess.check_output(['env'], text=True)\n"
            "assert all(f'{name}=' not in child_env for name in private)\n"
            "tool_result, children = await asyncio.gather(\n"
            f"    resume_recall.run(codeword={CODEWORD!r}),\n"
            "    asyncio.gather(rlm('child-one'), rlm('child-two')),\n"
            ")\n"
            "print(tool_result)\n"
            "print([child.answer for child in children])"
        )
        message, label = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-contract-ipython",
                        "type": "function",
                        "function": {
                            "name": "ipython",
                            "arguments": json.dumps({"code": code, "timeout": 120}),
                        },
                    }
                ],
            },
            "root-tool",
        )
    else:
        message, label = {"role": "assistant", "content": "READY"}, "root-ready"

    finish_reason = "tool_calls" if "tool_calls" in message else "stop"
    return (
        {
            "id": f"chatcmpl-contract-{sequence}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "contract-model"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
        label,
    )


@pytest.mark.e2e
@pytest.mark.prime
async def test_rlm_training_contract_in_prime_vm(run_v1, tmp_path, monkeypatch):
    calls: list[str] = []

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            headers = {
                name.strip().lower(): value.strip()
                for line in head.decode("latin-1").split("\r\n")[1:]
                if line and (name_value := line.split(":", 1))
                for name, value in [name_value]
            }
            body = json.loads(
                await reader.readexactly(int(headers.get("content-length", "0")))
            )
            response, label = _completion(body, len(calls))
            calls.append(label)
            payload = json.dumps(response, separators=(",", ":")).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    monkeypatch.setenv(FAKE_API_KEY_VAR, "contract-secret")
    harness = RLMHarnessConfig(
        id="rlm",
        version=os.environ.get("NANO_RLM_E2E_VERSION", "main"),
        max_depth=1,
        max_concurrent_subagents=2,
        max_subagent_calls=2,
        kernel_env={"TASK_VISIBLE": "yes", "EXPLICIT_TASK": "also-yes"},
        env={
            "RLM_API_KEY": "ambient-provider-secret",
            "SERPER_API_KEY": "search-secret",
        },
    )

    try:
        (trace,) = await run_v1(
            "echo-acp-resume-v1",
            harness=harness,
            runtime={"type": "prime", "vm": True},
            env={
                "agent": {
                    "client": {
                        "type": "eval",
                        "base_url": f"http://127.0.0.1:{port}/v1",
                        "api_key_var": FAKE_API_KEY_VAR,
                    }
                }
            },
            output_dir=tmp_path,
            max_turns=8,
            max_tokens=8192,
            rollout_timeout=600,
        )
    finally:
        server.close()
        await server.wait_closed()

    assert trace.ok, trace.errors
    assert trace.rewards["resumed"].score == 1.0
    assert set(calls) == {
        "root-ready",
        "root-tool",
        "root-final",
        "child-one",
        "child-two",
    }
    assert trace.metrics["sub_rlm_num_calls"] == 2
    assert trace.metrics["has_sub_rlm"] == 1
