"""Scripted inference and event log for the real-harness delivery examples."""

import asyncio
import json
import time
from dataclasses import dataclass, field

from starlette.responses import JSONResponse, StreamingResponse

LABELS = ("MESSAGE_ALPHA", "MESSAGE_BETA")


def labels(value):
    text = json.dumps(value)
    return sorted((label for label in LABELS if label in text), key=text.index)


@dataclass
class Scenario:
    harness: str
    state: str
    events: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    model_entered: asyncio.Event = field(default_factory=asyncio.Event)
    model_release: asyncio.Event = field(default_factory=asyncio.Event)
    tool_entered: asyncio.Event = field(default_factory=asyncio.Event)
    tool_release: asyncio.Event = field(default_factory=asyncio.Event)
    response_sent: asyncio.Event = field(default_factory=asyncio.Event)
    endpoint: str = ""
    hold_call: int = 1

    def record(self, event, **data):
        self.events.append({"index": len(self.events), "event": event, **data})

    async def barrier(self, request):
        self.record("tool_started")
        self.tool_entered.set()
        await self.tool_release.wait()
        self.record("tool_finished")
        return JSONResponse({"result": "work finished"})

    async def respond(self, request):
        data = await request.json()
        number = len(self.calls) + 1
        visible = labels(data.get("messages", data.get("input", [])))
        self.calls.append({"number": number, "messages": visible})
        self.record("model_request", call=number, messages=visible)
        if number == self.hold_call:
            self.model_entered.set()
        text = "Observed: " + (", ".join(visible) or "opening prompt") + "."
        tool = None
        if number == 1 and self.state in ("tool", "wait"):
            if self.state == "wait":
                tool = ("wait", {"timeout": 2})
            elif self.harness == "rlm":
                tool = (
                    "ipython",
                    {
                        "code": f"import urllib.request\nprint(urllib.request.urlopen('{self.endpoint}/barrier').read().decode())"
                    },
                )
            else:
                tool = (
                    "Bash",
                    {
                        "command": f"curl -fsS {self.endpoint}/barrier",
                        "description": "Wait for the demonstration work to finish",
                    },
                )
        if self.harness == "rlm":
            if number == self.hold_call and self.state in (
                "model",
                "idle",
                "idle-burst",
            ):
                await self.model_release.wait()
            message = {"role": "assistant", "content": text}
            if tool:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tool_{number}",
                            "type": "function",
                            "function": {
                                "name": tool[0],
                                "arguments": json.dumps(tool[1]),
                            },
                        }
                    ],
                }
            self.record("model_response", call=number, tool=tool[0] if tool else None)
            if number == 1:
                self.response_sent.set()
            return JSONResponse(
                {
                    "id": f"chatcmpl-{number}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": data["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if tool else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                }
            )

        async def stream():
            def event(kind, **fields):
                return (
                    "event: "
                    + kind
                    + "\ndata: "
                    + json.dumps({"type": kind, **fields})
                    + "\n\n"
                )

            yield event(
                "message_start",
                message={
                    "id": f"msg_{number}",
                    "type": "message",
                    "role": "assistant",
                    "model": data["model"],
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                },
            )
            try:
                if number == self.hold_call and self.state in (
                    "model",
                    "idle",
                    "idle-burst",
                ):
                    await self.model_release.wait()
                if tool:
                    yield event(
                        "content_block_start",
                        index=0,
                        content_block={
                            "type": "tool_use",
                            "id": f"tool_{number}",
                            "name": tool[0],
                            "input": {},
                        },
                    )
                    yield event(
                        "content_block_delta",
                        index=0,
                        delta={
                            "type": "input_json_delta",
                            "partial_json": json.dumps(tool[1]),
                        },
                    )
                else:
                    yield event(
                        "content_block_start",
                        index=0,
                        content_block={"type": "text", "text": ""},
                    )
                    yield event(
                        "content_block_delta",
                        index=0,
                        delta={"type": "text_delta", "text": text},
                    )
                yield event("content_block_stop", index=0)
                yield event(
                    "message_delta",
                    delta={
                        "stop_reason": "tool_use" if tool else "end_turn",
                        "stop_sequence": None,
                    },
                    usage={"output_tokens": 10},
                )
                yield event("message_stop")
                self.record(
                    "model_response", call=number, tool=tool[0] if tool else None
                )
                if number == 1:
                    self.response_sent.set()
            except asyncio.CancelledError:
                self.record("model_stream_cancelled", call=number)
                raise

        return StreamingResponse(stream(), media_type="text/event-stream")
