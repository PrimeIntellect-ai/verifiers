#!/usr/bin/env python3
"""Simple gpt-5 sanity check using the Responses API."""

import json
import os

from openai import OpenAI

SYSTEM = "You are a calibration assistant for the Fetch MCP environment."
USER = "Fetch http://127.0.0.1:31415/html/index.html and return the exact H1 text."

messages = [
    {"role": "system", "content": [{"type": "input_text", "text": SYSTEM}]},
    {"role": "user", "content": [{"type": "input_text", "text": USER}]},
]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.responses.create(
    model="gpt-5",
    input=messages,
    text={"format": {"type": "text"}},
    reasoning={"effort": "medium", "summary": "auto"},
)


def _extract_output(resp_dict: dict) -> str:
    output = resp_dict.get("output", [])
    text_parts = []
    for item in output:
        if item.get("type") == "message":
            for chunk in item.get("content", []):
                if chunk.get("type") in {"text", "output_text"}:
                    text_parts.append(chunk.get("text", ""))
        elif item.get("type") == "tool_call":
            print("Tool call:", json.dumps(item, indent=2))
    return "\n".join(text_parts).strip()


resp_dict = response.model_dump()
print("Raw response:\n", json.dumps(resp_dict, indent=2))
print("\nExtracted output:\n", _extract_output(resp_dict) or "<empty>")
