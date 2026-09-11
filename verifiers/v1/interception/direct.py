"""Runtime-side client embedded in harness programs with in-process tool loops."""

import os
import sys

import httpx


class ToolInterceptionClient:
    def __init__(self, url: str, secret: str) -> None:
        self.url = url
        self.secret = secret
        self.client = httpx.Client(
            timeout=httpx.Timeout(35, connect=5), follow_redirects=False
        )

    def request(self, body: dict) -> dict:
        response = self.client.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.secret}",
            },
            json=body,
        )
        response.raise_for_status()
        decision = response.json()
        if not isinstance(decision, dict):
            raise TypeError("tool policy returned a non-object decision")
        action = decision.get("action")
        if action not in {"allow", "rewrite", "stop"}:
            raise RuntimeError(f"invalid tool policy action: {action!r}")
        if action == "rewrite" and not isinstance(decision.get("message"), dict):
            raise RuntimeError("tool policy rewrite omitted its message")
        return decision

    def call(self, phase: str, message: dict) -> dict:
        decision = self.request({"phase": phase, "message": message})
        if decision["action"] == "stop":
            raise RuntimeError(
                decision.get("reason") or "tool policy stopped the rollout"
            )
        return decision

    def close(self) -> None:
        self.client.close()


def read_tool_secret(size: int, harness: str) -> str:
    if not size:
        return ""
    payload = sys.stdin.buffer.read(size)
    if len(payload) != size:
        raise RuntimeError(f"{harness} interception credential handoff ended early")
    devnull = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(devnull, sys.stdin.fileno())
    finally:
        os.close(devnull)
    return payload.decode()
