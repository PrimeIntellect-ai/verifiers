#!/usr/bin/env python3
"""Offline CI shard for the Fetch MCP environment.

This script launches the deterministic fixtures server and runs a targeted suite
of fetch requests using the same helper (`fetch_url_async`) that powers the MCP
tool. Each response is validated with the canonical verifiers to ensure the
endpoints, headers, query params, and truncation metadata behave as expected.

The shard intentionally covers representative scenarios (HTML parsing, HEAD,
query params, auth headers, large payload hashing, and derived metrics) while
remaining fully offline/deterministic so it can run in CI without external
dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from environments.mcp_fetch.tools.fetch_mcp_server import fetch_url_async
from environments.mcp_fetch.utils.mini_httpd import serve
from environments.mcp_fetch.verifiers import run_verifier

SCRIPT_ROOT = Path(__file__).resolve().parent
ENV_ROOT = SCRIPT_ROOT.parent
TASKS_PATH = ENV_ROOT / "tasks" / "qa.jsonl"
DEFAULT_PORT = 31415

SHARD_CASES: List[Dict[str, Any]] = [
    {
        "id": "fetch_014",
        "request": {"path": "/text/poem.txt", "method": "GET"},
    },
    {
        "id": "fetch_006",
        "request": {"path": "/html/about.html", "method": "HEAD"},
    },
    {
        "id": "fetch_009",
        "request": {"path": "/query?category=fruits&limit=2"},
    },
    {
        "id": "fetch_011",
        "request": {
            "path": "/auth",
            "headers": {"X-Token": "opensesame"},
        },
    },
    {
        "id": "fetch_018",
        "request": {"path": "/json/data_large.jsonl"},
    },
    {
        "id": "fetch_027",
        "request": {"path": "/json/data_large.jsonl"},
    },
    {
        "id": "fetch_030",
        "request": {"path": "/json/ledger.json"},
    },
]


def load_task_map() -> Dict[str, Dict[str, Any]]:
    tasks: Dict[str, Dict[str, Any]] = {}
    with TASKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tasks[str(data["id"])] = data
    return tasks


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def start_fixture_server(port: int) -> threading.Thread | None:
    if _port_in_use(port):
        print(f"[offline-shard] Detected running fixtures on port {port}; reusing existing server.")
        return None
    thread = threading.Thread(target=serve, kwargs={"port": port}, daemon=True)
    thread.start()
    time.sleep(0.5)
    return thread


def _build_request(case: Dict[str, Any], port: int) -> Dict[str, Any]:
    request = dict(case.get("request", {}))
    rel_path = request.pop("path", None)
    if not rel_path:
        raise ValueError(f"Shard case {case['id']} missing request path")
    request.setdefault("method", "GET")
    request["url"] = f"http://127.0.0.1:{port}{rel_path}"
    return request


async def run_case(case: Dict[str, Any], verifier: Dict[str, Any], port: int) -> bool:
    request = _build_request(case, port)
    payload = await fetch_url_async(**request)
    result = run_verifier(verifier, payload)
    if result is None:
        raise ValueError(f"Shard case {case['id']} references a judge verifier.")
    return bool(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline shard verifier.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Fixture server port.")
    args = parser.parse_args()

    start_fixture_server(args.port)
    tasks = load_task_map()

    failures: List[str] = []
    for case in SHARD_CASES:
        task_id = case["id"]
        verifier = tasks[task_id]["verifier"]
        try:
            success = asyncio.run(run_case(case, verifier, args.port))
        except Exception as exc:  # noqa: BLE001
            print(f"[offline-shard] {task_id}: ERROR ({exc})")
            failures.append(task_id)
            continue
        status = "PASS" if success else "FAIL"
        print(f"[offline-shard] {task_id}: {status}")
        if not success:
            failures.append(task_id)

    if failures:
        print(f"[offline-shard] Failed cases: {', '.join(failures)}")
        return 1

    print("[offline-shard] All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
