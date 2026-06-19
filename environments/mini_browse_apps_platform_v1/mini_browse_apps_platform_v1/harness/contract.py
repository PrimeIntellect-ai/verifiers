"""Public payload contract consumed by the Mini Browse harness."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

TASK_PAYLOAD_PATH = "/task/mini_browse/task.json"
RESULT_PATH = "/task/mini_browse/result.json"
TRANSCRIPT_PATH = "/logs/mini_browse/transcript.json"
METRICS_PATH = "/logs/mini_browse/metrics.json"
PROGRESS_PATH = "/logs/mini_browse/progress.jsonl"
WORKSPACE_ROOT = "/workspace/mini-browse"


class MiniBrowseTaskPayload(BaseModel):
    """Sandbox-visible task payload for the Mini Browse harness."""

    model_config = ConfigDict(extra="forbid")

    instruction: str
    output_schema: dict[str, Any]
    browser_api_url: str
    start_url: str = "about:blank"
    http_proxy: str | None = None
    source: str = "verifiers-mini-browse"
    task_preamble: str | None = None
