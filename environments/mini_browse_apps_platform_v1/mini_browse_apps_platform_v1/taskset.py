"""mini-browse-apps-platform-v1: local-app Mini Browse tasks pulled from the Prime hub."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any

from pydantic import Field
from verifiers.v1.errors import TasksetError

import verifiers.v1 as vf

from .harness.contract import (
    METRICS_PATH,
    PROGRESS_PATH,
    RESULT_PATH,
    TASK_PAYLOAD_PATH,
    TRANSCRIPT_PATH,
    MiniBrowseTaskPayload,
)
from .harness.diagnostics import read_jsonl_tail
from .judge import judge_answer_key, score_from_judge_payload

DEFAULT_SANDBOX_IMAGE = (
    "team-cmlr3u2er002zhr01tj8f48ts/"
    "mini-browse-apps:destination-autocomplete-tight-20260528-0027"
)
DEFAULT_HUB_ENV_ID = "prime/mini-browse-apps-platform-v1"
DEFAULT_DATASET_FILENAME = "google_flights_10.jsonl.gz"

APP_PORT = 5173
CDP_PORT = 18080
APP_URL = f"http://127.0.0.1:{APP_PORT}"
BROWSER_API_URL = f"http://127.0.0.1:{CDP_PORT}"
WORKDIR = "/workspace"
APP_SEED_PATH = "/task/app_seed.json"
SERVICE_LOG_DIR = "/logs/services"
APP_LOG_PATH = f"{SERVICE_LOG_DIR}/app.log"
CDP_LOG_PATH = f"{SERVICE_LOG_DIR}/cdp.log"
APP_SERVER = "/opt/mini-browse-services/spa_server.py"
CDP_SERVER = "/opt/mini-browse-services/local_cdp_service.py"
APP_ROOT = "/opt/mini-browse-app/dist"

SETUP_TIMEOUT_SECONDS = 600
FINALIZE_TIMEOUT_SECONDS = 120
SCORING_TIMEOUT_SECONDS = 180
DEFAULT_TIMEOUT_SECONDS = 3600.0
DEFAULT_SANDBOX_CPU = 2
DEFAULT_SANDBOX_MEMORY_GB = 4
DEFAULT_SANDBOX_DISK_GB = 10
TASKSET_SHUFFLE_SEED = "google_flights_kernel_v1_dense_hard_no_bag_1188_order_v1"


class MiniBrowseAppTask(vf.Task):
    """One Mini Browse task backed by a sandboxed local web app."""

    prompt: str
    output_schema: dict[str, Any]
    answer_key: dict[str, Any]
    app_seed_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MiniBrowseAppsConfig(vf.TasksetConfig):
    id: str = "mini-browse-apps-platform-v1"
    dataset_path: str | None = None
    """Explicit local dataset (JSONL/JSONL.GZ); when set, skips the hub pull."""
    hub_env_id: str = DEFAULT_HUB_ENV_ID
    """Prime hub environment the dataset is pulled from when no `dataset_path` is given."""
    hub_version: str = "latest"
    dataset_filename: str = DEFAULT_DATASET_FILENAME
    cache_dir: str | None = None
    """Where the pulled dataset is cached (default: ~/.cache/verifiers/mini-browse-apps)."""
    seed_store_path: str | None = None
    task_indices: list[int] | None = None
    task_profile: str = "default"
    shuffle_tasks: bool = True
    taskset_shuffle_seed: str = TASKSET_SHUFFLE_SEED
    sandbox_image: str = DEFAULT_SANDBOX_IMAGE
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    judge_model: str = "google/gemini-3.1-pro-preview"
    judge_base_url: str | None = "https://api.pinference.ai/api/v1"
    judge_api_key_env: str = "PRIME_API_KEY"


class MiniBrowseAppsTaskset(vf.Taskset[MiniBrowseAppTask, MiniBrowseAppsConfig]):
    """Owns local-app rows, sandbox app startup, and submitted-result judging."""

    NEEDS_CONTAINER = True

    def __init__(self, config: MiniBrowseAppsConfig) -> None:
        super().__init__(config)
        self._inline_app_seeds: dict[str, dict[str, Any]] = {}

    def load_tasks(self) -> list[MiniBrowseAppTask]:
        rows = self.load_rows()
        rows = self.filter_rows(rows)
        if self.config.shuffle_tasks:
            rows = self.stable_shuffle_rows(rows)
        if self.config.task_indices is not None:
            rows = self.select_task_indices(rows)
        if not rows:
            raise ValueError("No Mini Browse app tasks were loaded")
        return [self.normalize_row(i, row) for i, row in enumerate(rows)]

    async def setup(self, task: MiniBrowseAppTask, runtime: vf.Runtime) -> None:
        app_seed = self.app_seed_for_task(task)
        await ensure_runtime_dirs(runtime)
        await write_runtime_json(runtime, APP_SEED_PATH, app_seed)
        public_payload = MiniBrowseTaskPayload(
            instruction=task.prompt,
            start_url=APP_URL,
            output_schema=task.output_schema,
            browser_api_url=BROWSER_API_URL,
            source="mini-browse-apps-platform-v1",
        )
        await runtime.write(
            TASK_PAYLOAD_PATH,
            public_payload.model_dump_json(indent=2).encode("utf-8"),
        )
        await start_services(runtime)
        await wait_for_services(runtime)

    async def finalize(
        self, task: MiniBrowseAppTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> None:
        del task
        result = await read_runtime_json(runtime, RESULT_PATH)
        metrics = await read_runtime_json(runtime, METRICS_PATH)
        trace.info["mini_browse_result"] = result
        trace.info["mini_browse_metrics"] = metrics
        trace.info["mini_browse_artifacts"] = {
            "result_path": RESULT_PATH,
            "transcript_path": TRANSCRIPT_PATH,
            "metrics_path": METRICS_PATH,
            "progress_path": PROGRESS_PATH,
            "task_payload_path": TASK_PAYLOAD_PATH,
            "app_seed_path": APP_SEED_PATH,
            "app_log_path": APP_LOG_PATH,
            "cdp_log_path": CDP_LOG_PATH,
        }
        if isinstance(result, dict):
            trace.info["submitted_result"] = result.get("submitted_result")
            if result.get("is_error"):
                trace.info["mini_browse_progress_tail"] = await read_jsonl_tail(
                    runtime,
                    PROGRESS_PATH,
                )

    @vf.reward(weight=1.0)
    async def answer_key(self, task: MiniBrowseAppTask, trace: vf.Trace) -> float:
        result = trace_result(trace)
        submitted = result.get("submitted_result")
        if result.get("is_error") or not submitted:
            trace.info["mini_browse_judge"] = {
                "verdict": "no",
                "explanation": result.get("error") or "missing submitted result",
            }
            return 0.0

        judge_payload = await judge_answer_key(
            task_instruction=task.prompt,
            submitted_result=submitted,
            answer_key=task.answer_key,
            output_schema=task.output_schema,
            model=self.config.judge_model,
            base_url=self.config.judge_base_url,
            api_key_env=self.config.judge_api_key_env,
        )
        trace.info["mini_browse_judge"] = judge_payload
        return score_from_judge_payload(judge_payload)

    @vf.metric
    async def result_present(self, trace: vf.Trace) -> float:
        return float(bool(trace_result(trace)))

    @vf.metric
    async def submitted_result_present(self, trace: vf.Trace) -> float:
        return float(bool(trace_result(trace).get("submitted_result")))

    @vf.metric
    async def agent_error(self, trace: vf.Trace) -> float:
        return float(bool(trace_result(trace).get("is_error")))

    @vf.metric
    async def transcript_image_count(self, trace: vf.Trace) -> float:
        return metric(trace, "transcript_image_count")

    @vf.metric
    async def message_count(self, trace: vf.Trace) -> float:
        return metric(trace, "message_count")

    def load_rows(self) -> list[dict[str, Any]]:
        path = self.resolved_dataset_path()
        if path.suffix == ".gz" or path.suffixes[-2:] == [".jsonl", ".gz"]:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def resolved_dataset_path(self) -> Path:
        if self.config.dataset_path:
            path = Path(self.config.dataset_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Mini Browse app dataset not found: {path}")
            return path
        return self.ensure_cached_dataset()

    def ensure_cached_dataset(self) -> Path:
        cache_root = (
            Path(self.config.cache_dir).expanduser()
            if self.config.cache_dir
            else Path.home() / ".cache" / "verifiers" / "mini-browse-apps"
        )
        cached = cache_root / self.config.hub_version / self.config.dataset_filename
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            self.pull_dataset_into(cached)
        return cached

    def pull_dataset_into(self, dest: Path) -> None:
        """Pull the env package from the Prime hub into a temp dir and copy the dataset out."""
        with tempfile.TemporaryDirectory(prefix="mini-browse-hub-") as tmp:
            result = subprocess.run(
                [
                    "prime", "env", "pull", self.config.hub_env_id,
                    "-v", self.config.hub_version, "-t", tmp, "--plain",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()[-1000:]
                raise RuntimeError(
                    f"`prime env pull {self.config.hub_env_id}` failed: {detail}"
                )
            matches = sorted(Path(tmp).rglob(self.config.dataset_filename))
            if not matches:
                raise FileNotFoundError(
                    f"{self.config.dataset_filename!r} not found in pulled hub env "
                    f"{self.config.hub_env_id!r}"
                )
            staging = dest.with_name(dest.name + ".tmp")
            shutil.copyfile(matches[0], staging)
            os.replace(staging, dest)

    def filter_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.config.task_profile == "default":
            return rows
        return [row for row in rows if source_profile(row) == self.config.task_profile]

    def stable_shuffle_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seed = self.config.taskset_shuffle_seed

        def key(row: dict[str, Any]) -> tuple[str, str]:
            info = decode_info(row.get("info") or {})
            task_id = str(row.get("task_id") or info.get("task_name") or "")
            digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()
            return digest, task_id

        return sorted(rows, key=key)

    def select_task_indices(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected = []
        for index in self.config.task_indices or []:
            if index < 0 or index >= len(rows):
                raise ValueError(
                    f"task_indices contains out-of-range index {index}; "
                    f"filtered taskset has {len(rows)} rows"
                )
            selected.append(rows[index])
        return selected

    def normalize_row(self, index: int, row: dict[str, Any]) -> MiniBrowseAppTask:
        info = decode_info(row.get("info") or {})
        raw_instruction = info.get("instruction") or row.get("question")
        if not isinstance(raw_instruction, str) or not raw_instruction.strip():
            raise ValueError(f"row {index} is missing a task instruction")
        output_schema = info.get("output_schema")
        if not isinstance(output_schema, dict):
            raise ValueError(f"row {index} is missing output_schema")
        answer_key = info.get("answer_key") or parse_answer(row.get("answer"))
        if not isinstance(answer_key, dict):
            raise ValueError(f"row {index} is missing answer_key")

        task_name = str(row.get("task_id") or info.get("task_name") or index)
        app_seed = info.get("app_seed")
        if app_seed is not None and not isinstance(app_seed, dict):
            raise ValueError(f"row {index} has non-object app_seed")
        app_seed_ref = info.get("app_seed_ref")
        if app_seed is not None and app_seed_ref:
            self._inline_app_seeds[str(app_seed_ref)] = app_seed

        return MiniBrowseAppTask(
            idx=index,
            name=task_name,
            prompt=raw_instruction.strip(),
            image=self.config.sandbox_image,
            workdir=WORKDIR,
            timeout=vf.TaskTimeout(
                setup=SETUP_TIMEOUT_SECONDS,
                harness=self.config.timeout_seconds,
                finalize=FINALIZE_TIMEOUT_SECONDS,
                scoring=SCORING_TIMEOUT_SECONDS,
            ),
            resources=vf.TaskResources(
                cpu=DEFAULT_SANDBOX_CPU,
                memory=DEFAULT_SANDBOX_MEMORY_GB,
                disk=DEFAULT_SANDBOX_DISK_GB,
            ),
            output_schema=output_schema,
            answer_key=answer_key,
            app_seed_ref=str(app_seed_ref) if app_seed_ref else None,
            metadata={
                "task_name": info.get("task_name"),
                "task_id": row.get("task_id") or answer_key.get("task_id"),
                "answer_kind": answer_key.get("answer_kind"),
                "source_profile": source_profile(row),
                "source_dataset": info.get("source_dataset"),
            },
        )

    def app_seed_for_task(self, task: MiniBrowseAppTask) -> dict[str, Any]:
        if not task.app_seed_ref:
            raise ValueError(f"Task {task.name} has neither app_seed nor app_seed_ref")
        inline_seed = self._inline_app_seeds.get(task.app_seed_ref)
        if inline_seed is not None:
            return inline_seed
        seed_store = self.resolved_seed_store_path()
        if seed_store is None:
            raise ValueError(
                f"Task {task.name} needs seed {task.app_seed_ref}, but no seed store "
                "was configured"
            )
        return load_seed(seed_store, task.app_seed_ref)

    def resolved_seed_store_path(self) -> Path | None:
        if self.config.seed_store_path:
            return Path(self.config.seed_store_path).expanduser()
        if not self.config.dataset_path:
            return None
        dataset_path = Path(self.config.dataset_path).expanduser()
        return seed_store_for_artifact_path(dataset_path)


async def ensure_runtime_dirs(runtime: vf.Runtime) -> None:
    result = await runtime.run(
        [
            "bash",
            "-lc",
            f"mkdir -p /task {WORKDIR} {SERVICE_LOG_DIR} "
            f"{shlex.quote(str(Path(TASK_PAYLOAD_PATH).parent))}",
        ],
        {},
    )
    if result.exit_code != 0:
        raise TasksetError(
            f"Mini Browse app setup failed: {combined_output(result)}"
        )


async def start_services(runtime: vf.Runtime) -> None:
    await runtime.run_background(
        [
            "python3",
            APP_SERVER,
            "--host",
            "127.0.0.1",
            "--port",
            str(APP_PORT),
            "--root",
            APP_ROOT,
        ],
        {"TASK_SEED_PATH": APP_SEED_PATH},
        APP_LOG_PATH,
    )
    await runtime.run_background(
        [
            "python3",
            CDP_SERVER,
            "--host",
            "127.0.0.1",
            "--port",
            str(CDP_PORT),
            "--chrome",
            "/usr/bin/chromium",
            "--headless",
        ],
        {},
        CDP_LOG_PATH,
    )


async def wait_for_services(runtime: vf.Runtime) -> None:
    script = f"""\
set -e
for i in $(seq 1 90); do
  if curl --noproxy '*' -fsS --max-time 2 {APP_URL} >/dev/null \\
    && curl --noproxy '*' -fsS --max-time 2 {BROWSER_API_URL}/healthz >/dev/null; then
    echo "services ready"
    exit 0
  fi
  sleep 1
done
echo "service readiness failed"
echo "--- process list ---"
ps aux || true
echo "--- app log ---"
tail -120 {APP_LOG_PATH} 2>/dev/null || true
echo "--- cdp log ---"
tail -120 {CDP_LOG_PATH} 2>/dev/null || true
exit 1
"""
    result = await runtime.run(["bash", "-lc", script], {})
    if result.exit_code != 0:
        raise TasksetError(
            f"Mini Browse app services did not become ready: {combined_output(result)}"
        )


async def write_runtime_json(runtime: vf.Runtime, path: str, value: Any) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    await runtime.write(path, data)


async def read_runtime_json(runtime: vf.Runtime, path: str) -> Any:
    try:
        raw = await runtime.read(path)
    except Exception as exc:
        return {"is_error": True, "error": f"missing runtime artifact {path}: {exc}"}
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "is_error": True,
            "error": f"invalid JSON artifact {path}: {text[:500]}",
        }


def trace_result(trace: vf.Trace) -> dict[str, Any]:
    result = trace.info.get("mini_browse_result")
    return result if isinstance(result, dict) else {}


def metric(trace: vf.Trace, key: str) -> float:
    metrics = trace.info.get("mini_browse_metrics")
    if isinstance(metrics, dict):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    value = trace_result(trace).get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def decode_info(info: Any) -> dict[str, Any]:
    if isinstance(info, str):
        return json.loads(info)
    return dict(info or {})


def parse_answer(answer: Any) -> Any:
    if isinstance(answer, str):
        return json.loads(answer)
    return answer


def source_profile(row: dict[str, Any]) -> str | None:
    info = decode_info(row.get("info") or {})
    factory = info.get("factory") or {}
    if not isinstance(factory, dict):
        return None
    profile = factory.get("source_profile")
    return str(profile) if profile else None


def seed_store_for_artifact_path(path: Path) -> Path:
    name = path.name
    if name.endswith(".tasks.jsonl.gz"):
        return path.with_name(name.removesuffix(".tasks.jsonl.gz") + ".seeds.sqlite")
    if name.endswith(".jsonl.gz"):
        return path.with_name(name.removesuffix(".jsonl.gz") + ".seeds.sqlite")
    if path.suffix:
        return path.with_suffix(".seeds.sqlite")
    return path.with_name(name + ".seeds.sqlite")


def load_seed(seed_store: Path, seed_ref: str) -> dict[str, Any]:
    if not seed_store.exists():
        raise FileNotFoundError(f"Mini Browse app seed store not found: {seed_store}")
    with sqlite3.connect(seed_store) as db:
        columns = {
            row[1] for row in db.execute("PRAGMA table_info(app_seeds)").fetchall()
        }
        if "app_seed_zlib" in columns:
            row = db.execute(
                "SELECT app_seed_zlib FROM app_seeds WHERE seed_id = ?",
                (seed_ref,),
            ).fetchone()
            seed_json = None if row is None else zlib.decompress(row[0]).decode("utf-8")
        else:
            row = db.execute(
                "SELECT app_seed_json FROM app_seeds WHERE seed_id = ?",
                (seed_ref,),
            ).fetchone()
            seed_json = None if row is None else row[0]
    if seed_json is None:
        raise KeyError(f"Seed {seed_ref!r} not found in {seed_store}")
    seed = json.loads(seed_json)
    if not isinstance(seed, dict):
        raise ValueError(f"Seed {seed_ref!r} in {seed_store} is not an object")
    return seed


def combined_output(result: vf.ProgramResult) -> str:
    return ((result.stdout or "") + (result.stderr or "")).strip()[-2000:]


__all__ = ["MiniBrowseAppTask", "MiniBrowseAppsConfig", "MiniBrowseAppsTaskset"]
