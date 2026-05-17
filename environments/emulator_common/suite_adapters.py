import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonData: TypeAlias = dict[str, JsonValue]

CONTRACT_VERSION = "emulator-runner-v1"
DEFAULT_EXECUTABLE = "target/release/emulator-runner"

ROM_EXTENSIONS_BY_PLATFORM = {
    "chip8": (".ch8", ".c8", ".rom"),
    "i8080_space_invaders": (".bin", ".rom", ".com"),
    "gameboy_dmg": (".gb", ".gbc"),
    "gameboy_cgb": (".gbc", ".gb"),
    "nes": (".nes",),
    "sms": (".sms", ".gg", ".sg"),
    "gba": (".gba", ".mb", ".bin"),
    "genesis": (".md", ".gen", ".bin", ".smd", ".json", ".json.gz"),
    "snes": (".sfc", ".smc", ".bs"),
    "ps1": (".exe", ".ps-exe", ".cue", ".iso", ".bin"),
}

VISUAL_SIGNALS = {
    "framebuffer_hash",
    "frame_sequence_hash",
    "palette_hash",
    "stability_hash",
}
TRACE_SIGNALS = {
    "trace_crc",
    "cpu_trace_crc",
    "event_log_crc",
    "memory_snapshot_hash",
    "mapper_event_hash",
    "swi_trace_crc",
    "gte_trace_crc",
    "serial_pass",
    "pass_marker",
    "suite_pass_rate",
    "timeout_free_run",
}
AUDIO_SIGNALS = {"audio_crc"}


@dataclass(frozen=True)
class CaseRun:
    case_id: str
    case_name: str
    artifact: str
    passed: bool
    output: JsonData
    stdout: str
    stderr: str
    returncode: int
    elapsed_seconds: float


def stable_hash(value: JsonValue) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _text_tail(value: str | bytes | None, limit: int = 12000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def run_command(
    command: list[str],
    cwd: Path,
    timeout: int,
    *,
    env: dict[str, str] | None = None,
) -> JsonData:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": _text_tail(completed.stdout),
            "stderr": _text_tail(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": 124,
            "stdout": _text_tail(exc.stdout),
            "stderr": _text_tail(f"{_text_tail(exc.stderr)}\nTIMEOUT"),
        }


def build_workspace(workspace: Path, timeout: int) -> JsonData:
    cargo_test = run_command(
        ["cargo", "test", "--all", "--", "--nocapture"],
        workspace,
        timeout,
    )
    cargo_build = run_command(["cargo", "build", "--release"], workspace, timeout)
    return {"cargo_test": cargo_test, "cargo_build": cargo_build}


def locate_runner(workspace: Path) -> Path | None:
    env_runner = os.environ.get("EMULATOR_RUNNER")
    candidates = []
    if env_runner:
        candidates.append(Path(env_runner))
    candidates.extend(
        [
            workspace / "emulator-runner",
            workspace / DEFAULT_EXECUTABLE,
            workspace / "target/release/emulator_benchmark",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def check_runner_contract(
    workspace: Path, runner: Path | None, timeout: int
) -> JsonData:
    if runner is None:
        return {"ok": False, "error": "missing runner executable"}
    version = run_command([str(runner), "--contract-version"], workspace, timeout)
    self_test = run_command([str(runner), "--self-test"], workspace, timeout)
    version_text = str(version.get("stdout") or "").strip()
    version_ok = version["ok"] and CONTRACT_VERSION in version_text
    return {
        "ok": bool(version_ok and self_test["ok"]),
        "version": version,
        "self_test": self_test,
        "runner": str(runner),
    }


def fetch_source(source: JsonData, destination: Path, timeout: int) -> JsonData:
    name = str(source["name"])
    kind = str(source["kind"])
    target = destination / safe_name(name)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "git":
        sparse_paths = [str(path) for path in source.get("sparse_paths", [])]
        command = ["git", "clone", "--depth", "1"]
        if sparse_paths:
            command.extend(["--filter", "blob:none", "--sparse"])
        ref = source.get("ref")
        if ref:
            command.extend(["--branch", str(ref)])
        command.extend([str(source["url"]), str(target)])
        started = time.monotonic()
        result = run_command(command, destination, timeout)
        if result["ok"] and sparse_paths:
            sparse = run_command(
                ["git", "-C", str(target), "sparse-checkout", "set", *sparse_paths],
                destination,
                timeout,
            )
            result["ok"] = bool(sparse["ok"])
            result["returncode"] = sparse["returncode"]
            result["stdout"] = f"{result.get('stdout', '')}\n{sparse.get('stdout', '')}"
            result["stderr"] = f"{result.get('stderr', '')}\n{sparse.get('stderr', '')}"
        if result["ok"]:
            write_source_metadata(target, source)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["path"] = str(target)
        result["source"] = name
        result["sparse_paths"] = sparse_paths
        return result
    if kind == "http":
        target.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "verifiers-emulator-suite-fetch/0.1"}
        try:
            request = urllib.request.Request(str(source["url"]), headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
            (target / "index.html").write_bytes(body)
            write_source_metadata(target, source)
            return {
                "ok": True,
                "returncode": 0,
                "stdout": f"HTTP bytes={len(body)}",
                "stderr": "",
                "path": str(target),
                "source": name,
            }
        except Exception as exc:
            return {
                "ok": False,
                "returncode": 1,
                "stdout": "",
                "stderr": str(exc),
                "path": str(target),
                "source": name,
            }
    if kind == "local":
        local_path = Path(str(source["url"])).expanduser()
        if not local_path.exists():
            return {
                "ok": False,
                "returncode": 1,
                "stdout": "",
                "stderr": f"local source missing: {local_path}",
                "path": str(target),
                "source": name,
            }
        if local_path.is_dir():
            shutil.copytree(local_path, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, target / local_path.name)
        write_source_metadata(target, source)
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "local source copied",
            "stderr": "",
            "path": str(target),
            "source": name,
        }
    if kind == "private-artifact":
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "private artifact source documented; mounted separately",
            "stderr": "",
            "path": str(target),
            "source": name,
            "private_artifact": True,
        }
    return {
        "ok": False,
        "returncode": 1,
        "stdout": "",
        "stderr": f"unsupported source kind: {kind}",
        "path": str(target),
        "source": name,
    }


def safe_name(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in out.split("-") if part)[:96] or "source"


def write_source_metadata(target: Path, source: JsonData) -> None:
    metadata = {
        key: source[key]
        for key in ("artifact_globs", "artifact_extensions")
        if key in source
    }
    if metadata:
        (target / ".emulator_source.json").write_text(
            json.dumps(metadata, sort_keys=True),
            encoding="utf-8",
        )


def source_artifact_candidates(root: Path) -> list[Path]:
    metadata_path = root / ".emulator_source.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        globs = [str(pattern) for pattern in metadata.get("artifact_globs", [])]
        if globs:
            paths: list[Path] = []
            for pattern in globs:
                paths.extend(root.glob(pattern))
            return paths
    return list(root.rglob("*"))


def source_artifact_extensions(root: Path, default: tuple[str, ...]) -> tuple[str, ...]:
    metadata_path = root / ".emulator_source.json"
    if not metadata_path.exists():
        return default
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    extensions = tuple(
        str(value).lower() for value in metadata.get("artifact_extensions", [])
    )
    return extensions or default


def discover_artifacts(platform: str, source_roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in source_roots:
        if not root.exists():
            continue
        extensions = source_artifact_extensions(
            root, ROM_EXTENSIONS_BY_PLATFORM.get(platform, (".rom", ".bin"))
        )
        for path in source_artifact_candidates(root):
            if path.is_file() and path.name.lower().endswith(extensions):
                if path.stat().st_size > 0:
                    found.append(path)
    return sorted(set(found), key=lambda p: str(p))[:128]


def private_artifact_paths(public_sources: JsonData) -> list[JsonData]:
    artifact_root = Path(
        os.environ.get("EMULATOR_PRIVATE_ARTIFACT_DIR", "/private/emulator")
    )
    rows: list[JsonData] = []
    for artifact in public_sources.get("private_artifacts", []):
        name = str(artifact.get("name", "artifact"))
        candidates = [
            artifact_root / name,
            artifact_root / f"{name}.bin",
            artifact_root / f"{name}.rom",
        ]
        mounted = [str(path) for path in candidates if path.exists()]
        row = dict(artifact)
        row["mounted_paths"] = mounted
        row["available"] = bool(mounted)
        rows.append(row)
    return rows


def case_for_artifact(
    artifact: Path, public_cases: list[JsonData], index: int
) -> JsonData:
    if not public_cases:
        return {
            "id": f"public_artifact_{index}",
            "name": artifact.name,
            "expected_signal": "suite_pass_rate",
        }
    lowered = str(artifact).lower()
    for case in public_cases:
        tokens = [
            token
            for token in str(case.get("name", "")).lower().replace("-", " ").split()
            if len(token) >= 3
        ]
        if tokens and any(token in lowered for token in tokens):
            return case
    return public_cases[index % len(public_cases)]


def invoke_case(
    workspace: Path,
    runner: Path,
    platform: str,
    case: JsonData,
    artifact: Path,
    runtime: JsonData,
    timeout: int,
) -> CaseRun:
    output_path = (
        workspace
        / "verification"
        / f"{case['id']}-{stable_hash(str(artifact))[:10]}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cycles = int(runtime.get("smoke_cycles", 10000))
    frames = max(1, int(runtime.get("smoke_frames", 1)))
    command = [
        str(runner),
        "--platform",
        platform,
        "--case",
        str(case["id"]),
        "--rom",
        str(artifact),
        "--cycles",
        str(cycles),
        "--frames",
        str(frames),
        "--output",
        str(output_path),
    ]
    started = time.monotonic()
    result = run_command(command, workspace, timeout)
    elapsed = time.monotonic() - started
    output: JsonData = {}
    if output_path.exists():
        try:
            output = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            output = {"parse_error": str(exc)}
    passed = bool(result["ok"] and output.get("passed") is True)
    return CaseRun(
        case_id=str(case["id"]),
        case_name=str(case.get("name", case["id"])),
        artifact=str(artifact),
        passed=passed,
        output=output,
        stdout=str(result.get("stdout") or ""),
        stderr=str(result.get("stderr") or ""),
        returncode=int(result.get("returncode", 1)),
        elapsed_seconds=round(elapsed, 3),
    )


def replay_determinism(
    workspace: Path,
    runner: Path,
    platform: str,
    case: JsonData,
    artifact: Path,
    runtime: JsonData,
    timeout: int,
) -> JsonData:
    first = invoke_case(workspace, runner, platform, case, artifact, runtime, timeout)
    second = invoke_case(workspace, runner, platform, case, artifact, runtime, timeout)
    keys = [
        "passed",
        "cycles",
        "frames",
        "serial",
        "framebuffer_sha256",
        "frame_sequence_sha256",
        "trace_crc32",
        "audio_crc32",
        "perf",
    ]
    lhs = {key: first.output.get(key) for key in keys if key in first.output}
    rhs = {key: second.output.get(key) for key in keys if key in second.output}
    return {
        "ok": bool(first.passed and second.passed and lhs == rhs),
        "first": first.output,
        "second": second.output,
        "artifact": str(artifact),
        "case_id": str(case["id"]),
    }


def signal_scores(cases: list[JsonData], runs: list[CaseRun]) -> dict[str, float]:
    expected_by_id = {
        str(case["id"]): str(case.get("expected_signal", "")) for case in cases
    }
    visual_total = trace_total = audio_total = 0
    visual_hits = trace_hits = audio_hits = 0
    for run in runs:
        expected = expected_by_id.get(run.case_id, "")
        output = run.output
        if expected in VISUAL_SIGNALS or "frame" in expected or "palette" in expected:
            visual_total += 1
            if output.get("framebuffer_sha256") or output.get("frame_sequence_sha256"):
                visual_hits += 1
        if expected in TRACE_SIGNALS or "trace" in expected or "serial" in expected:
            trace_total += 1
            if output.get("trace_crc32") or output.get("serial") or run.passed:
                trace_hits += 1
        if expected in AUDIO_SIGNALS or "audio" in expected:
            audio_total += 1
            if output.get("audio_crc32"):
                audio_hits += 1
    return {
        "framebuffer_hash_score": visual_hits / visual_total if visual_total else 1.0,
        "trace_cpu_score": trace_hits / trace_total if trace_total else 1.0,
        "audio_perf_score": audio_hits / audio_total if audio_total else 1.0,
    }


def verify_workspace(
    workspace: Path | str,
    manifest: JsonData,
    public_sources: JsonData,
    *,
    source_root: Path | str | None = None,
    timeout: int | None = None,
    skip_build: bool = False,
) -> JsonData:
    workspace = Path(workspace)
    timeout = int(
        timeout or manifest.get("runtime", {}).get("verify_timeout_seconds", 900)
    )
    platform = str(manifest["platform"])
    public_cases = list(manifest.get("public_cases", []))
    runtime = dict(manifest.get("runtime", {}))

    if skip_build:
        build = {
            "cargo_test": {
                "ok": True,
                "returncode": 0,
                "stdout": "build skipped by fixture",
                "stderr": "",
            },
            "cargo_build": {
                "ok": True,
                "returncode": 0,
                "stdout": "build skipped by fixture",
                "stderr": "",
            },
        }
    else:
        build = build_workspace(workspace, timeout)
    runner = locate_runner(workspace)
    contract = check_runner_contract(workspace, runner, min(timeout, 120))

    source_cache = Path(source_root or workspace / "verification" / "public_suites")
    source_cache.mkdir(parents=True, exist_ok=True)
    fetches = [
        fetch_source(source, source_cache, min(timeout, 240))
        for source in public_sources.get("sources", [])
    ]
    fetched_roots = [
        Path(str(row["path"]))
        for row in fetches
        if row.get("ok") and not row.get("private_artifact")
    ]
    public_artifacts = discover_artifacts(platform, fetched_roots)
    private_artifacts = private_artifact_paths(public_sources)
    private_artifact_roots: list[Path] = []
    private_artifact_files: list[Path] = []
    for row in private_artifacts:
        for mounted_path in row.get("mounted_paths", []):
            path = Path(str(mounted_path))
            if path.is_dir():
                private_artifact_roots.append(path)
            elif path.is_file():
                private_artifact_files.append(path)
    private_discovered = discover_artifacts(platform, private_artifact_roots)
    artifacts = sorted(
        set([*public_artifacts, *private_discovered, *private_artifact_files]),
        key=lambda path: str(path),
    )[:128]

    runs: list[CaseRun] = []
    if runner is not None and contract.get("ok"):
        for index, artifact in enumerate(artifacts):
            case = case_for_artifact(artifact, public_cases, index)
            runs.append(
                invoke_case(
                    workspace,
                    runner,
                    platform,
                    case,
                    artifact,
                    runtime,
                    timeout,
                )
            )

    deterministic = {"ok": False, "reason": "no executed artifact"}
    if runner is not None and contract.get("ok") and artifacts:
        case = case_for_artifact(artifacts[0], public_cases, 0)
        deterministic = replay_determinism(
            workspace, runner, platform, case, artifacts[0], runtime, timeout
        )

    source_success = (
        sum(1 for row in fetches if row.get("ok")) / len(fetches) if fetches else 0.0
    )
    build_score = float(build["cargo_test"]["ok"] and build["cargo_build"]["ok"])
    contract_score = 1.0 if contract.get("ok") else 0.0
    public_pass_rate = sum(1 for run in runs if run.passed) / len(runs) if runs else 0.0
    runtime_stability = (
        1.0 if contract.get("ok") and build["cargo_build"]["ok"] else 0.0
    )
    signal = signal_scores(public_cases, runs)
    components = {
        "build_score": build_score,
        "runner_contract_score": contract_score,
        "public_source_score": source_success,
        "public_rom_pass_rate": public_pass_rate,
        "framebuffer_hash_score": signal["framebuffer_hash_score"],
        "trace_cpu_score": signal["trace_cpu_score"],
        "audio_perf_score": signal["audio_perf_score"],
        "deterministic_replay_score": 1.0 if deterministic.get("ok") else 0.0,
        "runtime_stability_score": runtime_stability,
        "private_artifact_availability": (
            sum(1 for row in private_artifacts if row.get("available"))
            / len(private_artifacts)
            if private_artifacts
            else 1.0
        ),
    }
    score = (
        0.12 * components["build_score"]
        + 0.10 * components["runner_contract_score"]
        + 0.08 * components["public_source_score"]
        + 0.40 * components["public_rom_pass_rate"]
        + 0.10 * components["framebuffer_hash_score"]
        + 0.08 * components["trace_cpu_score"]
        + 0.04 * components["audio_perf_score"]
        + 0.06 * components["deterministic_replay_score"]
        + 0.02 * components["runtime_stability_score"]
    )
    return {
        "score": round(float(score), 6),
        "passed": score >= 0.90,
        "components": components,
        "build": build,
        "runner_contract": contract,
        "public_sources": fetches,
        "artifact_count": len(artifacts),
        "public_artifact_count": len(public_artifacts),
        "private_artifact_count": len(private_discovered) + len(private_artifact_files),
        "artifacts": [str(path) for path in artifacts[:128]],
        "case_runs": [
            {
                "case_id": run.case_id,
                "case_name": run.case_name,
                "artifact": run.artifact,
                "passed": run.passed,
                "returncode": run.returncode,
                "elapsed_seconds": run.elapsed_seconds,
                "output": run.output,
                "stdout": run.stdout,
                "stderr": run.stderr,
            }
            for run in runs
        ],
        "deterministic_replay": deterministic,
        "private_artifacts": private_artifacts,
    }
