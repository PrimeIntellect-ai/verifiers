#!/usr/bin/env python3
"""
build_all_binaries.py

For each task in all_tasks.jsonl:
  1. Clone GitHub repo at pinned commit
  2. Build native Linux binary using compile_hint
  3. Run strings/nm/file/objdump analysis
  4. Upload binary to HuggingFace
  5. Append result to build_results.jsonl

Usage:
    HF_TOKEN=... uv run python3 scripts/build_all_binaries.py \
        --jsonl environments/programbench/data/all_tasks.jsonl \
        [--resume]    # skip tasks already in build_results.jsonl
        [--only-lang go]  # filter to a single language

Dependencies on PATH: go, cargo, gcc, g++, cmake, make, git, file, nm, objdump, strings
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or open(Path.home() / ".cache/huggingface/token").read().strip()
)
BINARY_HF_REPO = "PrimeIntellect/programbench-processed"


def run(
    cmd: str, cwd: str | None = None, timeout: int = 600, env: dict | None = None
) -> tuple[int, str]:
    full_env = dict(os.environ)
    full_env["PATH"] = (
        "/root/.cargo/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    if env:
        full_env.update(env)
    proc = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
        env=full_env,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def build_binary(
    task_id: str, lang: str, compile_hint: str, src_dir: Path, executable: Path
) -> tuple[bool, str]:
    """Try compile_hint then fallback heuristics. Return (success, log)."""
    actual_cmd = (
        compile_hint.replace("/workspace/src", str(src_dir))
        .replace("/workspace/executable", str(executable))
        .replace("/workspace", str(src_dir.parent))
    )

    rc, out = run(actual_cmd, timeout=600)
    if rc == 0 and executable.exists():
        return True, out

    log.warning("[%s] compile_hint failed (exit %d), trying fallbacks", task_id, rc)
    build_log = f"[compile_hint failed]\n{out[-2000:]}\n"

    if lang == "go":
        rc2, o2 = run(f"go build -o {executable} ./...", cwd=str(src_dir), timeout=300)
        if rc2 == 0 and executable.exists():
            return True, build_log + o2

    elif lang == "rust":
        rc2, o2 = run("cargo build --release", cwd=str(src_dir), timeout=600)
        build_log += o2
        if rc2 == 0:
            # Find the release binary
            rel = src_dir / "target" / "release"
            candidates = (
                [
                    f
                    for f in rel.iterdir()
                    if f.is_file()
                    and os.access(f, os.X_OK)
                    and f.suffix not in (".d", ".rlib", ".so", ".a", ".dylib")
                ]
                if rel.exists()
                else []
            )
            if candidates:
                shutil.copy2(str(candidates[0]), str(executable))
                return True, build_log

    elif lang in ("c", "cpp"):
        # cmake
        if (src_dir / "CMakeLists.txt").exists():
            build_dir = src_dir / "_cmake_build"
            build_dir.mkdir(exist_ok=True)
            rc2, o2 = run(
                f"cmake -DCMAKE_BUILD_TYPE=Release {src_dir} && make -j$(nproc)",
                cwd=str(build_dir),
                timeout=600,
            )
            build_log += o2
            if rc2 == 0:
                for f in build_dir.rglob("*"):
                    if f.is_file() and os.access(f, os.X_OK) and not f.suffix:
                        shutil.copy2(str(f), str(executable))
                        return True, build_log
        # autoconf
        if (src_dir / "configure").exists() or (src_dir / "autogen.sh").exists():
            if (src_dir / "autogen.sh").exists():
                run("bash autogen.sh", cwd=str(src_dir), timeout=120)
            elif not (src_dir / "configure").exists():
                run("autoreconf -fi", cwd=str(src_dir), timeout=120)
            rc2, o2 = run(
                "./configure && make -j$(nproc)", cwd=str(src_dir), timeout=600
            )
            build_log += o2
            if rc2 == 0:
                for f in src_dir.rglob("*"):
                    if f.is_file() and os.access(f, os.X_OK) and not f.suffix:
                        shutil.copy2(str(f), str(executable))
                        return True, build_log
        # plain make
        if (src_dir / "Makefile").exists():
            rc2, o2 = run("make -j$(nproc)", cwd=str(src_dir), timeout=600)
            build_log += o2
            if rc2 == 0:
                for f in src_dir.rglob("*"):
                    if f.is_file() and os.access(f, os.X_OK) and not f.suffix:
                        shutil.copy2(str(f), str(executable))
                        return True, build_log

    return False, build_log


def analyze(executable: Path) -> dict:
    rc, file_type = run(f"file -b {executable}")
    _, strings_out = run(f"strings {executable} | head -500")
    _, nm_out = run(f"nm --defined-only {executable} | head -300")
    _, objdump_head = run(f"objdump -d {executable} | head -200")
    return {
        "file_type": file_type.strip(),
        "strings_output": strings_out[:3000],
        "nm_output": nm_out[:3000],
        "objdump_head": objdump_head[:3000],
        "binary_size": executable.stat().st_size,
    }


def upload_binary(binary_path: Path, task_id: str, api: HfApi) -> str:
    hf_filename = f"binaries/{task_id}/executable"
    for attempt in range(3):
        try:
            api.upload_file(
                path_or_fileobj=str(binary_path),
                path_in_repo=hf_filename,
                repo_id=BINARY_HF_REPO,
                repo_type="dataset",
            )
            return hf_filename
        except Exception as e:
            log.warning(
                "HF upload attempt %d failed for %s: %s", attempt + 1, task_id, e
            )
            time.sleep(2**attempt)
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl", default="environments/programbench/data/all_tasks.jsonl"
    )
    parser.add_argument(
        "--output-dir", default="environments/programbench/data/binaries"
    )
    parser.add_argument(
        "--results",
        default="environments/programbench/data/binaries/build_results.jsonl",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip tasks already in results file"
    )
    parser.add_argument(
        "--only-lang", default=None, help="Only build tasks of this language"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results)

    # Load tasks
    tasks = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    log.info("Loaded %d tasks", len(tasks))

    if args.only_lang:
        tasks = [t for t in tasks if t.get("language") == args.only_lang]
        log.info("Filtered to %d %s tasks", len(tasks), args.only_lang)

    # Already done
    done: set[str] = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                if rec.get("success"):
                    done.add(rec["task_id"])
        log.info("Resuming: %d tasks already complete", len(done))

    api = HfApi(token=HF_TOKEN)

    built = 0
    failed = 0
    skipped = 0

    with (
        open(results_path, "a") as results_f,
        tempfile.TemporaryDirectory(prefix="pb_build_") as tmpdir,
    ):
        for task in tasks:
            task_id = task["task_id"]
            lang = task.get("language", "c")
            compile_hint = task.get("compile_hint", "")

            if task_id in done:
                log.info("[%s] already built, skipping", task_id)
                skipped += 1
                continue

            log.info("══ %s [%s] ══", task_id, lang)

            # Parse owner/repo/commit from task_id: owner__repo.commit
            base, short_commit = task_id.rsplit(".", 1)
            owner, repo = base.split("__", 1)
            clone_url = f"https://github.com/{owner}/{repo}.git"

            task_dir = Path(tmpdir) / task_id
            src_dir = task_dir / "src"
            executable = task_dir / "executable"
            task_dir.mkdir(exist_ok=True)

            # Clone
            rc, out = run(f"git clone --depth=200 {clone_url} {src_dir}", timeout=120)
            if rc != 0:
                log.error("[%s] clone failed", task_id)
                results_f.write(
                    json.dumps(
                        {"task_id": task_id, "success": False, "error": "clone_failed"}
                    )
                    + "\n"
                )
                results_f.flush()
                failed += 1
                continue

            # Checkout pinned commit
            rc, _ = run(f"git checkout {short_commit}", cwd=str(src_dir), timeout=30)
            if rc != 0:
                run(
                    f"git fetch --depth=200 origin {short_commit}",
                    cwd=str(src_dir),
                    timeout=60,
                )
                run("git checkout FETCH_HEAD", cwd=str(src_dir), timeout=30)

            # Build
            success, build_log = build_binary(
                task_id, lang, compile_hint, src_dir, executable
            )
            if not success or not executable.exists():
                log.error("[%s] build failed", task_id)
                results_f.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "success": False,
                            "error": "build_failed",
                            "build_log": build_log[-1000:],
                        }
                    )
                    + "\n"
                )
                results_f.flush()
                failed += 1
                # Clean up to free disk
                shutil.rmtree(str(task_dir), ignore_errors=True)
                continue

            log.info("[%s] built OK (%d bytes)", task_id, executable.stat().st_size)

            # Save binary locally
            saved_binary = out_dir / f"{task_id}_executable"
            shutil.copy2(str(executable), str(saved_binary))

            # Analyze
            analysis = analyze(executable)

            # Upload to HF
            hf_filename = upload_binary(saved_binary, task_id, api)
            if not hf_filename:
                log.warning("[%s] HF upload failed — keeping local only", task_id)

            rec = {
                "task_id": task_id,
                "success": True,
                "binary_hf_repo": BINARY_HF_REPO if hf_filename else "",
                "binary_hf_filename": hf_filename,
                **analysis,
            }
            results_f.write(json.dumps(rec) + "\n")
            results_f.flush()
            built += 1
            log.info("[%s] done", task_id)

            # Clean src to free disk (keep saved binary)
            shutil.rmtree(str(task_dir), ignore_errors=True)

    log.info(
        "═══ Build complete: built=%d failed=%d skipped=%d ═══", built, failed, skipped
    )


if __name__ == "__main__":
    main()
