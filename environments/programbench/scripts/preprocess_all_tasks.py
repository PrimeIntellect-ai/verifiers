#!/usr/bin/env python3
"""
preprocess_all_tasks.py

Fetch all 200 ProgramBench task metadata and push a complete (no-binary)
dataset to PrimeIntellect/programbench-processed.

Requires: HF_TOKEN, GITHUB_TOKEN (optional but avoids rate limiting).
Run on any machine — no special arch needed.

Usage:
    HF_TOKEN=$(cat ~/.cache/huggingface/token) \
    GITHUB_TOKEN=$(gh auth token) \
    uv run python3 scripts/preprocess_all_tasks.py
"""

from __future__ import annotations

import os
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import Dataset
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or open(Path.home() / ".cache/huggingface/token").read().strip()
)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""

TESTS_REPO = "programbench/ProgramBench-Tests"
OUT_REPO = "PrimeIntellect/programbench-processed"

LANG_MAP = {
    "Go": "go",
    "Rust": "rust",
    "C": "c",
    "C++": "cpp",
    "Python": "python",
    "Haskell": "haskell",
    "Java": "java",
    "Nix": "nix",
}

DEFAULT_COMPILE_HINTS = {
    "go": "cd /workspace/src && go build -o /workspace/executable .",
    "rust": "cd /workspace/src && cargo build --release && cp target/release/$(basename $(ls -1 target/release/ | grep -vE '\\.(d|rlib|so|a|dylib|pdb)$' | head -1)) /workspace/executable",
    "c": "cd /workspace/src && gcc -O2 -o /workspace/executable *.c",
    "cpp": "cd /workspace/src && g++ -O2 -std=c++17 -o /workspace/executable *.cpp",
}

# Overrides for known-tricky compile hints
COMPILE_HINT_OVERRIDES: dict[str, str] = {
    # C/C++ projects that need cmake or specific make targets
    "doxygen__doxygen": "cd /workspace/src && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) doxygen && cp bin/doxygen /workspace/executable",
    "duckdb__duckdb": "cd /workspace/src && make -j$(nproc) && cp build/release/duckdb /workspace/executable",
    "ninja-build__ninja": "cd /workspace/src && python3 configure.py --bootstrap && cp ninja /workspace/executable",
    "facebook__zstd": "cd /workspace/src && make -j$(nproc) && cp zstd /workspace/executable",
    "lz4__lz4": "cd /workspace/src && make -j$(nproc) && cp lz4 /workspace/executable",
    "google__brotli": "cd /workspace/src && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) brotli && cp brotli /workspace/executable",
    "gromacs__gromacs": "cd /workspace/src && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DGMX_BUILD_OWN_FFTW=ON .. && make -j$(nproc) gmx && cp bin/gmx /workspace/executable",
    "ffmpeg__ffmpeg": "cd /workspace/src && ./configure --enable-gpl --disable-x86asm && make -j$(nproc) && cp ffmpeg /workspace/executable",
    "facebookresearch__fasttext": "cd /workspace/src && make -j$(nproc) && cp fasttext /workspace/executable",
    "sqlite__sqlite": "cd /workspace/src && ./configure && make -j$(nproc) sqlite3 && cp sqlite3 /workspace/executable",
    "lua__lua": "cd /workspace/src && make -j$(nproc) linux && cp src/lua /workspace/executable",
    "luajit__luajit": "cd /workspace/src && make -j$(nproc) && cp src/luajit /workspace/executable",
    "php__php-src": "cd /workspace/src && ./buildconf && ./configure --disable-all && make -j$(nproc) && cp sapi/cli/php /workspace/executable",
    "htop-dev__htop": "cd /workspace/src && ./autogen.sh && ./configure && make -j$(nproc) && cp htop /workspace/executable",
    "osgeo__gdal": "cd /workspace/src && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) gdal_translate && cp apps/gdal_translate /workspace/executable",
    "osgeo__proj": "cd /workspace/src && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) proj && cp bin/proj /workspace/executable",
    "tree-sitter__tree-sitter": "cd /workspace/src && cargo build --release && cp target/release/tree-sitter /workspace/executable",
    "bellard__quickjs": "cd /workspace/src && make -j$(nproc) && cp qjs /workspace/executable",
    "tinycc__tinycc": "cd /workspace/src && ./configure && make -j$(nproc) && cp tcc /workspace/executable",
    "ggreer__the_silver_searcher": "cd /workspace/src && ./build.sh && cp ag /workspace/executable",
    "robertdavidgraham__masscan": "cd /workspace/src && make -j$(nproc) && cp bin/masscan /workspace/executable",
    "jarun__nnn": "cd /workspace/src && make -j$(nproc) && cp nnn /workspace/executable",
    "halitechallenge__halite": "cd /workspace/src && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) && cp halite /workspace/executable",
    "arthursonzogni__json-tui": "cd /workspace/src && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) && cp ./json-tui /workspace/executable",
    "danmar__cppcheck": "cd /workspace/src && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) cppcheck && cp bin/cppcheck /workspace/executable",
    "chirlu__sox": "cd /workspace/src && autoreconf -fi && ./configure && make -j$(nproc) && cp src/sox /workspace/executable",
    "hpjansson__chafa": "cd /workspace/src && autoreconf -fi && ./configure && make -j$(nproc) && cp tools/chafa /workspace/executable",
    "jonas__tig": "cd /workspace/src && autoreconf -fi && ./configure && make -j$(nproc) && cp src/tig /workspace/executable",
    "lfos__calcurse": "cd /workspace/src && autoreconf -fi && ./configure && make -j$(nproc) && cp src/calcurse /workspace/executable",
    "lh3__seqtk": "cd /workspace/src && make -j$(nproc) && cp seqtk /workspace/executable",
    "mkj__dropbear": "cd /workspace/src && autoreconf -fi && ./configure && make -j$(nproc) dropbear && cp dropbear /workspace/executable",
    "cslarsen__jp2a": "cd /workspace/src && autoreconf -fi && ./configure && make -j$(nproc) && cp src/jp2a /workspace/executable",
    "eradman__entr": "cd /workspace/src && ./configure && make -j$(nproc) && cp entr /workspace/executable",
    "samtools__samtools": "cd /workspace/src && autoreconf -fi && ./configure && make -j$(nproc) && cp samtools /workspace/executable",
    "arq5x__bedtools2": "cd /workspace/src && make -j$(nproc) && cp bin/bedtools /workspace/executable",
    "jqlang__jq": "cd /workspace/src && autoreconf -fi && ./configure --with-oniguruma=builtin && make -j$(nproc) && cp jq /workspace/executable",
    "universal-ctags__ctags": "cd /workspace/src && ./autogen.sh && ./configure && make -j$(nproc) && cp ctags /workspace/executable",
    "abishekvashok__cmatrix": "cd /workspace/src && cmake . && make -j$(nproc) && cp cmatrix /workspace/executable",
    "cmatsuoka__figlet": "cd /workspace/src && make -j$(nproc) && cp figlet /workspace/executable",
    "xorg62__tty-clock": "cd /workspace/src && make -j$(nproc) && cp tty-clock /workspace/executable",
    "lymphatus__caesium-clt": "cd /workspace/src && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) && cp caesium /workspace/executable",
    "ip7z__7zip": "cd /workspace/src/CPP/7zip/Bundles/Alone2 && make -f makefile.gcc -j$(nproc) && cp _o/7zz /workspace/executable",
    "zevv__duc": "cd /workspace/src && autoreconf -fi && ./configure && make -j$(nproc) && cp src/duc /workspace/executable",
    "tstack__lnav": "cd /workspace/src && autoreconf -fi && ./configure && make -j$(nproc) && cp src/lnav /workspace/executable",
    "tukaani-project__xz": "cd /workspace/src && autoreconf -fi && ./configure && make -j$(nproc) && cp src/xz/xz /workspace/executable",
    # Rust projects with non-standard binary names
    "burntsushi__ripgrep": "cd /workspace/src && cargo build --release && cp target/release/rg /workspace/executable",
    "burntsushi__xsv": "cd /workspace/src && cargo build --release && cp target/release/xsv /workspace/executable",
    "sharkdp__bat": "cd /workspace/src && cargo build --release && cp target/release/bat /workspace/executable",
    "sharkdp__fd": "cd /workspace/src && cargo build --release && cp target/release/fd /workspace/executable",
    "sharkdp__hexyl": "cd /workspace/src && cargo build --release && cp target/release/hexyl /workspace/executable",
    "sharkdp__hyperfine": "cd /workspace/src && cargo build --release && cp target/release/hyperfine /workspace/executable",
    "sharkdp__pastel": "cd /workspace/src && cargo build --release && cp target/release/pastel /workspace/executable",
    "ajeetdsouza__zoxide": "cd /workspace/src && cargo build --release && cp target/release/zoxide /workspace/executable",
    "dandavison__delta": "cd /workspace/src && cargo build --release && cp target/release/delta /workspace/executable",
    "bootandy__dust": "cd /workspace/src && cargo build --release && cp target/release/dust /workspace/executable",
    "canop__broot": "cd /workspace/src && cargo build --release && cp target/release/broot /workspace/executable",
    "ducaale__xh": "cd /workspace/src && cargo build --release && cp target/release/xh /workspace/executable",
    "ekzhang__bore": "cd /workspace/src && cargo build --release && cp target/release/bore /workspace/executable",
    "rust-lang__mdbook": "cd /workspace/src && cargo build --release && cp target/release/mdbook /workspace/executable",
    "typst__typst": "cd /workspace/src && cargo build --release --bin typst && cp target/release/typst /workspace/executable",
    "svenstaro__miniserve": "cd /workspace/src && cargo build --release && cp target/release/miniserve /workspace/executable",
    "nukesor__pueue": "cd /workspace/src && cargo build --release --bin pueue && cp target/release/pueue /workspace/executable",
    "o2sh__onefetch": "cd /workspace/src && cargo build --release && cp target/release/onefetch /workspace/executable",
    "svenstaro__genact": "cd /workspace/src && cargo build --release && cp target/release/genact /workspace/executable",
    # Go projects needing specific build paths
    "junegunn__fzf": "cd /workspace/src && go build -o /workspace/executable ./",
    "jesseduffield__lazygit": "cd /workspace/src && go build -o /workspace/executable .",
    "johnkerl__miller": "cd /workspace/src && go build -o /workspace/executable ./cmd/mlr",
    "hairyhenderson__gomplate": "cd /workspace/src && go build -o /workspace/executable .",
    "mikefarah__yq": "cd /workspace/src && go build -o /workspace/executable .",
    "filosottile__age": "cd /workspace/src && go build -o /workspace/executable ./cmd/age",
    "ariga__atlas": "cd /workspace/src && go build -o /workspace/executable ./cmd/atlas",
    "dundee__gdu": "cd /workspace/src && go build -o /workspace/executable ./cmd/gdu",
}


def gh_headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h


def get_github_language(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=gh_headers(), timeout=15)
            if r.status_code == 429 or r.status_code == 403:
                wait = int(r.headers.get("Retry-After", 60))
                log.warning("GitHub rate limit hit, sleeping %ds", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("language") or "Unknown"
        except Exception as e:
            log.warning("GitHub API error for %s/%s: %s", owner, repo, e)
            time.sleep(2**attempt)
    return "Unknown"


def fetch_readme(owner: str, repo: str, commit: str) -> str:
    ref = commit or "main"
    for name in ["README.md", "readme.md", "README.rst", "README"]:
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{name}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.text[:8000]
        except Exception:
            pass
    return ""


def parse_attribution(text: str) -> tuple[str, str, str]:
    """Return (owner, repo, commit) from ATTRIBUTION.md content."""
    m = re.search(r"https://github\.com/([^/]+)/([^/]+)/tree/([a-f0-9]{7,40})", text)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.search(r"https://github\.com/([^/\s]+)/([^/\s]+)", text)
    if m:
        return m.group(1), m.group(2).rstrip(")"), ""
    return "", "", ""


def process_task(task_id: str, all_files: list[str], api: HfApi) -> dict | None:
    log.info("Processing %s", task_id)
    prefix = f"{task_id}/"

    # Get test_branches
    test_branches = []
    for f in all_files:
        if f.startswith(f"{prefix}tests/") and f.endswith(".tar.gz"):
            stem = Path(f).stem  # strip .tar.gz
            test_branches.append(stem)

    # Fetch ATTRIBUTION.md
    owner, repo, commit = "", "", ""
    try:
        from huggingface_hub import hf_hub_download

        attr_path = hf_hub_download(
            TESTS_REPO, f"{task_id}/ATTRIBUTION.md", repo_type="dataset", token=HF_TOKEN
        )
        owner, repo, commit = parse_attribution(open(attr_path).read())
    except Exception as e:
        log.warning("ATTRIBUTION fetch failed for %s: %s", task_id, e)
        # Fallback: parse from task_id
        base = task_id.rsplit(".", 1)[0]
        commit = task_id.rsplit(".", 1)[1] if "." in task_id else ""
        owner = base.split("__")[0]
        repo = base.split("__")[1] if "__" in base else base

    if not owner or not repo:
        log.error("Could not parse owner/repo for %s", task_id)
        return None

    # Detect language
    gh_lang = get_github_language(owner, repo)
    lang = LANG_MAP.get(gh_lang, gh_lang.lower() if gh_lang != "Unknown" else "c")
    if lang not in ("go", "rust", "c", "cpp"):
        log.warning("Unsupported lang %s for %s — skipping", lang, task_id)
        return None

    # Fetch README
    readme = fetch_readme(owner, repo, commit)

    # Compile hint
    repo_key = f"{owner}__{repo}"
    compile_hint = COMPILE_HINT_OVERRIDES.get(repo_key) or DEFAULT_COMPILE_HINTS.get(
        lang, "make"
    )

    return {
        "task_id": task_id,
        "language": lang,
        "difficulty": None,
        "readme": readme,
        "docs": "",
        "strings_output": "",
        "nm_output": "",
        "objdump_head": "",
        "file_type": "",
        "binary_size": None,
        "binary_hf_repo": "",
        "binary_hf_filename": "",
        "test_branches": test_branches,
        "compile_hint": compile_hint,
        "example_io": [],
    }


def main():
    api = HfApi(token=HF_TOKEN)

    log.info("Listing all files in %s ...", TESTS_REPO)
    all_files = list(api.list_repo_files(TESTS_REPO, repo_type="dataset"))
    log.info("Total files: %d", len(all_files))

    # Unique task dirs
    task_ids = sorted(
        {
            f.split("/")[0]
            for f in all_files
            if "/" in f and "." in f.split("/")[0] and not f.startswith(".")
        }
    )
    log.info("Unique tasks: %d", len(task_ids))

    # Load existing dataset to preserve any manually-set fields
    existing: dict[str, dict] = {}
    try:
        from datasets import load_dataset as lds

        ds = lds(OUT_REPO, split="train", token=HF_TOKEN)
        for row in ds:
            existing[row["task_id"]] = dict(row)
        log.info("Loaded %d existing rows from %s", len(existing), OUT_REPO)
    except Exception as e:
        log.info("No existing dataset (will create fresh): %s", e)

    rows: list[dict] = []
    skipped: list[str] = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(process_task, tid, all_files, api): tid for tid in task_ids
        }
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                log.error("Task %s failed: %s", tid, e)
                skipped.append(tid)
                continue
            if row is None:
                skipped.append(tid)
                continue
            # Merge: prefer existing binary/analysis fields
            if tid in existing:
                ex = existing[tid]
                for field in (
                    "strings_output",
                    "nm_output",
                    "objdump_head",
                    "file_type",
                    "binary_size",
                    "binary_hf_repo",
                    "binary_hf_filename",
                ):
                    if ex.get(field):
                        row[field] = ex[field]
                if ex.get("compile_hint"):
                    row["compile_hint"] = ex["compile_hint"]
            rows.append(row)

    rows.sort(key=lambda r: r["task_id"])
    log.info("Collected %d rows; skipped %d: %s", len(rows), len(skipped), skipped)

    # Build HF dataset
    ds_new = Dataset.from_list(rows)
    log.info("Pushing %d rows to %s ...", len(ds_new), OUT_REPO)

    # Ensure dataset repo exists
    try:
        api.create_repo(OUT_REPO, repo_type="dataset", private=True, exist_ok=True)
    except Exception:
        pass

    ds_new.push_to_hub(OUT_REPO, token=HF_TOKEN, private=True)
    log.info("Done. Dataset at https://huggingface.co/datasets/%s", OUT_REPO)

    if skipped:
        log.warning(
            "Skipped tasks (non-supported language or parse error): %s", skipped
        )


if __name__ == "__main__":
    main()
