"""Dummy end-to-end check for the r2e-gym-v1 taskset against real prime sandboxes.

For each of the first N tasks: provision the task's prime sandbox, run the taskset
``setup`` (venv symlink + pycache clean + test hiding), score the *unmodified* repo
(should be 0.0 — the fail-to-pass test still fails), then apply the reconstructed gold
patch and score again (should be 1.0). Proves load → setup → hide/restore tests →
run_tests.sh → pytest-parse → reward all work end to end.

Run: uv run python deps/verifiers/examples/tasksets/r2e_gym_v1/dummy_rollout.py [N]
"""

import asyncio
import sys

from datasets import load_dataset

import r2e_gym_v1 as r
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def build_tasks(n: int) -> list[r.R2EGymTask]:
    ds = load_dataset(r.DATASET, split="train", streaming=True)
    tasks = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        tasks.append(
            r.R2EGymTask(
                idx=i,
                name=row.get("commit_hash") or f"r2e-{i}",
                instruction=row["problem_statement"],
                image=f"{r.REGISTRY_PREFIX}/{row['docker_image']}",
                workdir=r.REPO_PATH,
                expected_output_json=row["expected_output_json"],
                parsed_commit_content=row.get("parsed_commit_content") or "",
                commit_hash=row.get("commit_hash") or "",
                repo_name=row.get("repo_name") or "",
            )
        )
    return tasks


async def check(task: r.R2EGymTask) -> tuple[float, float]:
    ts = r.R2EGymTaskset.__new__(r.R2EGymTaskset)  # methods only; no config needed
    rt = PrimeRuntime(PrimeConfig(image=task.image, workdir=r.REPO_PATH, labels=["r2e-gym-dummy"]), name=f"r2e-dummy-{task.idx}")
    await rt.start()
    try:
        await ts.setup(task, rt)
        base = await ts.solved(task, rt)       # unmodified repo → expect 0.0
        await ts.apply_gold_patch(task, rt)
        gold = await ts.solved(task, rt)       # gold solution → expect 1.0
        return base, gold
    finally:
        await rt.stop()


async def main() -> None:
    tasks = build_tasks(N)
    print(f"built {len(tasks)} tasks")
    for t in tasks:
        print(f"\n=== task {t.idx} {t.name} ({t.repo_name}) ===\n  image={t.image}")
        try:
            base, gold = await asyncio.wait_for(check(t), timeout=1500)
            verdict = "PASS" if (gold == 1.0 and base == 0.0) else "UNEXPECTED"
            print(f"  base_reward={base}  gold_reward={gold}  -> {verdict}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
