"""`vf-validate` end to end on the subprocess runtime, over the `validate-v1` fixture
tasks: the gold check applies the reference answer, the setup check scores the
untouched task, each in its own runtime."""

import json

import pytest
import validate_v1

from verifiers.v1.cli.validate import RESULTS_FILE, SUMMARY_FILE, run_validate
from verifiers.v1.configs.cli.validate import ValidateConfig
from verifiers.v1.utils.paths import CACHE_DIR


@pytest.fixture(autouse=True)
def grades():
    validate_v1.GRADES.clear()
    validate_v1.RUNTIMES.clear()
    yield validate_v1.GRADES


def config(tmp_path, cases, **overrides) -> ValidateConfig:
    return ValidateConfig.model_validate(
        {
            "taskset": {"id": "validate-v1", "cases": cases},
            "runtime": {"type": "subprocess"},
            "output_dir": tmp_path,
            "run": {"name": "run"},
            "rich": False,
            **overrides,
        }
    )


def by_name(rows) -> dict:
    return {row["name"]: row for row in rows}


async def test_setup_check_scores_the_untouched_task(tmp_path, grades) -> None:
    rows = by_name(
        await run_validate(
            config(
                tmp_path,
                ["ok", "partial", "trivial", "unscored", "score"],
                only_setup=True,
            )
        )
    )

    # The scorer ran after finalize, on the state setup left, in setup runtimes only.
    assert sorted(state for _, state in grades) == ["0.0", "0.5", "1.0"]
    assert all(name.startswith("validate-setup-") for name, _ in grades)
    assert {k: (r["reason"], r["valid"]) for k, r in rows.items()} == {
        "ok": ("valid", True),
        "partial": ("valid", True),
        "trivial": ("invalid", False),
        "unscored": ("unchecked", None),
        "score": ("error", False),
    }
    assert rows["ok"]["reward"] == 0.0
    assert rows["partial"]["rewards"] == {"passing": 0.5}
    assert rows["trivial"]["reward"] == 1.0
    assert rows["unscored"]["reward"] is None
    assert rows["score"]["reward"] is None
    assert rows["score"]["rewards"] == {"passing": None}
    assert "grader crashed" in rows["score"]["error"]
    assert rows["score"]["error_type"] == "TaskError"

    summary = json.loads((tmp_path / "run" / SUMMARY_FILE).read_text())
    assert summary["mode"] == "setup"
    assert summary["outcomes"]["invalid"] == 1
    assert summary["outcomes"]["unchecked"] == 1


async def test_default_runs_gold_and_setup_in_independent_runtimes(
    tmp_path, grades
) -> None:
    rows = by_name(await run_validate(config(tmp_path, ["ok", "trivial"])))

    ok, trivial = rows["ok"], rows["trivial"]
    assert ok["mode"] == "all" and ok["reason"] == "valid"
    # Gold wrote a solved state; the setup check still saw its own untouched one.
    assert ok["gold"]["reason"] == "valid"
    assert ok["setup"]["reward"] == 0.0
    runtimes = {name.rsplit("-", 1)[0]: state for name, state in grades}
    assert runtimes == {
        "validate-gold-0": "1.0",
        "validate-setup-0": "0.0",
        "validate-gold-1": "1.0",
        "validate-setup-1": "1.0",
    }
    assert (trivial["reason"], trivial["valid"]) == ("invalid", False)
    assert trivial["gold"]["reason"] == "valid"
    assert trivial["error"] == "setup: passes untouched (reward=1)"

    summary = json.loads((tmp_path / "run" / SUMMARY_FILE).read_text())
    assert summary["checks"]["setup"]["invalid"] == 1
    assert summary["checks"]["gold"]["valid"] == 2


async def test_only_gold_never_scores_untouched(tmp_path, grades) -> None:
    [row] = await run_validate(config(tmp_path, ["trivial"], only_gold=True))

    assert row["mode"] == "gold" and row["reason"] == "valid"
    assert "rewards" not in row
    assert [name.rsplit("-", 1)[0] for name, _ in grades] == ["validate-gold-0"]


async def test_setup_check_timeouts_and_setup_failures(tmp_path, grades) -> None:
    rows = by_name(
        await run_validate(
            config(
                tmp_path,
                ["slow", "setup"],
                only_setup=True,
                timeout={"total": 0.5},
            )
        )
    )

    assert (rows["slow"]["reason"], rows["slow"]["valid"]) == ("timeout", False)
    assert rows["slow"]["reward"] is None
    assert (rows["setup"]["reason"], rows["setup"]["valid"]) == ("error", False)
    assert rows["setup"]["rewards"] == {}
    assert not grades
    # Both runtimes were torn down, though neither check finished.
    assert len(validate_v1.RUNTIMES) == 2
    for name in validate_v1.RUNTIMES:
        assert not (CACHE_DIR / "runtimes" / "subprocess" / name).exists()


async def test_resume_reruns_setup_rows_that_never_scored(tmp_path, grades) -> None:
    cfg = config(tmp_path, ["ok", "trivial"])
    rows = await run_validate(cfg)
    results = tmp_path / "run" / RESULTS_FILE
    # A setup check persisted without untouched scoring (as before it scored) is
    # not final, even though it claims `valid`.
    stale = [
        {**row, "setup": {k: v for k, v in row["setup"].items() if k != "rewards"}}
        if row["name"] == "trivial"
        else row
        for row in rows
    ]
    results.write_text("".join(json.dumps(row) + "\n" for row in stale))
    grades.clear()

    resumed = by_name(await run_validate(cfg.model_copy(update={"resume": True})))

    assert {name.rsplit("-", 2)[0] for name, _ in grades} == {
        "validate-gold",
        "validate-setup",
    }
    assert all(name.split("-")[2] == "1" for name, _ in grades)
    assert resumed["trivial"]["reason"] == "invalid"
    assert resumed["ok"]["reason"] == "valid"
