import sys
import types

from datasets import Dataset
import pytest

from verifiers.v1 import Env, Task, Taskset
from verifiers.v1.eval import eval_inputs
from verifiers.v1.utils.taskset_utils import (
    dataset_from_result_typed,
    discover_sibling_dir,
    tasks_from_result_typed,
)


class ReverseTextTask(Task):
    question: str
    answer: str


def test_dataset_from_result_assigns_example_id_to_iterable_records():
    dataset = dataset_from_result_typed(
        [
            {"question": "Reverse abc.", "answer": "cba"},
            {"question": "Reverse xyz.", "answer": "zyx"},
        ],
        ReverseTextTask,
    )

    rows = list(dataset)

    assert [row["example_id"] for row in rows] == [0, 1]
    assert [row["row_id"] for row in rows] == [0, 1]
    assert [row["answer"] for row in rows] == ["cba", "zyx"]
    assert all(len(row["task_id"]) == 24 for row in rows)
    assert {row["task_id"] for row in rows}.isdisjoint({"0", "1"})
    assert rows[0]["task_id"] != rows[1]["task_id"]


def test_dataset_from_result_overwrites_existing_example_id_column():
    raw_dataset = Dataset.from_list(
        [
            {"question": "Reverse abc.", "answer": "cba", "example_id": None},
            {"question": "Reverse xyz.", "answer": "zyx", "example_id": 99},
        ]
    )

    dataset = dataset_from_result_typed(raw_dataset, ReverseTextTask)

    rows = list(dataset)

    assert [row["example_id"] for row in rows] == [0, 1]
    assert [row["row_id"] for row in rows] == [0, 1]
    assert [row["answer"] for row in rows] == ["cba", "zyx"]
    assert all(len(row["task_id"]) == 24 for row in rows)
    assert {row["task_id"] for row in rows}.isdisjoint({"0", "1", "99"})
    assert rows[0]["task_id"] != rows[1]["task_id"]


def test_tasks_from_result_typed_validates_existing_task_objects():
    base_task = Task(prompt="Reverse abc.", row_id=3)

    with pytest.raises(ValueError):
        tasks_from_result_typed([base_task], ReverseTextTask)

    typed_task = ReverseTextTask(
        prompt="Reverse abc.",
        question="Reverse abc.",
        answer="cba",
    )

    assert tasks_from_result_typed([typed_task], ReverseTextTask) == [typed_task]


def test_task_system_prompt_accepts_config_mapping():
    prompt_path = {"messages": [{"role": "system", "content": "Use short answers."}]}

    task = Task(prompt="hello", system_prompt=prompt_path)

    assert task.system_prompt == [{"role": "system", "content": "Use short answers."}]


def test_discover_sibling_dir_returns_empty_existing_dir(tmp_path, monkeypatch) -> None:
    env_dir = tmp_path / "empty_data_env"
    data_dir = env_dir / "data"
    data_dir.mkdir(parents=True)
    module_path = env_dir / "empty_data_env.py"
    module_path.write_text("", encoding="utf-8")

    module = types.ModuleType("empty_data_env")
    module.__file__ = str(module_path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    taskset_type = type(
        "EmptyDataTaskset",
        (),
        {"__module__": module.__name__},
    )

    assert discover_sibling_dir(taskset_type, "data") == data_dir


def test_discover_sibling_dir_can_require_non_empty(tmp_path, monkeypatch) -> None:
    env_dir = tmp_path / "empty_skills_env"
    skills_dir = env_dir / "skills"
    skills_dir.mkdir(parents=True)
    module_path = env_dir / "empty_skills_env.py"
    module_path.write_text("", encoding="utf-8")

    module = types.ModuleType("empty_skills_env")
    module.__file__ = str(module_path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    taskset_type = type(
        "EmptySkillsTaskset",
        (),
        {"__module__": module.__name__},
    )

    assert discover_sibling_dir(taskset_type, "skills", require_non_empty=True) is None

    (skills_dir / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    assert (
        discover_sibling_dir(taskset_type, "skills", require_non_empty=True)
        == skills_dir
    )


def test_taskset_skips_empty_skills_dir(tmp_path, monkeypatch) -> None:
    env_dir = tmp_path / "empty_taskset_skills_env"
    skills_dir = env_dir / "skills"
    skills_dir.mkdir(parents=True)
    module_path = env_dir / "empty_taskset_skills_env.py"
    module_path.write_text("", encoding="utf-8")

    module = types.ModuleType("empty_taskset_skills_env")
    module.__file__ = str(module_path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    taskset_type = type(
        "EmptyTasksetSkills",
        (Taskset,),
        {"__module__": module.__name__},
    )
    taskset = taskset_type()

    assert taskset.get_skills_dir() is None
    assert taskset.get_upload_dirs() == {}

    (skills_dir / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    assert taskset.get_skills_dir() == skills_dir
    assert taskset.get_upload_dirs() == {"skills": skills_dir}


def test_v1_env_eval_inputs_can_shuffle_taskset_dataset() -> None:
    class DemoTaskset(Taskset):
        task_type = ReverseTextTask

        def load_tasks(self, split: str = "train"):
            return [{"question": f"Reverse {i}.", "answer": str(i)} for i in range(6)]

    env = Env(taskset=DemoTaskset())

    inputs = eval_inputs(env, num_examples=3, rollouts_per_example=2, seed=7)
    expected = (
        env.get_eval_dataset().shuffle(seed=7).select(range(3)).repeat(2).to_list()
    )

    assert [row["example_id"] for row in inputs] == [
        row["example_id"] for row in expected
    ]
