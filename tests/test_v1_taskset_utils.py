import json

from datasets import Dataset

from verifiers.v1.utils.taskset_utils import dataset_from_result


def task_payload(row: dict) -> dict:
    return json.loads(row["info"]["task"])


def test_dataset_from_result_assigns_example_id_to_iterable_records():
    dataset = dataset_from_result(
        [
            {"question": "Reverse abc.", "answer": "cba"},
            {"question": "Reverse xyz.", "answer": "zyx"},
        ],
        "ReverseTextTaskset",
    )

    rows = list(dataset)
    payloads = [task_payload(row) for row in rows]

    assert [row["example_id"] for row in rows] == [0, 1]
    assert [payload["example_id"] for payload in payloads] == [0, 1]
    assert all(len(payload["task_id"]) == 32 for payload in payloads)
    assert {payload["task_id"] for payload in payloads}.isdisjoint({"0", "1"})


def test_dataset_from_result_overwrites_existing_example_id_column():
    raw_dataset = Dataset.from_list(
        [
            {"question": "Reverse abc.", "answer": "cba", "example_id": None},
            {"question": "Reverse xyz.", "answer": "zyx", "example_id": 99},
        ]
    )

    dataset = dataset_from_result(raw_dataset, "ReverseTextTaskset")

    rows = list(dataset)
    payloads = [task_payload(row) for row in rows]

    assert [row["example_id"] for row in rows] == [0, 1]
    assert [payload["example_id"] for payload in payloads] == [0, 1]
    assert all(len(payload["task_id"]) == 32 for payload in payloads)
    assert {payload["task_id"] for payload in payloads}.isdisjoint({"0", "1", "99"})
