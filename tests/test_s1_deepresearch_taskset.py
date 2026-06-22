from verifiers.envs.experimental.composable.tasksets.search.s1_deepresearch.taskset import (
    _build_s1_rows,
)


def test_s1_deepresearch_rows_keep_english_verifiable_meta_only():
    records = [
        {
            "meta": {
                "id": "closed-en",
                "question": "Who?",
                "answer": "Ada",
                "language": "en",
                "type": "Closed-ended Multi-hop Resolution",
            },
            "messages": [{"role": "assistant", "content": "ignored"}],
        },
        {
            "meta": {
                "id": "closed-zh",
                "question": "谁？",
                "answer": "阿达",
                "language": "zh",
                "type": "Closed-ended Multi-hop Resolution",
            },
            "messages": [{"role": "assistant", "content": "ignored"}],
        },
        {
            "meta": {
                "id": "open-en",
                "question": "Write a report",
                "language": "en",
                "type": "Open-ended Exploration",
            },
            "messages": [{"role": "assistant", "content": "ignored"}],
        },
    ]

    rows = _build_s1_rows(
        records,
        dataset_name="ScienceOne-AI/S1-DeepResearch-15k",
        split="train",
        answer_file="/task/answer.txt",
    )

    assert rows == [
        {
            "question": "Who?",
            "answer": "Ada",
            "info": {
                "question": "Who?",
                "answer": "Ada",
                "source_dataset": "ScienceOne-AI/S1-DeepResearch-15k",
                "source_id": "closed-en",
                "split": "train",
                "row_index": 0,
                "language": "en",
                "task_type": "Closed-ended Multi-hop Resolution",
                "answer_file": "/task/answer.txt",
            },
        }
    ]


def test_s1_deepresearch_rows_can_include_all_languages():
    records = [
        {
            "meta": {
                "id": "closed-en",
                "question": "Who?",
                "answer": "Ada",
                "language": "en",
                "type": "Closed-ended Multi-hop Resolution",
            }
        },
        {
            "meta": {
                "id": "closed-zh",
                "question": "谁？",
                "answer": "阿达",
                "language": "zh",
                "type": "Closed-ended Multi-hop Resolution",
            }
        },
    ]

    rows = _build_s1_rows(
        records,
        dataset_name="ScienceOne-AI/S1-DeepResearch-15k",
        split="train",
        answer_file="/task/answer.txt",
        language=None,
    )

    assert [row["info"]["language"] for row in rows] == ["en", "zh"]
