from verifiers.scripts.build_search_dataset import (
    normalize_openseeker_row,
    normalize_redsearcher_row,
    normalize_s1_record,
)


def test_normalize_openseeker_row():
    row = {
        "question": " Question? ",
        "answer": " Answer ",
        "number of tool calls": 7,
        "trajectory correctness": "Correct",
    }

    normalized = normalize_openseeker_row(row, row_index=3)

    assert normalized == {
        "question": "Question?",
        "answer": "Answer",
        "info": {
            "source": "openseeker",
            "language": "en",
            "source_task_type": "binary_semantic_answer",
        },
    }


def test_normalize_redsearcher_row():
    row = {
        "problem": " Problem? ",
        "answer": " Answer ",
        "difficulty": " hard ",
    }

    normalized = normalize_redsearcher_row(row, row_index=4)

    assert normalized == {
        "question": "Problem?",
        "answer": "Answer",
        "info": {
            "source": "redsearcher",
            "language": "en",
            "source_task_type": "text_rl_query",
        },
    }


def test_normalize_s1_record():
    record = {
        "meta": {
            "id": "abc",
            "question": " Question? ",
            "answer": " Answer ",
            "language": "en",
            "type": "Closed-ended Multi-hop Resolution",
        },
        "messages": [{"role": "assistant", "content": "ignored"}],
    }

    normalized = normalize_s1_record(record, row_index=5)

    assert normalized == {
        "question": "Question?",
        "answer": "Answer",
        "info": {
            "source": "s1_deepresearch",
            "language": "en",
            "source_task_type": "Closed-ended Multi-hop Resolution",
        },
    }
