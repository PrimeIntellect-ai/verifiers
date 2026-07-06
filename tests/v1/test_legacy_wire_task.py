from verifiers.v1.legacy import _to_wire_task


def test_legacy_wire_task_preserves_falsy_answers():
    prompt = [{"role": "user", "content": "answer"}]

    assert _to_wire_task(0, prompt, 0).model_extra == {"answer": 0}
    assert _to_wire_task(1, prompt, False).model_extra == {"answer": False}
    assert _to_wire_task(2, prompt, "").model_extra == {"answer": ""}
    assert _to_wire_task(3, prompt, None).model_extra == {}
