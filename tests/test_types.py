import verifiers as vf


def test_system_message_from_path_reads_file_verbatim(tmp_path):
    prompt = "You are precise.\n\nPreserve trailing newline.\n"
    path = tmp_path / "system_prompt.txt"
    path.write_text(prompt, encoding="utf-8")

    message = vf.SystemMessage.from_path(path)

    assert message == {"role": "system", "content": prompt}
