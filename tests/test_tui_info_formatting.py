import json

from verifiers.scripts.tui import format_info_for_details


def test_format_info_for_details_handles_dict() -> None:
    info = {"status": "ok", "attempt": 2}

    rendered = format_info_for_details(info)

    assert rendered == json.dumps(info, ensure_ascii=False, indent=2)


def test_format_info_for_details_parses_json_string() -> None:
    info = '{"status":"ok","nested":{"value":1}}'

    rendered = format_info_for_details(info)

    assert rendered == json.dumps(
        {"status": "ok", "nested": {"value": 1}},
        ensure_ascii=False,
        indent=2,
    )


def test_format_info_for_details_truncates_large_content() -> None:
    info = {"payload": [f"line-{i}" for i in range(200)]}

    rendered = format_info_for_details(info, max_chars=80)

    assert rendered.endswith("chars total)")
    assert "(truncated;" in rendered
