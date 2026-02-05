from verifiers.scripts.tui import format_info_for_details


def test_format_info_for_details_handles_dict() -> None:
    info = {"status": "ok", "attempt": 2}

    rendered = format_info_for_details(info)

    assert rendered == '{"status": "ok","attempt": 2}'


def test_format_info_for_details_parses_json_string() -> None:
    info = '{"status":"ok","nested":{"value":1}}'

    rendered = format_info_for_details(info)

    assert rendered == '{"status": "ok","nested": {"value": 1}}'


def test_format_info_for_details_truncates_large_content() -> None:
    info = {"payload": [f"line-{i}" for i in range(50)]}

    rendered = format_info_for_details(info, max_chars=80, max_lines=1)

    assert rendered.endswith("lines total)")
    assert "(truncated;" in rendered
    assert "\n" not in rendered
