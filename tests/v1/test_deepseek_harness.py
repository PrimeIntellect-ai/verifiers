from verifiers.v1.harnesses.deepseek_harness.harness import (
    INSTALL,
    _dsh_server_name,
)


def test_install_script_supports_alpine_build_dependencies():
    assert "command -v apk" in INSTALL
    assert "apk add --no-cache python3 make g++" in INSTALL


def test_dsh_server_name_preserves_bare_tool_namespace():
    assert _dsh_server_name("") == ""


def test_dsh_server_name_hashes_sanitized_or_long_names():
    assert _dsh_server_name("server.with.dots").startswith("server_with_dots_")

    long_name = "a" * 40
    normalized = _dsh_server_name(long_name)

    assert normalized.startswith("a" * 23)
    assert len(normalized) == 32
