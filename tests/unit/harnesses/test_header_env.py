"""A shared tool server's own header variables reach one rollout's harness, never the agent's."""

from verifiers.v1.harnesses.rlm.harness import RLMHarness, RLMHarnessConfig


def test_a_rollouts_server_headers_join_the_harness_config_and_read_the_hosts_env(
    monkeypatch,
):
    config = RLMHarnessConfig(
        id="rlm",
        mcp_header_env={"docs": {"X-Key": "DOCS_KEY"}, "world": {"X-Run": "RUN_ID"}},
    )
    harness = RLMHarness(config)
    made = harness.with_mcp_header_env({"world": {"Authorization": "WORLD_TOKEN_B1"}})

    assert made is not harness and type(made) is RLMHarness
    assert harness.config.mcp_header_env["world"] == {"X-Run": "RUN_ID"}
    assert made.config.mcp_header_env == {
        "docs": {"X-Key": "DOCS_KEY"},
        "world": {"X-Run": "RUN_ID", "Authorization": "WORLD_TOKEN_B1"},
    }
    monkeypatch.setenv("WORLD_TOKEN_B1", "Bearer b1")
    monkeypatch.setenv("RUN_ID", "r7")
    assert made.config.resolve_mcp_headers({"world": "http://w/mcp"}) == {
        "world": {"X-Run": "r7", "Authorization": "Bearer b1"}
    }
    assert "Bearer b1" not in made.config.model_dump_json(), "only names are configured"
