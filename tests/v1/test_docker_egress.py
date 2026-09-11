from verifiers.v1.runtimes.docker.egress import HOST_ALIAS, NetworkPolicy


def test_framework_http_route_allows_exact_connect_tunnel() -> None:
    policy = NetworkPolicy(
        allow=[], block=["*"], routes=[f"http://{HOST_ALIAS}:43123"]
    )

    assert policy.permits("https", HOST_ALIAS, 43123, connect=True)
    assert not policy.permits("https", HOST_ALIAS, 43124, connect=True)
    assert not policy.permits("https", "other.internal", 43123, connect=True)
