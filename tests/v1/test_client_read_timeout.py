"""A client config's read timeout reaches the eval client's HTTP client."""

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.configs.client import EvalClientConfig


def test_read_timeout_is_configurable():
    config = EvalClientConfig(base_url="http://model.test/v1", api_key_var="TEST_KEY")
    assert EvalClient(config).client.timeout.read == 600.0
    longer = EvalClientConfig(
        base_url="http://model.test/v1", api_key_var="TEST_KEY", read_timeout=1800
    )
    assert EvalClient(longer).client.timeout.read == 1800.0
    assert EvalClient(longer).client.timeout.connect == 5.0
