"""Tests for Cloudflare tunnel utilities."""

from unittest.mock import MagicMock, patch

import pytest

from verifiers.utils import tunnel_utils


def test_tunnel_pool_requires_name_and_url():
    with pytest.raises(ValueError, match="cloudflared_tunnel_name"):
        tunnel_utils.TunnelPool(port=8766, cloudflared_tunnel_name="rlm")

    with pytest.raises(ValueError, match="cloudflared_tunnel_name"):
        tunnel_utils.TunnelPool(
            port=8766, cloudflared_tunnel_url="https://tunnel.example.com"
        )


@pytest.mark.asyncio
async def test_tunnel_pool_uses_named_tunnel():
    with patch.object(
        tunnel_utils, "start_cloudflared_named_tunnel", autospec=True
    ) as start_named:
        start_named.return_value = (
            "https://tunnel.example.com",
            MagicMock(),
        )
        pool = tunnel_utils.TunnelPool(
            port=8766,
            cloudflared_tunnel_name="rlm",
            cloudflared_tunnel_url="https://tunnel.example.com/",
        )

        url = await pool.get_tunnel_url(active_rollout_count=1)

        assert url == "https://tunnel.example.com"
        start_named.assert_called_once_with(
            "rlm",
            "https://tunnel.example.com",
            pool.max_wait_seconds,
        )
