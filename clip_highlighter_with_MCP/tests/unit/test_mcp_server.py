from src.mcp.server import mcp


def test_mcp_server_is_configured() -> None:
    assert mcp is not None
