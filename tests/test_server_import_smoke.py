def test_server_import_smoke():
    import mcp_gateway.server as server
    assert server is not None
    assert getattr(server, "app", None) is not None
