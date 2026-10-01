from starlette.testclient import TestClient

from mcp_gateway import server


def test_v215_signal_ledger_route_is_mounted_on_canonical_server() -> None:
    client = TestClient(server.app)
    response = client.post(
        "/internal/signal-ledger-postgres-v4/build",
        json={"since": "2026-09-22T18:34:09-06:00", "after_event_id": 0, "max_rows": 1},
    )
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"
