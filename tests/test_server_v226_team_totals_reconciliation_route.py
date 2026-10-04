from starlette.testclient import TestClient

from mcp_gateway import server


def test_v226_team_totals_reconciliation_route_is_mounted_on_canonical_server() -> None:
    client = TestClient(server.app)
    response = client.post(
        "/internal/team-totals-capture-signal-reconciliation-v4/build",
        json={"lookback_days": 1},
    )
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"
