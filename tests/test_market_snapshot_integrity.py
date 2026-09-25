from mcp_gateway import persistence


class RecordingCursor:
    def __init__(self):
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append((" ".join(str(query).split()), params))


def _event(source: str, resolution_status: str):
    return {
        "event_type": "SOCCER_REFRESH",
        "stage": "T-20",
        "classification": "WATCH",
        "fixture": {
            "fixture_id": 9001,
            "league_id": 1,
            "league": "Test",
            "country": "Test",
            "season": 2026,
            "round": "R1",
            "kickoff": "2026-09-25T03:00:00+00:00",
            "status": "NS",
            "status_long": "Not Started",
            "home_team_id": 1,
            "home_team": "Home",
            "away_team_id": 2,
            "away_team": "Away",
            "venue": None,
            "city": None,
        },
        "market": {
            "source": source,
            "resolution_status": resolution_status,
            "markets": [{
                "bookmaker_id": 1,
                "bookmaker": "Book",
                "market_id": 16,
                "market": "Total - Home",
                "values": [
                    {"selection": "Over", "line": 1.5, "decimal_price": 1.9},
                    {"selection": "Under", "line": 1.5, "decimal_price": 1.9},
                ],
                "provider_update": "2026-09-25T02:10:00+00:00",
            }],
        },
    }


def test_cache_replay_is_not_persisted_as_fresh_market_snapshot():
    cur = RecordingCursor()
    persistence._persist_refresh_event(
        cur,
        {"generated_at_utc": "2026-09-25T02:20:00+00:00"},
        _event("POSTGRES_MARKET_SNAPSHOT_CACHE", "PRICE_CACHE_HIT"),
    )
    sql = "\n".join(query for query, _ in cur.queries)
    assert "INSERT INTO soccer_refresh_events" in sql
    assert "INSERT INTO soccer_market_snapshots" not in sql


def test_real_provider_quote_is_persisted_as_market_snapshot():
    cur = RecordingCursor()
    persistence._persist_refresh_event(
        cur,
        {"generated_at_utc": "2026-09-25T02:20:00+00:00"},
        _event("API_FOOTBALL_ODDS_V3", "PRICE_API_RESOLVED"),
    )
    sql = "\n".join(query for query, _ in cur.queries)
    assert "INSERT INTO soccer_market_snapshots" in sql
