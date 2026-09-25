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



def test_real_provider_research_sidecar_is_persisted_without_entering_canonical_markets():
    cur = RecordingCursor()
    event = _event("API_FOOTBALL_ODDS_V3", "PRICE_API_RESOLVED")
    event["market"]["markets"] = []
    event["market"]["research_cards_props_markets"] = [{
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 201,
        "market": "Player Shots",
        "values": [
            {"selection": "Player A Over 2.5", "price": "1.95", "parsed_line": 2.5},
        ],
        "provider_update": "2026-09-25T02:10:00+00:00",
        "research_only": True,
        "research_family": "PLAYER_PROPS",
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
    }]

    persistence._persist_refresh_event(
        cur,
        {"generated_at_utc": "2026-09-25T02:20:00+00:00"},
        event,
    )

    market_inserts = [
        params for query, params in cur.queries
        if "INSERT INTO soccer_market_snapshots" in query
    ]
    assert len(market_inserts) == 1
    assert market_inserts[0][6] == "Player Shots"


def test_cache_replay_research_sidecar_is_not_persisted_as_fresh_snapshot():
    cur = RecordingCursor()
    event = _event("POSTGRES_MARKET_SNAPSHOT_CACHE", "PRICE_CACHE_HIT")
    event["market"]["markets"] = []
    event["market"]["research_cards_props_markets"] = [{
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 200,
        "market": "Total Yellow Cards",
        "values": [{"selection": "Over 4.5", "price": "1.90", "parsed_line": 4.5}],
        "provider_update": "2026-09-25T02:10:00+00:00",
        "research_only": True,
        "research_family": "CARDS",
    }]

    persistence._persist_refresh_event(
        cur,
        {"generated_at_utc": "2026-09-25T02:20:00+00:00"},
        event,
    )

    sql = "\n".join(query for query, _ in cur.queries)
    assert "INSERT INTO soccer_refresh_events" in sql
    assert "INSERT INTO soccer_market_snapshots" not in sql
