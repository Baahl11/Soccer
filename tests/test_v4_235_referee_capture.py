from mcp_gateway import automation


def test_compact_fixture_preserves_provider_referee_assignment():
    row = {
        "fixture": {
            "id": 123,
            "date": "2026-10-05T18:00:00+00:00",
            "timestamp": 1791223200,
            "referee": "Jane Doe, Mexico",
            "status": {"short": "NS", "long": "Not Started", "elapsed": None},
            "venue": {"name": "Test Stadium", "city": "Puebla"},
        },
        "league": {"id": 1, "name": "Test League", "country": "Mexico", "season": 2026, "round": "1"},
        "teams": {
            "home": {"id": 10, "name": "Home"},
            "away": {"id": 20, "name": "Away"},
        },
        "goals": {"home": None, "away": None},
        "score": {},
    }

    compact = automation._compact_fixture(row)

    assert compact["fixture_id"] == 123
    assert compact["referee"] == "Jane Doe, Mexico"
