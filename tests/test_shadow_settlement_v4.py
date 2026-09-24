from mcp_gateway import shadow_settlement_v4 as v


def _watch(fid, ts, kickoff, market="Match Winner", selection="home", price=2.0):
    return {
        "fixture_id": fid,
        "generated_at_local": ts,
        "kickoff_local": kickoff,
        "stage": "T-10",
        "classification": "WATCH",
        "league": "League A",
        "home_team": "Home",
        "away_team": "Away",
        "best_market": {
            "market": market,
            "selection": selection,
            "decimal_price": price,
        },
    }


def _final(fid, home=2, away=1):
    return {
        "fixture_id": fid,
        "generated_at_local": "2026-09-20T20:00:00+00:00",
        "kickoff_local": "2026-09-20T18:00:00+00:00",
        "stage": "POSTGAME",
        "classification": "CLOSE",
        "result": {"goals": {"home": home, "away": away}},
    }


def test_shadow_uses_latest_watch_per_fixture_family():
    rows = [
        _watch(1, "2026-09-20T17:00:00+00:00", "2026-09-20T18:00:00+00:00", price=2.1),
        _watch(1, "2026-09-20T17:50:00+00:00", "2026-09-20T18:00:00+00:00", price=2.0),
        _final(1),
    ]
    ledger, summary = v.build(rows)
    assert len(ledger) == 1
    assert ledger[0]["generated_at_local"] == "2026-09-20T17:50:00+00:00"
    assert ledger[0]["shadow_settlement_status"] == "WIN"
    assert ledger[0]["real_wager_assumed"] is False
    assert summary["shadow_rows"] == 1


def test_shadow_does_not_include_bet_or_postkickoff_watch():
    rows = [
        {**_watch(1, "2026-09-20T17:00:00+00:00", "2026-09-20T18:00:00+00:00"), "classification": "BET"},
        _watch(2, "2026-09-20T18:05:00+00:00", "2026-09-20T18:00:00+00:00"),
        _final(1),
        _final(2),
    ]
    ledger, summary = v.build(rows)
    assert ledger == []
    assert summary["shadow_rows"] == 0
    assert summary["bankroll_impact"] == 0.0


def test_shadow_sample_status_uses_independent_fixture_family_decisions():
    rows = []
    for fid in range(1, 21):
        rows.append(_watch(fid, "2026-09-20T17:50:00+00:00", "2026-09-20T18:00:00+00:00"))
        rows.append(_final(fid))
    _, summary = v.build(rows)
    family = summary["by_market_family"]["FT_1X2"]
    assert family["unique_fixtures"] == 20
    assert family["settled"] == 20
    assert family["sample_status"] == "DIRECTIONAL_SHADOW"
    assert summary["production_promotion_allowed"] is False
