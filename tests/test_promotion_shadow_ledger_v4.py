from mcp_gateway import promotion_shadow_ledger_v4 as v


CAP_REASON = "Tier A/S blocked until advanced-metric layer is verified; automated model is limited goal-rate baseline."


def _tick(ts, fid, stage, decisions=None, result=None):
    event = {
        "stage": stage,
        "fixture": {
            "fixture_id": fid,
            "kickoff": "2026-09-20T18:00:00+00:00",
            "status": "NS" if stage != "POSTGAME" else "FT",
            "home_team": "Home",
            "away_team": "Away",
            "home_team_id": 1,
            "away_team_id": 2,
        },
    }
    if decisions is not None:
        event["market_decision"] = {"decisions": decisions}
    if result is not None:
        event["result"] = {"goals": result}
    return {"generated_at_utc": ts, "events": [event]}


def _decision(selection="home", tier="A", reasons=None, edge=8.0, price=2.0):
    return {
        "classification": "WATCH",
        "tier": tier,
        "family": "1X2",
        "market": "Match Winner",
        "selection": selection,
        "decimal_price": price,
        "prob_edge_pp": edge,
        "estimated_ev": 0.10,
        "p_raw": 0.55,
        "p_shrunk": 0.52,
        "p_market_fair": 0.44,
        "reasons": list(reasons if reasons is not None else [CAP_REASON]),
    }


def test_only_clean_tier_cap_watch_is_eligible():
    assert v._clean_tier_cap_watch(_decision()) is True
    assert v._clean_tier_cap_watch(_decision(reasons=[CAP_REASON, "extra gate"])) is False
    assert v._clean_tier_cap_watch(_decision(tier="B")) is False
    assert v._clean_tier_cap_watch({**_decision(), "classification": "BET"}) is False


def test_builder_excludes_rechecks_and_keeps_latest_clean_fixture_family_candidate():
    ticks = [
        _tick("2026-09-20T17:20:00+00:00", 1, "T-40", [_decision(edge=6.0, price=2.1)]),
        _tick("2026-09-20T17:50:00+00:00", 1, "T-10", [_decision(edge=9.0, price=2.0)]),
        _tick(
            "2026-09-20T17:55:00+00:00",
            2,
            "T-10",
            [_decision(reasons=["1X2 raw-vs-market gap >=12pp requires RECHECK."])],
        ),
        _tick("2026-09-20T20:00:00+00:00", 1, "POSTGAME", result={"home": 2, "away": 1}),
        _tick("2026-09-20T20:00:00+00:00", 2, "POSTGAME", result={"home": 2, "away": 1}),
    ]
    ledger, summary = v.build_from_ticks(ticks)
    assert len(ledger) == 1
    assert ledger[0]["fixture_id"] == 1
    assert ledger[0]["stage"] == "T-10"
    assert ledger[0]["settlement_status"] == "WIN"
    assert summary["rows"] == 1
    assert summary["real_wagers_assumed"] is False


def test_builder_keeps_one_best_candidate_per_family_at_same_timestamp():
    ticks = [
        _tick(
            "2026-09-20T17:50:00+00:00",
            1,
            "T-10",
            [
                _decision(selection="home", edge=6.0, price=2.0),
                _decision(selection="away", edge=9.0, price=3.0),
            ],
        ),
        _tick("2026-09-20T20:00:00+00:00", 1, "POSTGAME", result={"home": 0, "away": 1}),
    ]
    ledger, _ = v.build_from_ticks(ticks)
    assert len(ledger) == 1
    assert ledger[0]["selection"] == "away"
    assert ledger[0]["settlement_status"] == "WIN"
    assert ledger[0]["roi_units"] == 2.0
